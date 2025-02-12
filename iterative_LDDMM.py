import numpy as np
import torch
import trimesh
import scipy.io
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import grad

from plot import surf_to_pv, PlotResSurf, plot_surf, plot_pts, PlotRes3D

plt.rcParams['figure.figsize'] = [16, 10]

# Optional:

# KeOps library for kernel convolutions -- useless for small datasets
#!pip install pykeops 
use_keops = True # use of 

# pyvista for displaying 3D graphics
#!pip install pyvista[all]
use_pyvista = True

if use_keops:
    from pykeops.torch import LazyTensor

# Kernel functions

def GaussKernel(sigma):
    oos2 = 1/sigma**2

    def K(x,y,b):
        x,y = x[:,None,:],y[None,:,:]
        if use_keops:
            x,y = LazyTensor(x),LazyTensor(y)

        return (-oos2*((x-y)**2).sum(dim=2)).exp()@b
    
    return K

# defines a composite kernel that combines a Gaussian (radial basis function) kernel with a 
# linear kernel, tailored for use with vectors representing (measure similarity) geometric  
# entities (such as normals or directions) in addition to positions.
def GaussLinKernel(sigma, lib="keops"):
    oos2 = 1/sigma**2
    
    def K(x,y,u,v,b):
        # calculates the similarity based on the Euclidean distance between points  
        # x and y in a high-dimensional space
        Kxy = torch.exp(-oos2*torch.sum((x[:,None,:]-y[None,:,:])**2,dim=2))
        # computes the squared dot product between corresponding vectors u and v 
        # associated with points x and y, respectively. This part captures the 
        # similarity in directions (e.g., surface normals) at the points
        Sxy = torch.sum(u[:,None,:]*v[None,:,:],dim=2)**2
        
        # composite kernel
        return (Kxy*Sxy)@b
    
    return K

# Ordinary Differential Equations (ODEs) solver and Optimizer

# numerical integrator for solving ordinary differential equations (ODEs)
# solves the Hamiltonian system dynamics during the shape deformation process
def RalstonIntegrator(nt=10):
    # nt:  number of time steps to divide the integration interval into
    def f(ODESystem,x0,deltat=1.0):
        # x0: initial conditions (p0, q0)
        # deltat: total integration time
        x = tuple(map(lambda x:x.clone(), x0))
        dt = deltat/nt
        for i in range(nt):
            # computes the derivatives (system dynamics) of the current state x
            xdot = ODESystem(*x)
            # temporary state xi = x + (2*dt/3)*xdot
            # predicts the system state after two-thirds of the time step, 
            # guided by the initial derivative
            xi = tuple(map(lambda x,xdot:x+(2*dt/3)*xdot,x,xdot))
            # derivatives at this intermediate state xi
            xdoti = ODESystem(*xi)
            # final state for the time step is computed using the 
            # combination of the initial derivative and the intermediate derivative
            # weighted average of the initial and intermediate derivatives to estimate the next state
            x = tuple(map(lambda x,xdot,xdoti:x+(.25*dt)*(xdot+3*xdoti),x,xdot,xdoti))
            
        return x
    
    return f

# function to minimize the loss
def Optimize(loss,args,niter=5):
    # loss: function to compute the loss given the current set of parameters (args)
    # args: p0
    optimizer = torch.optim.LBFGS(args)
    losses = []
    print('performing optimization...')
    # repeatedly adjusts the parameters to minimize the loss function
    for i in range(niter):
        print("iteration ",i+1,"/",niter)
        def closure():
            # reset optimizer gradients to 0 (previous data doesn't affect the current update)
            optimizer.zero_grad()
            # compute loss
            L = loss(*args)
            losses.append(L.item())
            # backprop to compute the gradients of the loss w.r.t the parameters
            L.backward()
            
            return L
        
        # update optimizer parameters based on the loss
        optimizer.step(closure)
        
    print("Done.")

    return args, losses

# LDDMM algorithm

# function of momentum p and position q to represents the total energy of the system
# measures the energy associated with the deformation, using the kernel K 
# to mediate the influence of points on each other.
def Hamiltonian(K):
    def H(p,q):
        return .5*(p*K(q,q,p)).sum()
        
    return H

# builds the Hamiltonian system that needs to be solved during the "shooting" process
# calculates the gradients of the Hamiltonian with respect to p and q, 
# which represent the rates of change of these quantities
# the system is defined by -Gq, Gp (gradients)
def HamiltonianSystem(K):
    H = Hamiltonian(K)
    
    def HS(p,q):
        Gp,Gq = grad(H(p,q),(p,q), create_graph=True)
        
        return -Gq,Gp
        
    return HS

# integrates the Hamiltonian system over time to find the end state (p, q) 
# starting from initial conditions (p0, q0)
def Shooting(p0, q0, K, deltat=1.0, Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystem(K),(p0, q0), deltat)

# intégration des équations de flot
def Flow(x0, p0, q0, K, deltat=1.0, Integrator=RalstonIntegrator()):
    HS = HamiltonianSystem(K)
    
    def FlowEq(x,p,q):
        return (K(x,q,p),)+HS(p,q)
        
    return Integrator(FlowEq,(x0,p0,q0),deltat)[0]

# defines the loss function to be minimized, 
# combining the Hamiltonian (energy of the deformation) 
# and a data attachment loss 
def LDDMMloss(q0,K,dataloss,gamma=0.):
    # dataloss: measures the discrepancy between the deformed source shape and the target shape
    # q0: initial configuration of the source shape
    # K: kernel function
    # gamma: regularization parameter
    def loss(p0):
        # finding p0 that minimizes the loss
        # p, q: final momentum and deformed shape after the shooting process
        p,q = Shooting(p0,q0,K)
        # Hamiltonian(K): computes the energy of the initial configuration q0 with the initial momentum p0. 
        # This represents the energy required to deform the shape regularized by gamma
        # dataloss(q): computes the mismatch between the final deformed shape q and the target
        return gamma * Hamiltonian(K)(p0,q0) + dataloss(q)
    
    return loss

# Data attachment functions

# data attachment function for triangulated surfaces, varifold model
# focuses on matching geometric features like positions and normals without requiring explicit 
# point correspondence. this function is useful when the exact matching of points between shapes 
# is infeasible or not desired
def lossVarifoldSurf(FS,VT,FT,K):
    # VT: coordinates of the points of the target surface
    # FS,FT: indices of the triangles of the source and target surfaces
    # K: varifold kernel
    
    # compute centers (C), normals (N), and areas (L) of the triangles for a given surface
    # based on vertices V and faces F
    def CompCLNn(F,V):
        # V0, V1, V2: vertices of each triangle
        # center(C) of each triangle is calculated as the average of its vertices
        # normal(N): taking the cross product of two edges of the triangle, which is then normalized
        # area(L): length of the normal vector before normalization
        V0, V1, V2 = V.index_select(0,F[:,0]), V.index_select(0,F[:,1]), V.index_select(0,F[:,2])
        C, N = .5*(V0+V1+V2), .5*torch.linalg.cross(V1-V0,V2-V0)
        L = (N**2).sum(dim=1)[:,None].sqrt()
        
        return C,L,N/L
    
    CT,LT,NTn = CompCLNn(FT,VT)
    # self-interaction term for the target surface using the varifold kernel  K
    cst = (LT*K(CT,CT,NTn,NTn,LT)).sum()
    
    # calculates the varifold distance between the source and target surfaces
    
    # loss is formulated as the sum of the source self-interaction and the 
    # target self-interaction (cst), minus twice the cross term (interaction between their geometric features)
    # the intuition is to minimize the difference in geometric features (both position and direction of normals) 
    # between the source and target surfaces, thereby aligning them
    def loss(VS):
        CS,LS,NSn = CompCLNn(FS,VS)
        
        return cst + (LS*K(CS,CS,NSn,NSn,LS)).sum() - 2*(LS*K(CS,CT,NSn,NTn,LT)).sum()
    
    return loss

# Function to load mesh and convert to tensors
def load_mesh_as_tensors(filepath):
    # Load the mesh
    mesh = trimesh.load(filepath, process=False)
    
    # Convert vertices and faces to PyTorch tensors
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)
    
    return vertices, faces
