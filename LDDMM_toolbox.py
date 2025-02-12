import torch
from torch.autograd import grad
from pykeops.torch import Vi, Vj
import time
import numpy as np

use_cuda = torch.cuda.is_available()
torchdeviceId = torch.device("cuda:0") if use_cuda else "cpu"
torchdtype = torch.float32

def GaussKernel(sigma):
    x, y, b = Vi(0, 3), Vj(1, 3), Vj(2, 3)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp()

    return (K * b).sum_reduction(axis=1)

# Define Energy Distance kernel :math:`(K(x,y))_i = \sum_j (-\|x_i-y_j\|)`
def EnergyKernel():
    x, y, b = Vi(0, 3), Vj(1, 3), Vj(2, 3)
    D = x.sqdist(y).sqrt()

    return (-D).sum_reduction(axis=1)

def GaussLinKernel(sigma):
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp() * (u * v).sum() ** 2

    return (K * b).sum_reduction(axis=1)

# Custom ODE solver, for ODE systems which are defined on tuples
def RalstonIntegrator():
    def f(ODESystem, x0, nt=51, deltat=1.0):
        x = tuple(map(lambda x: x.clone(), x0))
        dt = deltat / nt
        l = [x]
        for i in range(nt):
            xdot = ODESystem(*x)
            xi = tuple(map(lambda x, xdot: x + (2 * dt / 3) * xdot, x, xdot))
            xdoti = ODESystem(*xi)
            x = tuple(
                map(
                    lambda x, xdot, xdoti: x + (0.25 * dt) * (xdot + 3 * xdoti),
                    x,
                    xdot,
                    xdoti,
                )
            )
            l.append(x)

        return l

    return f

def Hamiltonian(K):
    def H(p, q):
        return 0.5 * (p * K(q, q, p)).sum()

    return H

def HamiltonianSystem(K):
    H = Hamiltonian(K)

    def HS(p, q):
        Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
        return -Gq, Gp

    return HS

def Shooting(p0, q0, K, nt=10, Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystem(K), (p0, q0), nt)


def Flow(x0, p0, q0, K, nt=10, deltat=1.0, Integrator=RalstonIntegrator()):
    HS = HamiltonianSystem(K)

    def FlowEq(x, p, q):
        return (K(x, q, p),) + HS(p, q)

    return Integrator(FlowEq, (x0, p0, q0), nt, deltat)[0]

def LDDMMloss(K, dataloss, gamma=0):
    def loss(p0, q0):
        p, q = Shooting(p0, q0, K)[-1]
        
        return gamma * Hamiltonian(K)(p0, q0) + dataloss(q)

    return loss

def SmallDefloss(K, dataloss, gamma=0.3):
    def loss(p0, q0):
        v = K(q0,q0,p0)
        q = q0 + v
        return gamma * (v*p0).sum() + dataloss(q)

    return loss

# Measure data attachment loss for points
# VT: vertices coordinates of target points, K: kernel
def lossMeas(VT, K):
    nT = VT.shape[0]
    cst = K(VT, VT).sum()/nT**2

    def loss(VS):
        nS = VS.shape[0]
        
        return cst + K(VS, VS).sum()/nS**2 - 2*K(VS, VT).sum()/(nS*nT)
        
    return loss

class LDDMM_def:
    def __init__(self, p0, q0, Kv):
        self.init_mom = p0
        self.init_pos = q0
        self.kernel = Kv
        
    def shoot(self):
        return Shooting(self.init_mom, self.init_pos, self.kernel)
        
    def flow(self, x0, nt=10):
        x0 = torch.tensor(x0, dtype=torchdtype, device=torchdeviceId)

        return Flow(x0, self.init_mom, self.init_pos, self.kernel, nt)
    
class SmallDef_def:
    def __init__(self,p0,q0,Kv):
        self.init_mom = p0
        self.init_pos = q0
        self.kernel = Kv
    def shoot(self,t=1):
        q0, p0 = self.init_pos, self.init_mom
        res = q0 + t*self.kernel(q0,q0,p0)
        return np.array(res.data.cpu())
    def flow(self, x0, t=1):
        x0 = torch.tensor(x0, dtype=torchdtype, device=torchdeviceId)
        q0, p0 = self.init_pos, self.init_mom
        res = x0 + t*self.kernel(x0,q0,p0)
        return np.array(res.data.cpu())
    
def Optimize(loss,x):
    optimizer = torch.optim.LBFGS([x], max_eval=10, max_iter=10)
    #print("performing optimization...")
    start = time.time()

    def closure():
        optimizer.zero_grad()
        L = loss(x)
        #print("loss", L.detach().cpu().numpy())
        L.backward()

        return L
    
    for i in range(10):
        optimizer.step(closure)

    #print("Optimization (L-BFGS) time: ", round(time.time() - start, 2), " seconds")

def LDDMM_Optimize(q0, dataloss, sigma):
    # Define LDDMM functional
    sigma = torch.tensor([sigma], dtype=torchdtype, device=torchdeviceId)
    Kv = GaussKernel(sigma=sigma)
    loss = LDDMMloss(Kv, dataloss)

    # initialize momentum vectors
    p0 = torch.zeros(q0.shape, dtype=torchdtype, device=torchdeviceId, requires_grad=True)
    
    # Perform optimization
    Optimize(lambda p0 : loss(p0,q0), p0)
            
    return LDDMM_def(p0,q0,Kv)

def SmallDef_Optimize(q0,dataloss,sigma):
    
    #####################################################################
    # Define SmallDef functional
    sigma = torch.tensor([sigma], dtype=torchdtype, device=torchdeviceId)
    Kv = GaussKernel(sigma=sigma)
    loss = SmallDefloss(Kv, dataloss)

    # initialize momentum vectors
    p0 = torch.zeros(q0.shape, dtype=torchdtype, device=torchdeviceId, requires_grad=True)
    
    # Perform optimization
    Optimize(lambda p0 : loss(p0,q0), p0)
            
    return SmallDef_def(p0,q0,Kv)

def MatchPoints(VS, VT, sigma=20, method=LDDMM_Optimize):        
    VS = torch.tensor(VS, dtype=torchdtype, device=torchdeviceId)
    VT = torch.tensor(VT, dtype=torchdtype, device=torchdeviceId)
    q0 = VS.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
    VT = VT.clone().detach().to(dtype=torchdtype, device=torchdeviceId)
    dataloss = lossMeas(VT, EnergyKernel())
    
    return method(q0, dataloss, sigma)