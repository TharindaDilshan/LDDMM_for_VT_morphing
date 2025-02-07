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
    def f(ODESystem, x0, nt, deltat=1.0):
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


def Flow(x0, p0, q0, K, deltat=1.0, Integrator=RalstonIntegrator()):
    HS = HamiltonianSystem(K)

    def FlowEq(x, p, q):
        return (K(x, q, p),) + HS(p, q)

    return Integrator(FlowEq, (x0, p0, q0), deltat)[0]

def LDDMMloss(K, dataloss, gamma=0):
    def loss(p0, q0):
        p, q = Shooting(p0, q0, K)[-1]
        
        return gamma * Hamiltonian(K)(p0, q0) + dataloss(q)

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
        
    def flow(self, x0):
        x0 = torch.tensor(x0, dtype=torchdtype, device=torchdeviceId)

        return Flow(x0, self.init_mom, self.init_pos, self.kernel)
    
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