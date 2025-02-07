import torch
import copy
from LDDMM_toolbox import LDDMM_Optimize, lossMeas, EnergyKernel, torchdeviceId, torchdtype

def MatchPoints(VS, VT, sigma=20, method=LDDMM_Optimize):        
    VS = torch.tensor(VS, dtype=torchdtype, device=torchdeviceId)
    VT = torch.tensor(VT, dtype=torchdtype, device=torchdeviceId)
    q0 = VS.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
    VT = VT.clone().detach().to(dtype=torchdtype, device=torchdeviceId)
    dataloss = lossMeas(VT, EnergyKernel())
    
    return method(q0, dataloss, sigma)

def deform_mesh(mesh, phi):
    phi_mesh = copy.deepcopy(mesh)
    phi_mesh["V"] = phi.flow(mesh["V"])
    if "points" in mesh:
        phi_mesh["points"] = phi.flow(mesh["points"])
        
    return phi_mesh