import torch
import copy
import LDDMM_toolbox
import importlib
importlib.reload(LDDMM_toolbox)

def MatchPoints(VS, VT, sigma=20, method=LDDMM_toolbox.SmallDef_Optimize):        
    VS = torch.tensor(VS, dtype=LDDMM_toolbox.torchdtype, device=LDDMM_toolbox.torchdeviceId)
    VT = torch.tensor(VT, dtype=LDDMM_toolbox.torchdtype, device=LDDMM_toolbox.torchdeviceId)
    q0 = VS.clone().detach().to(dtype=LDDMM_toolbox.torchdtype, device=LDDMM_toolbox.torchdeviceId).requires_grad_(True)
    VT = VT.clone().detach().to(dtype=LDDMM_toolbox.torchdtype, device=LDDMM_toolbox.torchdeviceId)
    dataloss = LDDMM_toolbox.lossMeas(VT, LDDMM_toolbox.EnergyKernel())
    
    return method(q0, dataloss, sigma)

def deform_mesh(mesh, phi):
    phi_mesh = copy.deepcopy(mesh)
    phi_mesh["V"] = phi.flow(mesh["V"])
    if "points" in mesh:
        phi_mesh["points"] = phi.flow(mesh["points"])
        
    return phi_mesh