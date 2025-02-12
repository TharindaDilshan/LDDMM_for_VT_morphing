import torch

# Cauchy (K(x,y)b)_i = sum_j (1/(1+|xi-yj|^2/sigma^2))bj
def CauchyKernel(sigma):
    oos2 = 1/sigma**2

    def K(x,y,b):

        return (1/(1+oos2*torch.sum((x[:,None,:]-y[None,:,:])**2,dim=2)))@b
    
    return K

# kernel with multiple sigmas
def SumKernel(*kernels):
    def K(*args):

        return sum(k(*args) for k in kernels)
    
    return K

##### Data attachment functions #####

# data attachment function for landmarks
def losslmk(z):
    def loss(q):
        return ((q-z)**2).sum()
    return loss

# data attachment function for point clouds via the measurement model
def lossmeas(z,Kw):
    nz = z.shape[0]
    wz = torch.ones(nz,1)
    cst = (1/nz**2)*Kw(z,z,wz).sum()
    def loss(q):
        nq = q.shape[0]
        wq = torch.ones(nq,1)
        return cst + (1/nq**2)*Kw(q,q,wq).sum() + (-2/(nq*nz))*Kw(q,z,wz).sum()
    return loss

# data attachment function for point clouds via regularized optimal transport
# (requires geomloss package)
def loss_OT(z):
    loss_ = SamplesLoss()
    nz = z.shape[0]
    wz = torch.ones(nz,1)
    def loss(q):
        nq = q.shape[0]
        wq = torch.ones(nq,1)
        return loss_(wq,q,wz,z)
    return loss

# data attachment function for curves, varifolds model
def lossVarifoldCurve(FS, VT, FT, K):
    def get_center_length_tangents(F, V):
        V0, V1 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1])
        centers, tangents = .5*(V0+V1), V1-V0
        length = (tangents**2).sum(dim=1)[:, None].sqrt()

        return centers, length, tangents / length
    
    CT, LT, TTn = get_center_length_tangents(FT, VT)
    cst = (LT * K(CT, CT, TTn, TTn, LT)).sum()

    def loss(VS):
        CS, LS, TSn = get_center_length_tangents(FS, VS)

        return cst + (LS * K(CS, CS, TSn, TSn, LS)).sum() - 2 * (LS * K(CS, CT, TSn, TTn, LT)).sum()
    
    return loss

def Optimize_with_vis(loss, args, niter=10):
    optimizer = torch.optim.LBFGS(args)
    losses = []
    print('Performing optimization with visualization...')
    for i in range(niter):
        print("Iteration", i+1, "of", niter)
        def closure():
            optimizer.zero_grad()
            L = loss(*args)
            losses.append(L.item())
            L.backward()
            return L
        optimizer.step(closure)

        # Create clones of current momentum and source that require grad,
        # so that Shooting (which calls autograd.grad) works correctly.
        p_vis = args[0].detach().clone().requires_grad_()
        q_vis = q0.detach().clone().requires_grad_()
        
        # Optionally, compute the deformed shape:
        p_current, q_current = Shooting(p_vis, q_vis, Kv)
        
        # Visualize using the PlotRes3D function:
        filename = f"output/deformation_iter_{i+1}.html"
        PlotRes3D(VS,FS,VT,FT, filename)(q_vis, p_vis, Kv, src_opacity=0, tgt_opacity=0, def_opacity=1, showgrid=False)
        
    print("Optimization done.")
    
    return args, losses

# Interploate 3D boundary points ()
# from scipy.interpolate import PchipInterpolator, interp1d
# def interpolate_points(coords, angle, total_points):
#     rot_angle = angle * (np.pi / 180)
#     rot_mat = np.array([[np.cos(rot_angle), np.sin(rot_angle)],
#                         [-np.sin(rot_angle), np.cos(rot_angle)]])

#     coords_rot = (rot_mat @ coords.T).T
#     num_points = coords_rot.shape[0]

#     if num_points < total_points:
#         points_per_segment = total_points // num_points
        
#         coords_interp = []
#         for i in range(num_points - 1):
#             segment_start = coords_rot[i, :]
#             segment_end = coords_rot[i + 1, :]
        
#             if i == num_points - 2:
#                 points_per_segment = total_points - (points_per_segment * (num_points - 2))
        
#             segment_x = np.linspace(segment_start[0], segment_end[0], points_per_segment)
#             if segment_start[0] == segment_end[0]:  # Avoid interpolation error
#                 segment_y = np.linspace(segment_start[1], segment_end[1], points_per_segment)
#             else:
#                 interpolator = interp1d([segment_start[0], segment_end[0]], [segment_start[1], segment_end[1]], kind='linear', fill_value='extrapolate')
#                 segment_y = interpolator(segment_x)
            
#             coords_interp.append(np.column_stack((segment_x, segment_y)))
        
#         coords_interp = np.vstack(coords_interp)
#     else:
#         random_indices = np.random.permutation(num_points)[:total_points]
#         coords_interp = coords_rot[random_indices, :]

#     coords_interp_rot = coords_interp.copy()

#     rot_mat = np.array([[np.cos(-rot_angle), np.sin(-rot_angle)],
#                         [-np.sin(-rot_angle), np.cos(-rot_angle)]])
    
#     coords_interp = (rot_mat @ coords_interp_rot.T).T

#     return coords_interp