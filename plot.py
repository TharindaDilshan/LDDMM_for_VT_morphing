import torch
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from iterative_LDDMM import Shooting, Flow

use_pyvista = True

# result display function for landmark or point cloud data
def PlotRes2D(z, pts=None):
    def plotfun(q0,p0,Kv, showgrid=True):
        p,q = Shooting(p0,q0,Kv)
        q0np, qnp = q0.data.numpy(), q.data.numpy()
        q0np, qnp, znp = q0.data.numpy(), q.data.numpy(), z.data.numpy()
        plt.plot(znp[:,0],znp[:,1],'.');
        plt.plot(q0np[:,0],q0np[:,1],'+');
        plt.plot(qnp[:,0],qnp[:,1],'o');
        plt.axis('equal');
        if showgrid:
            X = get_def_grid(p0,q0,Kv)
            plt.plot(X[0],X[1],'k',linewidth=.25);
            plt.plot(X[0].T,X[1].T,'k',linewidth=.25); 
        n,d = q0.shape
        nt = 20
        Q = np.zeros((n,d,nt))
        for i in range(nt):
            t = i/(nt-1)
            Q[:,:,i] = Shooting(t*p0,q0,Kv)[1].data.numpy()
        plt.plot(Q[:,0,:].T,Q[:,1,:].T,'y');
        if type(pts)!=type(None):
            phipts = Flow(pts,p0,q0,Kv).data
            plt.plot(phipts.numpy()[:,0],phipts.numpy()[:,1],'.b',markersize=.1);
    return plotfun

# display function for triangulated surface type data
def PlotRes3D(VS, FS, VT, FT, filename="deformation.html"):
    def plotfun(q0, p0, Kv, src_opacity=1, tgt_opacity=1, def_opacity=1, showgrid=True):
        # q0, p0: Initial vertices and momenta (optimized) for the source shape
        # simulate the deformation process and compute the final deformed shape q
        p,q = Shooting(p0,q0,Kv)
        # q0np, qnp: numpy arrays of the initial and final vertex positions
        q0np, qnp = q0.data.numpy(), q.data.numpy()
        # numpy arrays of source and target faces
        FSnp,VTnp, FTnp = FS.data.numpy(),  VT.data.numpy(), FT.data.numpy() 
        if use_pyvista:
            p = pv.Plotter()
            opacity = 1
            # mesh for the initial shape
            p.add_mesh(surf_to_pv(q0np,FSnp), color='lightblue', opacity=src_opacity)
            # mesh for the deformed shape
            p.add_mesh(surf_to_pv(qnp,FSnp), color='lightcoral', opacity=def_opacity)
            # mesh for target shape
            p.add_mesh(surf_to_pv(VTnp,FTnp), color='lightgreen', opacity=tgt_opacity)
            if showgrid:
                ng = 20
                X = get_def_grid(p0,q0,Kv,ng=ng)
                for k in range(3):
                    for i in range(ng):
                        for j in range(ng):
                            p.add_mesh(lines_from_points(X[:,i,j,:].T))
                    X = X.transpose((0,2,3,1))
            # p.show()
            p.export_html(filename)
            p.close()
        else:
            fig = plt.figure();
            plt.axis('off')
            plt.title('LDDMM matching example')     
#             ax = Axes3D(fig, auto_add_to_figure=False)
            ax = Axes3D(fig)
            # triangular mesh for the initial shape
            ax.plot_trisurf(q0np[:,0],q0np[:,1],q0np[:,2],triangles=FSnp,alpha=.5)
            # triangular mesh for the deformed shape
            ax.plot_trisurf(qnp[:,0],qnp[:,1],qnp[:,2],triangles=FSnp,alpha=.5)
            # triangular mesh for the target shape
            ax.plot_trisurf(VTnp[:,0],VTnp[:,1],VTnp[:,2],triangles=FTnp,alpha=.5)
            if showgrid:
                ng = 20
                X = get_def_grid(p0,q0,Kv,ng=ng)
                for k in range(3):
                    for i in range(ng):
                        for j in range(ng):
                            ax.plot(X[0,i,j,:],X[1,i,j,:],X[2,i,j,:],'k',linewidth=.25);
                    X = X.transpose((0,2,3,1))
            fig.add_axes(ax)
    return plotfun

def get_def_grid(p0,q0,Kv,ng=50):
    d = p0.shape[1]
    p,q = Shooting(p0,q0,Kv)
    q0np, qnp = q0.data.numpy(), q.data.numpy()
    q0np, qnp = q0.data.numpy(), q.data.numpy()
    # calculates the minimum (a) and maximum (b) coordinates for the vertices in the 
    # initial (q0) and final (q) positions to establish the bounds of the grid
    a = list(np.min(np.vstack((q0np[:,k],qnp[:,k]))) for k in range(d))
    b = list(np.max(np.vstack((q0np[:,k],qnp[:,k]))) for k in range(d))
    # expands these bounds by 20% to ensure the grid extends slightly beyond the 
    # immediate area covered by the initial and deformed shapes
    sz = 0.2
    lsp = list(np.linspace(a[k]-sz*(b[k]-a[k]),b[k]+sz*(b[k]-a[k]),ng,dtype=np.float32) for k in range(d))
    X = np.meshgrid(*lsp)
    x = np.concatenate(list(X[k].reshape(ng**d,1) for k in range(d)),axis=1)
    # transform grid (x) according to the LDDMM mapping
    # Flow():  integrates how each point in the grid moves under the transformation 
    # defined by Kv, p0, and q0. This step effectively applies the diffeomorphic map to the entire grid
    phix = Flow(torch.from_numpy(x),p0,q0,Kv).detach().numpy()
    X = phix.transpose().reshape([d]+[ng]*d)
    return X

def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly

def surf_to_pv(V,F):
    nf = F.shape[0]
    F = np.hstack((np.ones((nf,1),dtype="int")*3,F))
    F = F.flatten()
    surf = pv.PolyData(V,F)
    return surf

# fonction d'affichage pour des données de type surface triangulée
def PlotResSurf(VS,FS,VT,FT):
    def plotfun(q0,p0,Kv):
        fig = plt.figure();
        plt.axis('off')
        plt.title('LDDMM matching example')  
        p,q = Shooting(p0,q0,Kv)
        q0np, qnp = q0.data.numpy(), q.data.numpy()
        FSnp,VTnp, FTnp = FS.data.numpy(),  VT.data.numpy(), FT.data.numpy()    
        ax = Axes3D(fig, auto_add_to_figure=False)
        ax.plot_trisurf(q0np[:,0],q0np[:,1],q0np[:,2],triangles=FSnp,alpha=.5)
        ax.plot_trisurf(qnp[:,0],qnp[:,1],qnp[:,2],triangles=FSnp,alpha=.5)
        ax.plot_trisurf(VTnp[:,0],VTnp[:,1],VTnp[:,2],triangles=FTnp,alpha=.5)
        fig.add_axes(ax)
    return plotfun