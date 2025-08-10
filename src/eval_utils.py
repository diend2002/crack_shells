import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# your local imports
from src.utils import *
from src.pinn_model import PINN               # MLP PINN
from src.pikan_model import PyKAN_PINN        # PyKAN version
from src.crack_geometry import CrackGeometry

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def grid_eval_pinn(
    geometry,
    FEM_sol_dir,
    crack_type='center',
    crack_param=None,
    smoothing=1e-2,
    gate_eps=1e-3,
    bcs='ls',
    arch=[50,'gelu',50,'gelu',50,'gelu'],     # a reasonable MLP by default
    model_type='pinn',                        # 'pinn' or 'kan'
    model_path='./checkpoints/epoch_000100.pt'
):
    """
    Figure layout (2x2):

      [ (1) FEM 3D surface ]   [ (2) PINN 3D surface ]
      [ (3) FEM u_z heatmap ]  [ (4) PINN u_z heatmap ]

    Expectation: model was trained with inputDim = 3: [xi1, xi2, gamma].
    """

    # ---------- grid in ξ-space ----------
    num_points = 100
    if geometry == 'hemisphere':
        radius_grid = torch.linspace(0., 1., num_points).double().to(device) * (1. - 1e-3)
        theta = torch.linspace(0, 1., num_points).double().to(device) * 2.*np.pi
        r_mesh, t_mesh = torch.meshgrid(radius_grid, theta, indexing='ij')
        x_mesh = r_mesh * torch.cos(t_mesh)
        y_mesh = r_mesh * torch.sin(t_mesh)
    else:
        x = torch.linspace(-0.5, 0.5, num_points).double().to(device)
        y = torch.linspace(-0.5, 0.5, num_points).double().to(device)
        x_mesh, y_mesh = torch.meshgrid(x, y, indexing='ij')

    mesh_input = torch.stack((torch.flatten(x_mesh), torch.flatten(y_mesh)), dim=1)  # (N,2)

    x_mesh = x_mesh.float()
    y_mesh = y_mesh.float()
    mesh_input = mesh_input.float()

    # ---------- crack embedding (same as training) ----------
    if crack_param is None:
        crack_param = {'length': 0.3, 'center': (0.0, 0.0), 'angle': 0.0}
    crack = CrackGeometry(crack_type, crack_param, smoothing=smoothing)
    gamma = crack.strong_embedding(mesh_input, gate_eps=gate_eps)               # (N,)
    mesh_crack = torch.cat([mesh_input, gamma[:, None]], dim=1)                    # (N,3)

    # ---------- load model ----------
    if model_type.lower() == 'kan':
        model = PyKAN_PINN(3, [5], 5, bcs)
    else:
        model = PINN(3, arch, 5, bcs)

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    # ---------- model predictions ----------
    with torch.no_grad():
        u = model(mesh_crack)
    ux_p = u[:, 0].cpu().detach().numpy()
    uy_p = u[:, 1].cpu().detach().numpy()
    uz_p = u[:, 2].cpu().detach().numpy()
    th1 = u[:, 3].cpu().detach().numpy()
    th2 = u[:, 4].cpu().detach().numpy() 

    xi_p = mesh_input.cpu().detach().numpy()

    # ---------- FEM CSV ----------
    data_fem = np.genfromtxt(FEM_sol_dir, delimiter=',', skip_header=1)
    xi_fem = data_fem[:, 0:2]
    fem_ux, fem_uy, fem_uz, fem_th1, fem_th2 = [data_fem[:, i] for i in range(2, 7)]

    # ---------- plotting grid in ξ ----------
    if geometry in ('hyperb_parab', 'scordelis_lo'):
        L = 1.0; nn = 200
        X, Y = np.meshgrid(np.linspace(-L/2, L/2, nn),
                           np.linspace(-L/2, L/2, nn), indexing='ij')
    elif geometry == 'hemisphere':
        radius = 1.0; nn = 200
        radius_grid = np.linspace(0, radius, nn) * (1. - 2e-3)
        theta = np.linspace(0, 2.*np.pi, nn)
        rv, tv = np.meshgrid(radius_grid, theta, indexing='ij')
        X = rv * np.cos(tv); Y = rv * np.sin(tv)
    else:
        raise ValueError('Unknown geometry for plotting.')

    # ---------- interpolate fields to plotting grid ----------
    FEM_x = griddata(xi_fem, fem_ux.flatten(), (X, Y), method='cubic')
    FEM_y = griddata(xi_fem, fem_uy.flatten(), (X, Y), method='cubic')
    FEM_z = griddata(xi_fem, fem_uz.flatten(), (X, Y), method='cubic')
    FEM_thetas = griddata(xi_fem, np.abs(fem_th1.flatten())+np.abs(fem_th2.flatten()), (X, Y), method='cubic')

    PINN_x = griddata(xi_p, ux_p.flatten(), (X, Y), method='cubic')
    PINN_y = griddata(xi_p, uy_p.flatten(), (X, Y), method='cubic')
    PINN_z = griddata(xi_p, uz_p.flatten(), (X, Y), method='cubic')
    PINN_thetas = griddata(xi_p, np.abs(th1.flatten())+np.abs(th2.flatten()), (X, Y), method='cubic')

    # ---------- undeformed midsurface mapping ----------
    def undef(x_mesh, y_mesh, mapname):
        if mapname == 'hyperb_parab':
            x = x_mesh
            y = y_mesh
            z = (x_mesh**2 - y_mesh**2)
        elif mapname == 'scordelis_lo':
            radius = 25./50.
            length = 50./50.
            angle = y_mesh * 4.*np.pi/9.
            x = length * x_mesh
            y = radius * np.sin(angle)
            z = radius * np.cos(angle)
        elif mapname == 'hemisphere':
            z_temp = np.clip(1.001 - x_mesh**2 - y_mesh**2, a_min=0, a_max=None)
            x = x_mesh; y = y_mesh; z = np.sqrt(z_temp)
        else:
            raise ValueError('Mapping not implemented.')
        return x, y, z

    x_undef, y_undef, z_undef = undef(X, Y, geometry)

    # ---------- visual displacement scale ----------
    factor = 0.005 if geometry == 'hyperb_parab' else (0.001 if geometry == 'scordelis_lo' else 0.05)

    # ---------- colors for 3D (|theta1|+|theta2|) ----------
    color_dimension1 = FEM_thetas * factor
    color_dimension2 = PINN_thetas * factor

    # minn, maxx = np.nanmin([color_dim1, color_dim2]), np.nanmax([color_dim1, color_dim2])
    minn, maxx = np.concatenate((color_dimension1,color_dimension2)).min(), np.concatenate((color_dimension1,color_dimension2)).max()

    norm3d = matplotlib.colors.Normalize(minn, maxx)
    m3d = plt.cm.ScalarMappable(norm=norm3d, cmap='inferno'); m3d.set_array([])
    fcolors1 = m3d.to_rgba(color_dimension1)
    fcolors2 = m3d.to_rgba(color_dimension2)

    # ---------- figure ----------
    fig = plt.figure()
    ax1 = fig.add_subplot(1,19,(1,8),projection='3d')
    ax2 = fig.add_subplot(1,19,(10,17),projection='3d')
    ax3 = fig.add_subplot(1,19,(18,19),adjustable='box')
    ax4 = fig.add_subplot(1,19,9,adjustable='box')

    ax1.plot_surface(x_undef+FEM_x*factor, y_undef+FEM_y*factor, z_undef+FEM_z*factor, rstride=1, cstride=1, facecolors=fcolors1, vmin=minn, vmax=maxx, shade='auto', rasterized=True)
    if geometry == 'hemisphere':
        ax1.set_title('FEniCS', pad=-20)
        ax1.set_xticks([-1.0,0,1.0])
        ax1.set_yticks([-1.0,0,1.0])
        ax1.set_zticks([0.0,0.35,0.7])
        ax1.view_init(80, 35)
    else:
        ax1.set_title('FEniCS', pad=-50)
        ax1.set_xticks([-0.5,0,0.5])
        ax1.set_yticks([-0.5,0,0.5])
    ax1.tick_params(axis='x', pad=-5)
    ax1.set_xlabel(r'$x_1$',labelpad=-12)
    ax1.tick_params(axis='y',pad=-4)
    ax1.set_ylabel(r'$x_2$',labelpad=-9)
    ax1.tick_params(axis='z',pad=-1)
    ax1.set_zlabel(r'$x_3$',labelpad=-5)

    ax2.plot_surface(x_undef+PINN_x*factor, y_undef+PINN_y*factor, z_undef+PINN_z*factor, rstride=1, cstride=1, facecolors=fcolors2, vmin=minn, vmax=maxx, shade='auto', rasterized=True)
    if geometry == 'hemisphere':
        ax2.set_title('\n\n\n\nPINN', pad=-20)
        ax2.set_xticks([-1.0,0,1.0])
        ax2.set_yticks([-1.0,0,1.0])
        ax2.set_zticks([0.0,0.35,0.7])
        ax2.view_init(80, 35)
    else:
        ax2.set_title('\n\n\n\nPINN', pad=-50)
        ax2.set_xticks([-0.5,0,0.5])
        ax2.set_yticks([-0.5,0,0.5])
    ax2.tick_params(axis='x', pad=-5)
    ax2.set_xlabel(r'$x_1$',labelpad=-12)
    ax2.tick_params(axis='y',pad=-4)
    ax2.set_ylabel(r'$x_2$',labelpad=-9)
    ax2.tick_params(axis='z',pad=-1)
    ax2.set_zlabel(r'$x_3$',labelpad=-5)

    ax4.set_axis_off()
    ax4.set_box_aspect(1.)

    fig.colorbar(m3d,ax=ax3)
    ax3.set_title(r'$|{{\theta}}|$', x=1.3)
    ax3.set_axis_off()
    ax3.set_box_aspect(4.)
    
    fig.tight_layout()
    plt.show()
