import os
import torch
import re
from datetime import datetime
from src.utils import *
from src.eval_utils import *
from src.pinn_model import PINN
from src.pikan_model import PyKAN_PINN
from src.geometry import Geometry
from src.shell_model import LinearNagdhi
from src.material_model import LinearElastic
from src.crack_geometry import CrackGeometry
from params import create_param_dict

CHECKPT_DIR = "checkpoints"
SAVE_EVERY = 50
os.makedirs(CHECKPT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device :{device}")

def print_gpu_memory(step_name=""):
    """Print GPU Consumption"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**2  # to MB
        reserved = torch.cuda.memory_reserved(0) / 1024**2   # to MB
        print(f"[{step_name:^25s}] GPU Memory: {allocated:8.2f} MB allocated / {reserved:8.2f} MB reserved")
    else:
        print("CUDA is not available.")

def save_checkpoint(epoch, pn, opt_adam, opt_lbfgs, loss, l2, tag=None):
    """Save full training state so we can resume exactly."""
    stamp = f"epoch_{epoch:06d}" if tag is None else tag
    path = os.path.join(CHECKPT_DIR, f"{stamp}.pt")
    torch.save({
        "epoch": epoch,
        "model_state": pn.state_dict(),
        "adam_state": None if opt_adam is None else opt_adam.state_dict(),
        "lbfgs_state": None if opt_lbfgs is None else opt_lbfgs.state_dict(),
        "loss": loss,
        "L2": l2,
        "timestamp": datetime.now().isoformat(),
    }, path)
    print(f"[ckpt] saved → {path}")
    return path

if __name__ == '__main__':
    
    # select study: ['hyperb_parab', 'scordelis_lo', 'hemisphere']
    study = 'hyperb_parab'

    # to reproduce results from paper
    fix_seeds()

    # create directories
    os.makedirs('eval', exist_ok=True)
    os.makedirs('loss_history', exist_ok=True)

    # we consider double precision
    torch.set_default_dtype(torch.float64)

    # load study parameters
    param_dict = create_param_dict(study)
    geometry = param_dict['geometry']
    loading = param_dict['loading']
    loading_factor = param_dict['loading_factor']
    E = param_dict['E']
    thickness = param_dict['thickness']
    shell_density = param_dict['shell_density']
    nu = param_dict['nu']
    shear_factor = param_dict['shear_factor']
    bcs = param_dict['bcs']
    N_col = param_dict['N_col']
    col_sampling = param_dict['col_sampling']
    epochs = param_dict['epochs']
    opt_switch_epoch = param_dict['opt_switch_epoch']
    FEM_sol_dir = param_dict['FEM_sol_dir']
    
    # frequency of L2 error evaluation (takes some time)
    l2_eval_freq = 1
    
    # activate to print out losses
    verbose = True

    # activate to plot predictions and compare to FEM
    plot = True

    # sample collocation points according to col_sampling
    xi_col = get_col_sampling(col_sampling, N_col)

    # transform to reference domain (warning: if reference domain changed, BCs must be adjusted accordingly)
    if col_sampling != 'concentric':
        xi_col = xi_col * 1. - 0.5
    
    # activate gradient tracking for geometric measures
    xi_col.requires_grad = True

    # Crack Geometry: use a strong (jump) embedding with smooth sign
    crack_geom = CrackGeometry(
                                crack_type='center',
                                crack_param= {'length': 0.3, 'center': (0.0, 0.0), 'angle': 0.0},
                                smoothing=1e-2
                                )

    # PINN setup
    # arch = [50,'gelu',50,'gelu',50,'gelu']
    # pn = PINN(2,arch,5,bcs).to(device)

    # PIKAN setup 
    arch = [5]
    pn = PyKAN_PINN(3, arch, 5, bcs) 
    
    # define optimizers
    optimizer_ADAM = torch.optim.Adam(pn.parameters(), lr=1.e-3)
    optimizer_LBFGS = torch.optim.LBFGS(pn.parameters(), tolerance_grad=1e-20, tolerance_change=1e-20, line_search_fn='strong_wolfe')
    
    # tracking
    loss_weak_form = []
    L2_error_list = []
    energy_history = [] ## added to record energy 


    # initialize shell
    print('Precompute geometric measures at collocation points.')
    geom = Geometry(geometry,xi_col)
    shell = LinearNagdhi(geom)
    material = LinearElastic(geom,E,nu)
    # geometric quantities
    # we keep gradient tracking for S (frame transform) to properly evaluate derivatives and disable it for all other quantities
    S = geom.S
    sqrt_det_a = geom.sqrt_det_a.clone().detach()
    param_area = geom.parametric_area.clone().detach()
    cov_metric = geom.cov_metric_tensor.clone().detach()
    # we rearrange the strain contributions to 3 matrices to distinguish between terms acting directly
    # on the solution field or the two corresponding derivatives w.r.t. curvilinear coordinates (_1, _2)
    Bm, Bm1, Bm2 = [i.clone().detach() for i in shell.membrane_strain_matrix]
    Bk, Bk1, Bk2 = [i.clone().detach() for i in shell.bending_strain_matrix]
    By, By1, By2 = [i.clone().detach() for i in shell.shear_strain_matrix]
    # material properties, using plane-stress conditions for Lamé constant lambda
    C = material.C.clone().detach()
    B = material.C.clone().detach()
    D = material.D.clone().detach()
    print('Done.')

    # closure
    def closure():
        def global_to_local(xi):
            gamma = crack_geom.strong_embedding(xi, gate_eps=1e-3)
            x = torch.cat([xi,gamma[:,None]],dim=1)
            return bmv(S,pn(x))
        
        

        # obtain solution field and derivatives
        first_grad_list = []
        for i in range(5):
            v = torch.cat([torch.ones(batch_len,1,device=device)*(i==j) for j in range(5)],1)
            jacobian_vjp = vjp_inplace(global_to_local, xi_col, v, create_graph=True)[1]
            first_grad_list.append(jacobian_vjp)
        
        
        first_grad = torch.cat(first_grad_list,1)
        first_grad_reshape = torch.reshape(first_grad, (batch_len,5,2))

        pred_5d = global_to_local(xi_col)
        pred_5d_1 = first_grad_reshape[:,:,0]
        pred_5d_2 = first_grad_reshape[:,:,1]

        # assemble membrane energy
        membrane_strains = bmv(Bm,pred_5d) + bmv(Bm1,pred_5d_1) + bmv(Bm2,pred_5d_2)
        membrane_energy = 0.5 * thickness * bdot(membrane_strains,bmv(C,membrane_strains))
        # assemble bending energy
        bending_strains = bmv(Bk,pred_5d) + bmv(Bk1,pred_5d_1) + bmv(Bk2,pred_5d_2)
        bending_energy = 0.5 * (thickness**3/12.) * bdot(bending_strains,bmv(B,bending_strains))
        # assemble shear energy
        shear_strains = bmv(By,pred_5d) + bmv(By1,pred_5d_1) + bmv(By2,pred_5d_2)
        shear_energy = 0.5 * shear_factor * thickness * bdot(shear_strains,bmv(D,shear_strains))
        
        # assemble external work
        gamma = crack_geom.strong_embedding(xi_col, gate_eps=1e-3)
        inputs = torch.cat([xi_col, gamma[:, None]], dim=1)
        if loading == 'gravity':
            W_ext = -1. * pn(inputs)[:,2] * thickness * shell_density * loading_factor
        elif loading == 'concentrated_load':
            W_ext = -1. * pn(inputs)[:,2] * torch.exp(-(torch.pow(xi_col[:,0], 2) + torch.pow(xi_col[:,1], 2)) / 0.1) * loading_factor
        elif loading == 'none':
            W_ext = torch.zeros(batch_len,device=device)
        else:
            raise ValueError('Loading type not recognized.')

        work = torch.mean(W_ext * sqrt_det_a * param_area)
        inner_energy = torch.mean((membrane_energy + bending_energy + shear_energy) * sqrt_det_a * param_area) 

        loss = inner_energy - work

        # tracking progress
        split_memb = torch.mean(membrane_energy * sqrt_det_a * param_area) / inner_energy
        split_bend = torch.mean(bending_energy * sqrt_det_a * param_area) / inner_energy
        split_shear = torch.mean(shear_energy * sqrt_det_a * param_area) / inner_energy

        # save energy history for tracking
        energy_history.append([epoch+1, inner_energy.item(), work.item(), split_memb.item(), split_bend.item(), split_shear.item()])

        if verbose:
            print('Inner energy: {:.2e}, Energy share (memb./bend./shear): {:.2f}/{:.2f}/{:.2f}, Work: {:.2e}'
                .format(inner_energy, split_memb, split_bend, split_shear, work))

        # optimizer step
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        #print_gpu_memory("after backward")
        return loss
    
    best = {"L2": float("inf"), "path": None}
    print('Start training.')
    for epoch in range(epochs):
        batch_len = len(xi_col)
        if (epoch < opt_switch_epoch):   
            optimizer = optimizer_ADAM
            # Standard pattern for Adam
            loss = closure()  # closure() calls loss.backward()
            optimizer.step()  # Adam applies the updates
        else:
            optimizer = optimizer_LBFGS
            # Standard pattern for L-BFGS
            # The step function calls closure internally and returns the final loss
            loss = optimizer.step(closure)

        loss_weak_form.append([epoch+1,loss.item()])
        if epoch % l2_eval_freq == 0:
            L2_error = compute_average_L2_error(pn, FEM_sol_dir)
            L2_error_list.append([epoch+1,L2_error.item()])
            print('Epoch: {}, rel. L2 error: {:.2e}'.format(epoch, L2_error))
        if (epoch + 1) % SAVE_EVERY == 0:
            save_checkpoint(epoch + 1, pn, optimizer_ADAM, optimizer_LBFGS,
                            loss.item(), L2_error.item(), tag=None)
        # best model by L2
        if L2_error.item() < best["L2"]:
            best["L2"] = L2_error.item()
            best["path"] = save_checkpoint(epoch + 1, pn, optimizer_ADAM, optimizer_LBFGS,
                                        loss.item(), L2_error.item(), tag="best")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        #reak

    print('Training finished.')

    exportList('loss_history/loss_weak_form', loss_weak_form)
    exportList('loss_history/L2_error', L2_error_list)
    torch.save(pn.state_dict(), 'models/pn_statedict.pt')
    exportList('loss_history/energy_history', energy_history)

    if plot:
        grid_eval_pinn(geometry)
        # plot_shell(geometry, FEM_sol_dir)

