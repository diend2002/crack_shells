import os
import torch
from src.utils import *
from src.eval_utils import *
from src.pinn_model import PINN
from src.pinnkan_model import PyKAN_PINN
from src.geometry import Geometry
from src.shell_model import LinearNagdhi
from src.material_model import LinearElastic
from params import create_param_dict

from torch.func import jacrev


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
def print_gpu_memory(step_name=""):
    """Hàm để in ra thông tin bộ nhớ GPU hiện tại."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**2  # Chuyển đổi sang MB
        reserved = torch.cuda.memory_reserved(0) / 1024**2   # Chuyển đổi sang MB
        print(f"[{step_name:^25s}] GPU Memory: {allocated:8.2f} MB allocated / {reserved:8.2f} MB reserved")
    else:
        print("CUDA is not available.")

if __name__ == '__main__':
    
    # select study: ['hyperb_parab', 'scordelis_lo', 'hemisphere']
    study = 'scordelis_lo'

    # to reproduce results from paper
    fix_seeds()
    # create directories
    os.makedirs('kan_models', exist_ok=True)
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
    #print_gpu_memory("Initial state")
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
    #print_gpu_memory("Sampling get")
    
    # PINN setup
    # arch = [50,'gelu',50,'gelu',50,'gelu']
    # pn = PINN(2,arch,5,bcs).to(device)

    # PIKAN setup 
    arch = [5]
    pn = PyKAN_PINN(2,arch,5,bcs) 
    #print_gpu_memory("model_init")
    #print(pn)
    
    # define optimizers
    optimizer_ADAM = torch.optim.Adam(pn.parameters(), lr=1.e-3)
    optimizer_LBFGS = torch.optim.LBFGS(pn.parameters(), tolerance_grad=1e-20, tolerance_change=1e-20, line_search_fn='strong_wolfe')
    
    #print_gpu_memory("optimizer established")
    # tracking
    loss_weak_form = []
    L2_error_list = []
    energy_history = [] ## added to record energy 


    # initialize shell
    #print('Precompute geometric measures at collocation points.')
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
    #print_gpu_memory("compute geometric")
    print('Done.')

    # closure
    def closure():
        def global_to_local(x):
            return bmv(S,pn(x))
        
        # obtain solution field and derivatives
        #print_gpu_memory("after model forward")
        # first_grad_list = []
        # for i in range(5):
        #     v = torch.cat([torch.ones(batch_len,1,device=device)*(i==j) for j in range(5)],1)
        #     #print_gpu_memory(f"compute nu_{i}")
        #     jacobian_vjp = vjp_inplace(global_to_local, xi_col, v, create_graph=True)[1]
        #     #print_gpu_memory(f"compute jacobian_{i}")
        #     first_grad_list.append(jacobian_vjp)
        
        # #print_gpu_memory("before_grad")
        # first_grad = torch.cat(first_grad_list,1)
        # first_grad_reshape = torch.reshape(first_grad, (batch_len,5,2))
        #print_gpu_memory("before pred_5d")
        first_grad_reshape = jacrev(global_to_local)(xi_col)
        pred_5d = global_to_local(xi_col)
        #print_gpu_memory("after grad")
        #print(f"Kích thước đầu ra của mô hình (pred_5d): {pred_5d.shape}")
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
        
        #print_gpu_memory("measuring forces")
        
        # assemble external work
        if loading == 'gravity':
            W_ext = -1. * pn(xi_col)[:,2] * thickness * shell_density * loading_factor
        elif loading == 'concentrated_load':
            W_ext = -1. * pn(xi_col)[:,2] * torch.exp(-(torch.pow(xi_col[:,0], 2) + torch.pow(xi_col[:,1], 2)) / 0.1) * loading_factor
        elif loading == 'none':
            W_ext = torch.zeros(batch_len,device=device)
        else:
            raise ValueError('Loading type not recognized.')

        work = torch.mean(W_ext * sqrt_det_a * param_area)

        inner_energy = torch.mean((membrane_energy + bending_energy + shear_energy) * sqrt_det_a * param_area)

        loss = inner_energy - work

        #print_gpu_memory("after loss compute")

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
        # if optimizer == optimizer_LBFGS:
        #     loss.backward(retain_graph=True)
        # else:
        #     loss.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        #print_gpu_memory("after backward")
        return loss

    print('Start training.')
    for epoch in range(epochs):
        batch_len = len(xi_col)
        # if (epoch < opt_switch_epoch):   
        #     # Adam optimizer step
        #     optimizer = optimizer_ADAM
        # else:
        #     # LBFGS optimizer step
        #     optimizer = optimizer_LBFGS
        # optimizer.step(closure)
        # loss = closure()
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
        # Thêm dòng này để giải phóng bộ nhớ đệm không sử dụng sau mỗi epoch
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
        plot_shell(geometry, FEM_sol_dir)

