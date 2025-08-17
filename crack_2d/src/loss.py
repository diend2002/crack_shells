import torch 

# def calculate_potential_energy(u1, u2, x, y, params):
    
#     u1_x = torch.autograd.grad(u1.sum(), x, create_graph=True, allow_unused=True)[0]
#     u1_y = torch.autograd.grad(u1.sum(), y, create_graph=True, allow_unused=True)[0]
#     u2_x = torch.autograd.grad(u2.sum(), x, create_graph=True, allow_unused=True)[0]
#     u2_y = torch.autograd.grad(u2.sum(), y, create_graph=True, allow_unused=True)[0]
#     eps_xx, eps_yy, eps_xy = u1_x, u2_y, 0.5 * (u1_y + u2_x)
#     C11 = params['E'] / ((1 + params['nu']) * (1 - 2 * params['nu'])) * (1 - params['nu'])
#     C12 = params['E'] / ((1 + params['nu']) * (1 - 2 * params['nu'])) * params['nu']
#     C33 = params['E'] / (2 * (1 + params['nu']))
#     sigma_xx = C11 * eps_xx + C12 * eps_yy
#     sigma_yy = C12 * eps_xx + C11 * eps_yy
#     sigma_xy = 2 * C33 * eps_xy
#     strain_energy_density = 0.5 * (sigma_xx * eps_xx + sigma_yy * eps_yy + 2 * sigma_xy * eps_xy)
#     domain_area = (params['domain_x_max'] - params['domain_x_min']) * (params['domain_y_max'] - params['domain_y_min'])
#     total_strain_energy = torch.mean(strain_energy_density) * domain_area
#     boundary_mask = (y == params['domain_y_max'])
#     u2_boundary = u2[boundary_mask]
#     work_traction = torch.mean(u2_boundary * params['sigma_tension']) * (params['domain_x_max'] - params['domain_x_min'])
#     return total_strain_energy - work_traction

def calculate_potential_energy(u1, u2, x, y, params):
    u1_x = torch.autograd.grad(u1.sum(), x, create_graph=True)[0]
    u1_y = torch.autograd.grad(u1.sum(), y, create_graph=True)[0]
    u2_x = torch.autograd.grad(u2.sum(), x, create_graph=True)[0]
    u2_y = torch.autograd.grad(u2.sum(), y, create_graph=True)[0]

    eps_xx = u1_x
    eps_yy = u2_y
    eps_xy = 0.5 * (u1_y + u2_x)

    E, nu = params['E'], params['nu']
    factor = E / ((1 + nu) * (1 - 2 * nu))
    C11 = factor * (1 - nu)
    C12 = factor * nu
    C33 = E / (2 * (1 + nu))

    sigma_xx = C11 * eps_xx + C12 * eps_yy
    sigma_yy = C12 * eps_xx + C11 * eps_yy
    sigma_xy = 2 * C33 * eps_xy

    strain_energy_density = 0.5 * (sigma_xx * eps_xx + sigma_yy * eps_yy + 2 * sigma_xy * eps_xy)
    domain_area = (params['domain_x_max'] - params['domain_x_min']) * (params['domain_y_max'] - params['domain_y_min'])
    total_strain_energy = torch.mean(strain_energy_density) * domain_area

    # Sửa phần boundary mask
    boundary_mask = torch.isclose(y, torch.tensor(params['domain_y_max'], dtype=y.dtype, device=y.device), atol=1e-6)
    if boundary_mask.any():
        u2_boundary = u2[boundary_mask]
        work_traction = torch.mean(u2_boundary * params['sigma_tension']) * (params['domain_x_max'] - params['domain_x_min'])
    else:
        work_traction = 0.0

    return total_strain_energy - work_traction
