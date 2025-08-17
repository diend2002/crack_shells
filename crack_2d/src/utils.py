import torch
import numpy as np
import matplotlib.pyplot as plt

# def create_sampling(params, crack_a, grid_size, concentration):
#     nx, ny = grid_size['nx'], grid_size['ny']
#     x_min, x_max = params['domain_x_min'], params['domain_x_max']
#     y_min, y_max = params['domain_y_min'], params['domain_y_max']
#     tip_x, tip_y = crack_a, 0.0
#     beta_x, beta_y = concentration['beta_x'], concentration['beta_y']

#     u = torch.linspace(-1, 1, nx)
#     t = torch.tanh(u * beta_x)
#     x_coords = tip_x + (x_max - x_min - tip_x) * t / torch.tanh(torch.tensor(beta_x))
#     # Điều chỉnh để đảm bảo đầu mút bên trái là x_min
#     x_coords = x_coords - (x_coords[0] - x_min)

#     v = torch.linspace(-1, 1, ny)
#     s = torch.tanh(v * beta_y)
#     y_coords = tip_y + (y_max - y_min) * 0.5 * s / torch.tanh(torch.tensor(beta_y))
    
#     x_coords, _ = torch.sort(x_coords)
#     y_coords, _ = torch.sort(y_coords)
    
#     grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
#     x_flat = grid_x.flatten().unsqueeze(1)
#     y_flat = grid_y.flatten().unsqueeze(1)
    
#     return x_flat, y_flat

def create_sampling(params, crack_a):
    """
    Tạo một lưới điểm sampling với phân bố tùy chỉnh được chỉ định.
    Mật độ điểm được tập trung tại các vùng cụ thể quanh vết nứt.

    Args:
        params (dict): Dictionary chứa ranh giới của miền tính toán.
                       Ví dụ: {'domain_x_min': 0.0, 'domain_x_max': 1.0, ...}
        crack_a (float): Tọa độ x của đầu vết nứt.

    Returns:
        (torch.Tensor, torch.Tensor): Hai tensor đã được làm phẳng,
                                       chứa tọa độ x và y của lưới điểm.
    """
    # Lấy ranh giới từ dictionary params
    x_min, x_max = params['domain_x_min'], params['domain_x_max']
    y_min, y_max = params['domain_y_min'], params['domain_y_max']

    # --- Tạo tọa độ trục y với phân bố tùy chỉnh ---
    y_bottom = torch.linspace(-1.0, -0.3, 14, dtype=torch.float32)[:-1] # Bỏ điểm cuối
    y_middle = torch.linspace(-0.3, 0.3, 20, dtype=torch.float32)[:-1]  # Bỏ điểm cuối
    y_top = torch.linspace(0.3, 1.0, 14, dtype=torch.float32)
    y_coords = torch.cat([y_bottom, y_middle, y_top])

    # --- Tạo tọa độ trục x với phân bố tùy chỉnh ---
    x_refine_start = crack_a - 0.05
    x_refine_end = crack_a + 0.05
    
    x_left = torch.linspace(x_min, x_refine_start, 6, dtype=torch.float32)[:-1]
    x_middle = torch.linspace(x_refine_start, x_refine_end, 25, dtype=torch.float32)[:-1]
    x_right = torch.linspace(x_refine_end, x_max, 7, dtype=torch.float32)
    x_coords = torch.cat([x_left, x_middle, x_right])

    # --- Tạo lưới 2D và làm phẳng ---
    # torch.meshgrid tạo ra lưới tọa độ
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')

    # .flatten() để chuyển từ ma trận 2D thành vector 1D
    # .unsqueeze(1) để chuyển từ vector (N) thành cột (N, 1)
    x_flat = grid_x.flatten().unsqueeze(1)
    y_flat = grid_y.flatten().unsqueeze(1)

    return x_flat, y_flat

def crack_embedding_fn(x, y, a):
    
    relu_squared = torch.clamp(a - x, min=0)**2
    sign_x2 = torch.sign(y)
    sign_x2 = torch.where(y == 0, torch.tensor(1.0, dtype=y.dtype, device=y.device), sign_x2)
    return relu_squared * sign_x2

def apply_hard_constraints(u_hat_1, u_hat_2, x, y, domain_y_min, domain_y_max):
    
    u1 = u_hat_1 * x
    u2 = u_hat_2 * (y - domain_y_min) / (domain_y_max - domain_y_min)
    return u1, u2

def calculate_analytical_K1(sigma_pa, a_m, b_m):
    
    sigma_mpa = sigma_pa / 1e6
    a_mm = a_m * 1000
    ratio = a_m / b_m
    term1 = sigma_mpa * np.sqrt(np.pi * a_mm)
    term2 = 1 - 0.025 * (ratio)**2 + 0.06 * (ratio)**4
    term3 = np.sqrt(1 / np.cos(np.pi * ratio / 2))
    return term1 * term2 * term3

def calculate_K1_from_model_SI(net_u1, net_u2, a_m, params, device,
                               plane_strain=True,
                               r_min_ratio=0.02, r_max_ratio=0.20,
                               n_points=120, y_eps_m=1e-6):
    """
    Trả về K1 (MPa·√mm) từ mô hình SI.
    Yêu cầu: E, sigma ở Pa; x,y,a ở m; u ở m.
    """
    E, nu = params['E'], params['nu']        # Pa
    
    mu = E/(2*(1+nu))
    kappa = (3 - 4*nu) if plane_strain else (3 - nu)/(1 + nu)

    a = torch.tensor(a_m, dtype=torch.float32, device=device)
    r_min = (r_min_ratio * a).item()
    r_max = (r_max_ratio * a).item()
    r = torch.linspace(r_min, r_max, n_points, device=device).unsqueeze(1)  # m
    x = a - r
    y_above = torch.full_like(x, +y_eps_m)
    y_below = torch.full_like(x, -y_eps_m)

    with torch.no_grad():
        emb_a = crack_embedding_fn(x, y_above, a)
        ua1 = net_u1(torch.cat([x, y_above, emb_a], dim=1))
        ua2 = net_u2(torch.cat([x, y_above, emb_a], dim=1))
        _, u2a = apply_hard_constraints(ua1, ua2, x, y_above,
                                        params['domain_y_min'], params['domain_y_max'])  # m

        emb_b = crack_embedding_fn(x, y_below, a)
        ub1 = net_u1(torch.cat([x, y_below, emb_b], dim=1))
        ub2 = net_u2(torch.cat([x, y_below, emb_b], dim=1))
        _, u2b = apply_hard_constraints(ub1, ub2, x, y_below,
                                        params['domain_y_min'], params['domain_y_max'])  # m

    delta2 = (u2a - u2b)  # m
    print("Trường chuyển vị", delta2)
    
    K_local = (mu/(kappa+1.0)) * torch.sqrt(2.0*torch.pi / r) * delta2
    print("K_local", K_local)

    # Fit lấy intercept
    r_np = r.cpu().numpy().flatten()
    K_np = K_local.cpu().numpy().flatten()

    if K_np.size < 2:
        return float('nan')
    
    w = 1.0/np.sqrt(r_np + 1e-18)
    print("r_np", r_np)
    print("K_np", K_np)
    a1, b0_Pa_sqrt_m = np.polyfit(r_np, K_np, 1, w=w)
    print("a1, b0", a1, b0_Pa_sqrt_m)
    log_rk = torch.cat((r_np, K_np), dim =1)
    # ĐỔI ĐƠN VỊ: Pa·√m → MPa·√mm
    K1_MPa_sqrt_mm = b0_Pa_sqrt_m * (1e-6 * np.sqrt(1000.0))
    return float(K1_MPa_sqrt_mm), log_rk, a1, b0_Pa_sqrt_m


def visualize(net_u1, net_u2, crack_a, params, device):
    print(f"\n[TRỰC QUAN HÓA] Đang tạo biểu đồ cho a = {crack_a:.1f} m...")
    
    # 1. Tạo một lưới điểm ĐỀU và MỊN để vẽ biểu đồ
    nx_vis, ny_vis = 201, 401
    x_vis = torch.linspace(params['domain_x_min'], params['domain_x_max'], nx_vis, device=device)
    y_vis = torch.linspace(params['domain_y_min'], params['domain_y_max'], ny_vis, device=device)
    x_grid_vis, y_grid_vis = torch.meshgrid(x_vis, y_vis, indexing='ij')
    x_flat_vis = x_grid_vis.flatten().unsqueeze(1).requires_grad_(True)
    y_flat_vis = y_grid_vis.flatten().unsqueeze(1).requires_grad_(True)

    # 2. Tính toán chuyển vị từ mô hình
    net_u1.eval()
    net_u2.eval()
    with torch.no_grad():
        embedding_vis = crack_embedding_fn(x_flat_vis, y_flat_vis, crack_a)
        nn_input_vis = torch.cat((x_flat_vis, y_flat_vis, embedding_vis), dim=1)
        u_hat_1_vis = net_u1(nn_input_vis)
        u_hat_2_vis = net_u2(nn_input_vis)
    
    u1_vis, u2_vis = apply_hard_constraints(u_hat_1_vis, u_hat_2_vis, x_flat_vis, y_flat_vis, params['domain_y_min'], params['domain_y_max'])
    u2_vis_cpu = u2_vis.cpu().reshape(nx_vis, ny_vis).numpy()

    # 3. Tính toán ứng suất từ mô hình
    u1_vis.requires_grad_(True) # Cần bật lại grad để tính đạo hàm cho ứng suất
    u2_vis.requires_grad_(True)
    
    u1_x = torch.autograd.grad(u1_vis.sum(), x_flat_vis, create_graph=True)[0]
    u2_y = torch.autograd.grad(u2_vis.sum(), y_flat_vis, create_graph=True)[0]
    
    C11 = params['E'] / ((1 + params['nu']) * (1 - 2 * params['nu'])) * (1 - params['nu'])
    C12 = params['E'] / ((1 + params['nu']) * (1 - 2 * params['nu'])) * params['nu']
    
    sigma_yy = C12 * u1_x + C11 * u2_y
    sigma_yy_cpu = sigma_yy.cpu().detach().reshape(nx_vis, ny_vis).numpy()

    # 4. Vẽ 4 biểu đồ con
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f'Kết quả DEDEM cho a = {crack_a:.1f} m', fontsize=16)

    # Biểu đồ (e): Chuyển vị u2
    im1 = axs[0, 0].contourf(x_grid_vis.cpu(), y_grid_vis.cpu(), u2_vis_cpu, levels=50, cmap='jet')
    fig.colorbar(im1, ax=axs[0, 0])
    axs[0, 0].set_title('Chuyển vị u2 (m)')
    axs[0, 0].set_aspect('equal', adjustable='box')

    # Biểu đồ (f): Sai số u2
    axs[0, 1].text(0.5, 0.5, 'Biểu đồ sai số của u2\n(Yêu cầu dữ liệu FEM)', ha='center', va='center', fontsize=12, style='italic')
    axs[0, 1].set_title('Sai số tuyệt đối của u2')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 1].set_aspect('equal', adjustable='box')

    # Biểu đồ (g): Ứng suất sigma_22
    im2 = axs[1, 0].contourf(x_grid_vis.cpu(), y_grid_vis.cpu(), sigma_yy_cpu / 1e6, levels=50, cmap='jet')
    fig.colorbar(im2, ax=axs[1, 0], label='MPa')
    axs[1, 0].set_title('Ứng suất sigma_22 (MPa)')
    axs[1, 0].set_aspect('equal', adjustable='box')

    # Biểu đồ (h): Sai số sigma_22
    axs[1, 1].text(0.5, 0.5, 'Biểu đồ sai số của sigma_22\n(Yêu cầu dữ liệu FEM)', ha='center', va='center', fontsize=12, style='italic')
    axs[1, 1].set_title('Sai số tuyệt đối của sigma_22')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[1, 1].set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
