import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# ===================================================================
# PHẦN 1: CÁC HÀM TIỆN ÍCH VÀ ĐỊNH NGHĨA MÔ HÌNH
# ===================================================================

def set_seed(seed):
    """Hàm để đặt seed cho các thư viện nhằm đảm bảo kết quả có thể tái tạo."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(size, size), nn.Tanh(),
            nn.Linear(size, size), nn.Tanh()
        )
    def forward(self, x):
        return x + self.layers(x)

class DEDEM_Net(nn.Module):
    def __init__(self, input_size=3, hidden_size=30, output_size=1, num_blocks=2):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh()]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_size))
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def crack_embedding_fn(x, y, a):
    relu_squared = torch.clamp(a - x, min=0)**2
    sign_x2 = torch.sign(y)
    return relu_squared * sign_x2

def apply_hard_constraints(u_hat_1, u_hat_2, x, y, domain_y_min, domain_y_max):
    u1 = u_hat_1 * x
    u2 = u_hat_2 * (y - domain_y_min) / (domain_y_max - domain_y_min)
    return u1, u2

def calculate_potential_energy(u1, u2, x, y, params):
    u1_x = torch.autograd.grad(u1.sum(), x, create_graph=True, allow_unused=True)[0]
    u1_y = torch.autograd.grad(u1.sum(), y, create_graph=True, allow_unused=True)[0]
    u2_x = torch.autograd.grad(u2.sum(), x, create_graph=True, allow_unused=True)[0]
    u2_y = torch.autograd.grad(u2.sum(), y, create_graph=True, allow_unused=True)[0]
    
    eps_xx, eps_yy, eps_xy = u1_x, u2_y, 0.5 * (u1_y + u2_x)

    C11 = params['E'] / ((1 + params['nu']) * (1 - 2 * params['nu'])) * (1 - params['nu'])
    C12 = params['E'] / ((1 + params['nu']) * (1 - 2 * params['nu'])) * params['nu']
    C33 = params['E'] / (2 * (1 + params['nu']))

    sigma_xx = C11 * eps_xx + C12 * eps_yy
    sigma_yy = C12 * eps_xx + C11 * eps_yy
    sigma_xy = 2 * C33 * eps_xy

    strain_energy_density = 0.5 * (sigma_xx * eps_xx + sigma_yy * eps_yy + 2 * sigma_xy * eps_xy)
    domain_area = (params['domain_x_max'] - params['domain_x_min']) * (params['domain_y_max'] - params['domain_y_min'])
    total_strain_energy = torch.mean(strain_energy_density) * domain_area

    boundary_mask = (y == params['domain_y_max'])
    u2_boundary = u2[boundary_mask]
    work_traction = torch.mean(u2_boundary * params['sigma_far_field']) * (params['domain_x_max'] - params['domain_x_min'])

    return total_strain_energy - work_traction

def calculate_analytical_K1(sigma_pa, a_m, b_m):
    """
    Tính K1 tham chiếu. Kết quả trả về có đơn vị MPa*sqrt(mm).
    """
    sigma_mpa = sigma_pa / 1e6
    a_mm = a_m * 1000
    ratio = a_m / b_m
    
    term1 = sigma_mpa * np.sqrt(np.pi * a_mm)
    term2 = 1 - 0.025 * (ratio)**2 + 0.06 * (ratio)**4
    term3 = np.sqrt(1 / np.cos(np.pi * ratio / 2))
    
    return term1 * term2 * term3

def calculate_K1_from_model(net_u1, net_u2, crack_length_a, params, device):
    """
    Tính K1 từ mô hình đã huấn luyện. Kết quả trả về có đơn vị MPa*sqrt(mm).
    """
    mu = params['E'] / (2 * (1 + params['nu']))
    kappa = 3 - 4 * params['nu']

    x_crack = torch.linspace(0, crack_length_a * 0.99, 100, device=device).unsqueeze(1)
    y_eps = 1e-8
    y_above = torch.full_like(x_crack, y_eps)
    y_below = torch.full_like(x_crack, -y_eps)

    with torch.no_grad():
        embedding_above = crack_embedding_fn(x_crack, y_above, crack_length_a)
        nn_input_above = torch.cat((x_crack, y_above, embedding_above), dim=1)
        u_hat_1_above = net_u1(nn_input_above)
        u_hat_2_above = net_u2(nn_input_above)
        _, u2_above = apply_hard_constraints(u_hat_1_above, u_hat_2_above, x_crack, y_above, params['domain_y_min'], params['domain_y_max'])

        embedding_below = crack_embedding_fn(x_crack, y_below, crack_length_a)
        nn_input_below = torch.cat((x_crack, y_below, embedding_below), dim=1)
        u_hat_1_below = net_u1(nn_input_below)
        u_hat_2_below = net_u2(nn_input_below)
        _, u2_below = apply_hard_constraints(u_hat_1_below, u_hat_2_below, x_crack, y_below, params['domain_y_min'], params['domain_y_max'])

    delta_2 = u2_above - u2_below
    r = crack_length_a - x_crack
    
    K1_apparent_pa_sqrt_m = (mu / (kappa + 1)) * torch.sqrt(2 * torch.pi / r) * delta_2

    filter_mask = (r > 0.2 * crack_length_a) & (r < 0.8 * crack_length_a)
    r_filtered = r[filter_mask]
    K1_apparent_filtered = K1_apparent_pa_sqrt_m[filter_mask]
    
    r_np = r_filtered.cpu().numpy().flatten()
    K1_np = K1_apparent_filtered.cpu().numpy().flatten()

    if len(r_np) < 2: return np.nan
    
    coeffs = np.polyfit(r_np, K1_np, 1)
    K1_pa_sqrt_m = coeffs[1]
    
    # Chuyển đổi từ Pa*sqrt(m) sang MPa*sqrt(mm)
    # 1 Pa*sqrt(m) = 1e-6 MPa * sqrt(1000 mm) = 1e-6 * sqrt(1000) MPa*sqrt(mm)
    K1_final = K1_pa_sqrt_m * (1e-6 * np.sqrt(1000))
    return K1_final

# ===================================================================
# PHẦN 2: KHỐI THỰC THI CHÍNH
# ===================================================================

if __name__ == "__main__":
    
    # -- Cài đặt và Siêu tham số --
    SEED_VALUE = 1234
    set_seed(SEED_VALUE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    params = {
        'E': 100e9,
        'nu': 0.3,
        'sigma_far_field': 10e6,
        'plate_width_b': 1.0,
        'domain_x_min': 0,
        'domain_x_max': 1.0,
        'domain_y_min': -1.0,
        'domain_y_max': 1.0,
    }

    crack_lengths_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results_list = []
    
    # --- Vòng lặp huấn luyện chính ---
    for crack_a in crack_lengths_to_test:
        print(f"\n{'='*60}\nBẮT ĐẦU HUẤN LUYỆN CHO TRƯỜNG HỢP a = {crack_a:.1f} m\n{'='*60}")
        
        # 1. Khởi tạo lại mô hình và optimizer cho mỗi lần chạy
        set_seed(SEED_VALUE)
        net_u1 = DEDEM_Net().to(device)
        net_u2 = DEDEM_Net().to(device)

        learning_rate = 0.02
        epochs = 15000  # <<< TĂNG SỐ EPOCHS ĐỂ CÓ KẾT QUẢ TỐT HƠN
        optimizer = torch.optim.Adam(list(net_u1.parameters()) + list(net_u2.parameters()), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

        # 2. Tạo lưới điểm
        nx, ny = 80, 100
        x_p = torch.linspace(params['domain_x_min'], params['domain_x_max'], nx, requires_grad=True, device=device)
        y_p = torch.linspace(params['domain_y_min'], params['domain_y_max'], ny, requires_grad=True, device=device)
        x_grid, y_grid = torch.meshgrid(x_p, y_p, indexing='ij')
        x_flat = x_grid.flatten().unsqueeze(1)
        y_flat = y_grid.flatten().unsqueeze(1)
        
        start_time = time.time()
        # 3. Chạy vòng lặp huấn luyện
        for epoch in range(epochs):
            net_u1.train()
            net_u2.train()
            optimizer.zero_grad()
            
            embedding = crack_embedding_fn(x_flat, y_flat, crack_a)
            nn_input = torch.cat((x_flat, y_flat, embedding), dim=1)
            
            u_hat_1 = net_u1(nn_input)
            u_hat_2 = net_u2(nn_input)
            
            u1, u2 = apply_hard_constraints(u_hat_1, u_hat_2, x_flat, y_flat, params['domain_y_min'], params['domain_y_max'])
            loss = calculate_potential_energy(u1, u2, x_flat, y_flat, params)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 1000 == 0:
                print(f'  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4e}')
        
        end_time = time.time()
        print(f"-> Huấn luyện hoàn tất. Thời gian: {end_time - start_time:.2f} giây. Loss cuối cùng: {loss.item():.4e}")

        # 4. Đánh giá và lưu kết quả
        net_u1.eval()
        net_u2.eval()
        
        k1_from_model = calculate_K1_from_model(net_u1, net_u2, crack_a, params, device)
        k1_analytical = calculate_analytical_K1(params['sigma_far_field'], crack_a, params['plate_width_b'])
        error_percent = abs(k1_from_model - k1_analytical) / abs(k1_analytical) * 100
        
        print(f"  - K1 (Giải tích): {k1_analytical:.2f} MPa*sqrt(mm)")
        print(f"  - K1 (Từ mô hình): {k1_from_model:.2f} MPa*sqrt(mm)")
        print(f"  - Sai số: {error_percent:.2f} %")

        results_list.append({
            'Crack Length (a/b)': crack_a / params['plate_width_b'],
            'K1_Analytical (MPa*sqrt(mm))': k1_analytical,
            'K1_From_Model (MPa*sqrt(mm))': k1_from_model,
            'Error (%)': error_percent
        })
        
        # 5. Lưu trọng số mô hình
        torch.save(net_u1.state_dict(), f'/models/net_u1_a_{crack_a:.1f}.pth')
        torch.save(net_u2.state_dict(), f'/models/net_u2_a_{crack_a:.1f}.pth')

    # --- Hoàn tất ---
    results_df = pd.DataFrame(results_list)
    print("\n--- BẢNG KẾT QUẢ TỔNG HỢP ---")
    print(results_df)
    results_df.to_csv('/history/k1_comparison_results.csv', index=False)