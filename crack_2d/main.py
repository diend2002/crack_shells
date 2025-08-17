import torch
import os 
import time
import pandas as pd
from src.seed import set_seed
from src.dedem import DEDEM_Net
from src.loss import calculate_potential_energy
from src.utils import *


if __name__ == "__main__":

    # -- Cài đặt và Siêu tham số --
    SEED_VALUE = 1234
    set_seed(SEED_VALUE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    params = {
        'E': 100e9, 'nu': 0.3, 'sigma_tension': 10e6,
        'plate_width_b': 1.0, 'domain_x_min': 0, 'domain_x_max': 1.0,
        'domain_y_min': -1.0, 'domain_y_max': 1.0,
    }


    results_list = []

    # --- Vòng lặp huấn luyện chính ---
    crack_a = 0.5 #m
    print(f"\n{'='*60}\nBẮT ĐẦU HUẤN LUYỆN CHO TRƯỜNG HỢP a = {crack_a:.1f} m\n{'='*60}")

    # 1. Khởi tạo lại mô hình và optimizer
    set_seed(SEED_VALUE)
    net_u1 = DEDEM_Net().to(device)
    net_u2 = DEDEM_Net().to(device)

    learning_rate = 0.002
    epochs = 50000
    optimizer = torch.optim.Adam(list(net_u1.parameters()) + list(net_u2.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

      # 2. Tạo lưới điểm không đồng đều (làm mịn tại đầu vết nứt)
      # x_flat, y_flat = create_sampling(
      #     params=params,
      #     crack_a=crack_a,
      #     grid_size={'nx': 80, 'ny': 100},
      #     concentration={'beta_x': 3.0, 'beta_y': 3.0}
      # )
    x_flat, y_flat = create_sampling(params, crack_a) #m
    x_flat = x_flat.to(device).requires_grad_(True) #m
    y_flat = y_flat.to(device).requires_grad_(True) #m

    start_time = time.time()
      # 3. Chạy vòng lặp huấn luyện
    energy_log ={}
    total_strain_log = []
    traction_log = []
    total_energy_log = []
    for epoch in range(epochs):
          net_u1.train()
          net_u2.train()
          optimizer.zero_grad()
          embedding = crack_embedding_fn(x_flat, y_flat, crack_a)
          nn_input = torch.cat((x_flat, y_flat, embedding), dim=1)
          # print("x_flat",x_flat)
          # print("y_flat",y_flat)
          u_hat_1 = net_u1(nn_input)
          u_hat_2 = net_u2(nn_input)
          # print("u_hat_1",u_hat_1)
          # print("u_hat_2",u_hat_2)
          u1, u2 = apply_hard_constraints(u_hat_1, u_hat_2, x_flat, y_flat, params['domain_y_min'], params['domain_y_max'])
          # print("u_1",u1)
          # print("u_2",u2)
          total_strain_energy, traction, loss = calculate_potential_energy(u1, u2, x_flat, y_flat, params)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(list(net_u1.parameters()) + list(net_u2.parameters()), max_norm=1.0)
          optimizer.step()
          scheduler.step()
          if (epoch + 1) % 1000 == 0:
              print(f'  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4e}')
          total_strain_log.append(total_strain_energy.item())
          traction_log.append(traction.item())
          total_energy_log.append(total_strain_energy.item() - traction.item())

    energy_log['total_strain_energy'] = total_strain_energy.item()
    energy_log['work_traction'] = traction.item()
    energy_log['total_energy'] = total_strain_energy.item() - traction.item()

    end_time = time.time()
    print(f"-> Huấn luyện hoàn tất. Thời gian: {end_time - start_time:.2f} giây. Loss cuối cùng: {loss.item():.4e}")
    energy_log = pd.DataFrame(energy_log, index=[0])
    energy_log.to_csv(f'./results/energy_log_a_{crack_a:.1f}.csv', index=False)
    # 4. Đánh giá, trực quan hóa và lưu kết quả
    net_u1.eval()
    net_u2.eval()

    k1_from_model = calculate_K1_from_model_SI(
    net_u1, net_u2,
    a_m=0.5,                # crack length 0.5 m
    params=params,
    device=device
    )
    k1_analytical = calculate_analytical_K1(params['sigma_tension'], crack_a, params['plate_width_b'])
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

      # Chỉ trực quan hóa cho một trường hợp cụ thể để tránh quá nhiều biểu đồ
    # if crack_a == 0.5:
    #         visualize(net_u1, net_u2, crack_a, params, device)

      # 5. Lưu trọng số mô hình
    torch.save(net_u1.state_dict(), f'./models/net_u1_a_{crack_a:.1f}.pth')
    torch.save(net_u2.state_dict(), f'./models/net_u2_a_{crack_a:.1f}.pth')

    # --- Hoàn tất ---
    results_df = pd.DataFrame(results_list)
    print("\n--- BẢNG KẾT QUẢ TỔNG HỢP ---")
    print(results_df)
    results_df.to_csv('./results/k1_comparison_results.csv', index=False)