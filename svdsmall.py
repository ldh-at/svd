import os
import math
import csv
import glob
import numpy as np
from PIL import Image
import torch

# ---------------------------------------
# Config
# ---------------------------------------
MAX_SIDE = 512        # 너무 큰 이미지는 속도 위해 한 변을 여기로 축소 (None이면 원본 유지)
K_STEP   = 20         # k를 20 단위로
SAVE_EIG = True       # A^T A 고윳값(=sigma^2) 일부 로그 출력
OUTDIR   = "outputs"  # 결과 폴더

torch.set_float32_matmul_precision("high")


# ---------------------------------------
# Utils
# ---------------------------------------
def load_rgb_np(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0  # [0,1] HxWx3

def center_crop_square_np(A):
    H, W, _ = A.shape
    side = min(H, W)
    top = (H - side) // 2
    left = (W - side) // 2
    return A[top:top+side, left:left+side, :]

def resize_max_side_np(A, max_side):
    if max_side is None:
        return A
    H, W, _ = A.shape
    side = max(H, W)
    if side <= max_side:
        return A
    scale = max_side / side
    new_h = max(1, int(round(H * scale)))
    new_w = max(1, int(round(W * scale)))
    img = Image.fromarray((np.clip(A,0,1)*255+0.5).astype(np.uint8))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0

def to_uint8_np(A):
    A = np.clip(A, 0.0, 1.0)
    return (A * 255.0 + 0.5).astype(np.uint8)

def save_image(path, A_np):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(to_uint8_np(A_np)).save(path)

def rmse_np(A, B):
    return float(np.sqrt(np.mean((A - B) ** 2)))

def psnr_np(A, B, max_val=1.0):
    mse = float(np.mean((A - B) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)

def np_to_torch_img(A_np, device):
    # HxWxC -> CxHxW
    t = torch.from_numpy(A_np).to(device=device, dtype=torch.float32)
    return t.permute(2, 0, 1).contiguous()

def torch_to_np_img(A_t):
    # CxHxW -> HxWxC
    A_t = A_t.detach().clamp(0, 1).float()
    return A_t.permute(1, 2, 0).cpu().numpy()


# ---------------------------------------
# SVD helpers (torch)
# ---------------------------------------
def svd_square(M):
    # torch.linalg.svd: U (n,n), S (n,), Vh (n,n) for square
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return U, S, Vh

def uv_angles(U, Vh):
    # θ_i = arccos(|u_i^T v_i|), v_i = Vh[i,:]^T
    V = Vh.transpose(0, 1)    # (n,n)
    dots = torch.sum(U * V, dim=0).abs().clamp(0, 1)  # (n,)
    return torch.arccos(dots)  # radians

def reconstruct_from_indices(U, S, Vh, indices):
    if len(indices) == 0:
        return torch.zeros((U.shape[0], Vh.shape[1]), device=U.device, dtype=U.dtype)
    Uk = U[:, indices]           # n x k
    Vk = Vh[indices, :]          # k x n
    Sk = torch.diag(S[indices])  # k x k
    return Uk @ Sk @ Vk          # n x n


# ---------------------------------------
# Main
# ---------------------------------------
if __name__ == "__main__":
    print(">>> SVD Angle vs Sigma (GPU if available)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # 1) 이미지 고르기/로드
    img_path = "image.jpg"
    if not os.path.exists(img_path):
        cands = sorted(glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.png"))
        if not cands:
            raise FileNotFoundError("현재 폴더에 이미지가 없습니다 (jpg/png).")
        img_path = cands[0]
        print(f"[info] using {img_path}")

    A_np = load_rgb_np(img_path)
    H, W, _ = A_np.shape

    # 정사각형 크롭
    if H != W:
        A_np = center_crop_square_np(A_np)
    # 너무 크면 다운스케일
    A_np = resize_max_side_np(A_np, MAX_SIDE)

    H, W, _ = A_np.shape
    n = H
    print(f"[info] using square image {n} x {n}")

    os.makedirs(OUTDIR, exist_ok=True)
    save_image(os.path.join(OUTDIR, "original_square.png"), A_np)

    # torch로 이동
    A_t = np_to_torch_img(A_np, device)  # CxHxW

    # 2) 채널별 SVD & 각도, σ 수집
    U_list, S_list, Vh_list, ang_list, evals_list = [], [], [], [], []
    for c in range(3):
        M = A_t[c, :, :]  # HxW
        U, S, Vh = svd_square(M)
        U_list.append(U)
        S_list.append(S)
        Vh_list.append(Vh)
        ang_list.append(uv_angles(U, Vh))

        if SAVE_EIG:
            # (검증) A^T A 고윳값 = sigma^2 (내림차순)
            # 안정성 위해 float64로 계산 후 float32로 바꿈
            gram = (M.T @ M).to(torch.float64)
            ev = torch.linalg.eigvalsh(gram).flip(0).to(torch.float32)
            evals_list.append(ev)

    # 평균 각도/σ로 공통 인덱스 생성
    angles_mean = torch.stack(ang_list, dim=0).mean(dim=0)  # (n,)
    idx_angle_asc  = torch.argsort(angles_mean)                     # 작은 각도
    idx_angle_desc = torch.argsort(angles_mean, descending=True)    # 큰 각도

    S_mean = torch.stack(S_list, dim=0).mean(dim=0)                 # (n,)
    idx_sigma_desc = torch.argsort(S_mean, descending=True)         # 큰 σ

    idx_small = idx_angle_asc.detach().cpu().numpy().tolist()
    idx_large = idx_angle_desc.detach().cpu().numpy().tolist()
    idx_sigma = idx_sigma_desc.detach().cpu().numpy().tolist()

    # k 리스트 (20 단위)
    ks = list(range(K_STEP, n + 1, K_STEP))
    if ks[-1] != n:
        ks.append(n)

    print(f"[info] ks = {ks}")

    # 3) 루프: 재구성 & 저장 & 지표
    header = ["k", "RMSE_small", "PSNR_small", "RMSE_large", "PSNR_large", "RMSE_sigma", "PSNR_sigma"]
    rows = [header]

    with torch.no_grad():
        for k in ks:
            small_idx = idx_small[:k]
            large_idx = idx_large[:k]
            sigma_idx = idx_sigma[:k]

            rec_small = torch.zeros_like(A_t)
            rec_large = torch.zeros_like(A_t)
            rec_sigma = torch.zeros_like(A_t)

            for c in range(3):
                U, S, Vh = U_list[c], S_list[c], Vh_list[c]
                Ms = reconstruct_from_indices(U, S, Vh, small_idx)
                Ml = reconstruct_from_indices(U, S, Vh, large_idx)
                Mz = reconstruct_from_indices(U, S, Vh, sigma_idx)
                rec_small[c] = Ms
                rec_large[c] = Ml
                rec_sigma[c] = Mz

            # numpy로 변환해 지표 계산
            A_small = torch_to_np_img(rec_small)
            A_large = torch_to_np_img(rec_large)
            A_sigma = torch_to_np_img(rec_sigma)

            r_small, p_small = rmse_np(A_np, A_small), psnr_np(A_np, A_small)
            r_large, p_large = rmse_np(A_np, A_large), psnr_np(A_np, A_large)
            r_sigma, p_sigma = rmse_np(A_np, A_sigma), psnr_np(A_np, A_sigma)

            # 저장
            save_image(os.path.join(OUTDIR, f"angle_small{k}.png"), A_small)
            save_image(os.path.join(OUTDIR, f"angle_large{k}.png"), A_large)
            save_image(os.path.join(OUTDIR, f"sigma_top{k}.png"), A_sigma)

            print(f"[k={k:4d}] small  RMSE={r_small:.4f}, PSNR={p_small:6.2f} | "
                  f"large  RMSE={r_large:.4f}, PSNR={p_large:6.2f} | "
                  f"sigma  RMSE={r_sigma:.4f}, PSNR={p_sigma:6.2f}")

            rows.append([k, r_small, p_small, r_large, p_large, r_sigma, p_sigma])

    # 4) (선택) 고윳값 로그 출력
    if SAVE_EIG and len(evals_list) > 0:
        print("\n[Eigenvalues check: first 10 of sigma^2 (R channel)]")
        print(evals_list[0][:10].detach().cpu().numpy())

    # 5) CSV 저장
    csv_path = os.path.join(OUTDIR, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"\n[save] {csv_path}")
    print("✅ Done. Check outputs/ folder.")
