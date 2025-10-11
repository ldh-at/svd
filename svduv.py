import os
import glob
import numpy as np
from PIL import Image

# -----------------------------
# 기본 유틸
# -----------------------------
def load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0  # [0,1]

def to_uint8(A):
    return (np.clip(A, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

def rmse(A, B):
    return np.sqrt(np.mean((A - B) ** 2))

def psnr(A, B, max_val=1.0):
    mse = np.mean((A - B) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)

def overall_rmse_psnr(A, B):
    return rmse(A, B), psnr(A, B, 1.0)

def center_crop_to_square(A_rgb):
    """가운데 기준으로 정사각형 크롭"""
    H, W, _ = A_rgb.shape
    side = min(H, W)
    top = (H - side) // 2
    left = (W - side) // 2
    return A_rgb[top:top+side, left:left+side, :]

# -----------------------------
# SVD (Thin)
# -----------------------------
def thin_svd(A, eps=1e-12):
    m, n = A.shape
    p = min(m, n)
    eigvals, V = np.linalg.eigh(A.T @ A)
    eigvals = eigvals[::-1]
    V = V[:, ::-1]
    S = np.sqrt(np.clip(eigvals, 0.0, None))[:p]
    Vp = V[:, :p]
    idx_pos = np.where(S > eps)[0]
    if len(idx_pos) == 0:
        return np.eye(m)[:, :p], np.zeros((p,)), np.eye(n)[:p, :]
    U_r = np.zeros((m, len(idx_pos)))
    for j, i in enumerate(idx_pos):
        U_r[:, j] = (A @ Vp[:, i]) / S[i]
    Q, _ = np.linalg.qr(U_r)
    return Q, S, Vp.T

# -----------------------------
# 각도 계산
# -----------------------------
def uv_angles_from_svd(U, Vt):
    """θ_i = arccos(|u_i^T v_i|)"""
    n = U.shape[1]
    dots = np.clip(np.abs(np.sum(U * Vt.T, axis=0)), 0.0, 1.0)
    return np.arccos(dots)

# -----------------------------
# 복원 (특정 인덱스만)
# -----------------------------
def reconstruct_from_indices(U, S_vec, Vt, indices):
    if len(indices) == 0:
        return np.zeros((U.shape[0], Vt.shape[1]))
    Uk = U[:, indices]
    VkT = Vt[indices, :]
    Sk = np.diag(S_vec[indices])
    return Uk @ Sk @ VkT

def rgb_reconstruct_from_indices(A_rgb, indices_by_channel, eps=1e-12):
    H, W, C = A_rgb.shape
    out = np.zeros_like(A_rgb)
    for c in range(C):
        U, S, Vt = thin_svd(A_rgb[:, :, c], eps)
        out[:, :, c] = np.clip(reconstruct_from_indices(U, S, Vt, indices_by_channel[c]), 0.0, 1.0)
    return out

# -----------------------------
# 메인
# -----------------------------
if __name__ == "__main__":
    print(">>> Start Angle-based SVD Experiment")

    img_path = "image.jpg"
    if not os.path.exists(img_path):
        candidates = sorted(glob.glob("*.jpg") + glob.glob("*.png"))
        if not candidates:
            raise FileNotFoundError("No image found in current directory.")
        img_path = candidates[0]
        print(f"[info] Using {img_path}")

    A = load_rgb(img_path)
    H, W, _ = A.shape
    if H != W:
        print(f"[info] Not square (H={H}, W={W}) → center-cropping.")
        A = center_crop_to_square(A)

    os.makedirs("outputs", exist_ok=True)

    # --- 각도 계산 ---
    print("[step] computing U–V angles per channel...")
    angles_per_channel = []
    for c in range(3):
        Uc, Sc, Vtc = thin_svd(A[:, :, c])
        angles_per_channel.append(uv_angles_from_svd(Uc, Vtc))
    angles_mean = np.mean(np.stack(angles_per_channel, axis=0), axis=0)

    n = angles_mean.shape[0]
    idx_sorted = np.argsort(angles_mean)  # 작은 각도→큰 각도

    ks = [5, 10, 20]
    for k in ks:
        idx_small = idx_sorted[:k].tolist()
        idx_large = idx_sorted[::-1][:k].tolist()

        A_small = rgb_reconstruct_from_indices(A, [idx_small]*3)
        A_large = rgb_reconstruct_from_indices(A, [idx_large]*3)

        out_small = f"outputs/angle_small{k}.png"
        out_large = f"outputs/angle_large{k}.png"

        Image.fromarray(to_uint8(A_small)).save(out_small)
        Image.fromarray(to_uint8(A_large)).save(out_large)

        r_small, p_small = overall_rmse_psnr(A, A_small)
        r_large, p_large = overall_rmse_psnr(A, A_large)

        print(f"\n=== k={k} ===")
        print(f"Small-angle {k} → RMSE={r_small:.6f}, PSNR={p_small:.2f} dB")
        print(f"Large-angle {k} → RMSE={r_large:.6f}, PSNR={p_large:.2f} dB")
        print(f"indices (small): {idx_small}")
        print(f"indices (large): {idx_large}")

    print("\n✅ Done! Check the 'outputs/' folder.")
