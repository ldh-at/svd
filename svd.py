import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_rgb(path):
    img = Image.open(path).convert('RGB')
    A = np.asarray(img, dtype=np.float64) / 255.0 # RGB채널로 받기 
    return A

def to_uint8(A):
    A = np.clip(A, 0.0, 1.0) # A 값들을 0~1 범위로 정규화
    return (A * 255.0 + 0.5).astype(np.uint8) # 0~255로 변환하고 반올림 보정

def rmse(A, B):
    return np.sqrt(np.mean((A - B) ** 2))

def psnr(A, B, max_val=1.0):
    mse = np.mean((A - B) ** 2)
    if mse == 0: 
        return float('inf')
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)



def _orthonormalize_columns(X, eps=1e-12):
    # 그람 슈미트로 X에서 Q로 정규 직교화 
    m, k = X.shape
    Q = []
    for j in range(k):
        v = X[:, j].astype(float).copy()
        for q in Q:
            v -= q * (q @ v)
        nrm = np.linalg.norm(v)
        if nrm > eps:
            Q.append(v / nrm)
    if len(Q) == 0:
        return np.zeros((m, 0))
    return np.column_stack(Q)




def _complete_orthonormal_basis(Q, m, eps=1e-12):
    #직교열벡터 집합의 보완기저를 만들어 (m×m) 직교행렬 완성.

    if Q.size == 0:
        # 아무것도 없으면 표준기저 반환
        return np.eye(m)

    # 먼저 Q를 정규직교화 
    Q = _orthonormalize_columns(Q, eps=eps)
    r = Q.shape[1]
    if r == m:
        return Q  # 이미 꽉 찼음

    cols = [Q[:, i] for i in range(r)]
    

    # 표준기저에서 보완
    for i in range(m):
        e = np.zeros(m)
        e[i] = 1.0
        # 기존 열들 제거
        for q in cols:
            e -= q * (q @ e)
        nrm = np.linalg.norm(e)
        if nrm > eps:
            cols.append(e / nrm)
        if len(cols) == m:
            break


    return np.column_stack(cols) 




def svd_full(A, eps=1e-12):
    m, n = A.shape

    # 우특이벡터 
    eigvals, V = np.linalg.eigh(A.T @ A)   
    # 내림차순 정렬
    eigvals = eigvals[::-1]
    V = V[:, ::-1]

    # 특이값 계산하기 
    S = np.sqrt(np.clip(eigvals, 0.0, None))
    r = int(np.sum(S > eps))  # 부동소수점으로 너무 작은 숫자 거르기 

    # 좌특이 벡터 계산 이때 rank만큼 해서 넣고 밑에서 m,m형태로 바꿈 
    U_r = np.zeros((m, r))
    for i in range(r):
        U_r[:, i] = (A @ V[:, i]) / S[i]

    # 우선 정규직교화 
    U_r = _orthonormalize_columns(U_r, eps=eps)

    # 나머지 기저 둘에 직교하는 기저로 찾아서 채움 
    U = _complete_orthonormal_basis(U_r, m, eps=eps)  

    # 시그마를 m,n으로 해서 내보냄 
    Sigma = np.zeros((m, n))
    if r > 0:
        Sigma[np.arange(r), np.arange(r)] = S[:r]

    return U, Sigma, V.T


def thin_svd(A, eps=1e-12):
   
    m, n = A.shape
    p = min(m, n) # 더 작은 값 선택 

    eigvals, V = np.linalg.eigh(A.T @ A)   
    eigvals = eigvals[::-1]
    V = V[:, ::-1]  # 내림차순 바꿔주기 

    S = np.sqrt(np.clip(eigvals, 0.0, None))
    S = S[:p]
    Vp = V[:, :p]

    # 유효한 특이값만 선택 
    idx_pos = np.where(S > eps)[0]
    if len(idx_pos) == 0: # 없으면 임의의 직교기저 
        U_full = _complete_orthonormal_basis(np.zeros((m, 0)), m, eps=eps)[:, :p]
    else:
        U_r = np.zeros((m, len(idx_pos)))
        for j, i in enumerate(idx_pos):
            U_r[:, j] = (A @ Vp[:, i]) / S[i]  # v이용해서 u 계산 
        U_r = _orthonormalize_columns(U_r, eps=eps)
        U_full = _complete_orthonormal_basis(U_r, m, eps=eps)[:, :p]

    return U_full, S, Vp.T 




def compact_svd(A, eps=1e-12):
    # thin 먼저 해주기 
    U, S, Vt = thin_svd(A, eps=eps)
    # 유효 rank만 고르기 
    idx = np.where(S > eps)[0]
    r = len(idx)

    if r == 0:
        return U[:, :0], np.zeros((0,)), Vt[:0, :]
    return U[:, idx], S[idx], Vt[idx, :]


def svd_truncate(A, k, eps=1e-12):
    # k를 받아서 모양을 자름 
    U, S, Vt = thin_svd(A, eps=eps)
    r = S.shape[0]
    k = max(0, min(k, r))

    U  = U[:, :k]
    S = np.diag(S[:k]) 
    Vt = Vt[:k, :]
    # k까지 자르기 

    return U, S, Vt



def rgb_truncated_reconstruct(A_rgb, k, eps=1e-12):
    H, W, C = A_rgb.shape
    out = np.zeros_like(A_rgb)
    for c in range(C):  # 012 순서대로 RGB
        U, Sigma, Vt = svd_truncate(A_rgb[:, :, c], k, eps=eps)  # S는 대각행렬
        out[:, :, c] = np.clip(U @ Sigma @ Vt, 0.0, 1.0)
    return out

def rgb_compact_reconstruct(A_rgb, eps=1e-12):
    H, W, C = A_rgb.shape
    out = np.zeros_like(A_rgb)
    for c in range(C):
        U, S, Vt = compact_svd(A_rgb[:, :, c], eps=eps)  #여기선 S벡터로 받음 
        if S.size == 0:
            out[:, :, c] = 0.0
        else:
            out[:, :, c] = np.clip(U @ np.diag(S) @ Vt, 0.0, 1.0)
    return out


def rgb_thin_reconstruct(A_rgb, eps=1e-12):
    H, W, C = A_rgb.shape
    out = np.zeros_like(A_rgb)
    for c in range(C):
        U, S, Vt = thin_svd(A_rgb[:, :, c], eps=eps)  # S는 벡터로 받음 
        out[:, :, c] = np.clip(U @ np.diag(S) @ Vt, 0.0, 1.0)
    return out


def overall_rmse_psnr(A, B):
    #RGB 전체 RMSE, PSNR 계산함 
    r = rmse(A, B)
    p = psnr(A, B, max_val=1.0)
    return r, p






if __name__ == "__main__":
    # 이미지 경로 지정 및 불러오기 
    img_path = "image.jpg"  
    assert os.path.exists(img_path), f"이미지 못찾음: {img_path}"
    A = load_rgb(img_path)  # float64

    #실험할 k 값들
    ks = [5, 10, 20, 50, 100]

    #이미지 저장 폴더
    os.makedirs("outputs", exist_ok=True)
    Image.fromarray(to_uint8(A)).save("outputs/original.png")


    # Full SVD 복원
    U, Sigma, Vt = svd_full(A[:, :, 0])  # R채널 기준 예시
    A_full = np.zeros_like(A)
    for c in range(3):
        U, Sigma, Vt = svd_full(A[:, :, c])
        A_full[:, :, c] = np.clip(U @ Sigma @ Vt, 0.0, 1.0)

    Image.fromarray(to_uint8(A_full)).save("outputs/rec_full.png")
    r_full, p_full = overall_rmse_psnr(A, A_full)
    print(f"Full SVD → RMSE={r_full:.6f}, PSNR={p_full:.2f} dB")


    singular_values_all = []
    for c in range(3):
        _, Sigma_c, _ = svd_full(A[:, :, c])
        singular_values_all.append(np.diag(Sigma_c)[:100])
    mean_sv = np.mean(singular_values_all, axis=0)
    print("\nSingular Values 100개:")
    print(np.round(mean_sv, 6))    


    # Thin SVD 복원
    A_thin = rgb_thin_reconstruct(A)
    Image.fromarray(to_uint8(A_thin)).save("outputs/rec_thin.png")
    r_thin, p_thin = overall_rmse_psnr(A, A_thin)
    print(f"Thin SVD → RMSE={r_thin:.6f}, PSNR={p_thin:.2f} dB")

    # Compact SVD 복원
    A_compact = rgb_compact_reconstruct(A)
    Image.fromarray(to_uint8(A_compact)).save("outputs/rec_compact.png")
    r_comp, p_comp = overall_rmse_psnr(A, A_compact)
    print(f"Compact SVD → RMSE={r_comp:.6f}, PSNR={p_comp:.2f} dB")

    # k별 실험 및 지표 저장 
    grid_imgs = [A]
    grid_titles = ["Original"]
    for k in ks:
        B = rgb_truncated_reconstruct(A, k)
        out_file = f"outputs/rec_k{k}.png"
        Image.fromarray(to_uint8(B)).save(out_file)

        r_all, p_all = overall_rmse_psnr(A, B)
        print(f"\n=== k={k} ===")
        print(f"RMSE={r_all:.6f}, PSNR={p_all:.2f} dB")

        grid_imgs.append(B)
        grid_titles.append(f"k={k}")




