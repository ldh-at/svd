# SVD Experiments

이미지에 대해 SVD(Singular Value Decomposition)를 적용해서  
- 기본적인 저랭크 복원,
- ***좌/우 특이벡터 \(u_i, v_i\) 사이의 각도 분석***,

을 해보는 실험용 레포입니다. (단, 이때 u, v의 cos_sim은 정사각행렬 분해에서만 가능함.)

---

## Features

- RGB 이미지에 대한 **thin SVD** 구현
- 각 특이벡터 쌍에 대해  
  \(\theta_i = \arccos(|u_i^\top v_i|)\) 각도를 계산하고,
  - **각도가 작은 성분만** 사용해 복원한 이미지
  - **각도가 큰 성분만** 사용해 복원한 이미지  
  를 비교
- 원본 대비 **RMSE / PSNR**로 복원 품질 측정
- 결과 이미지를 `outputs/` 폴더에 자동 저장

---

## Requirements

- Python 3.8+
- [NumPy](https://numpy.org/)
- [Pillow](https://python-pillow.org/)
- [Matplotlib](https://matplotlib.org/)  (그래프를 추가로 그리고 싶을 때)

설치는 예를 들어:

```bash
pip install numpy pillow matplotlib
