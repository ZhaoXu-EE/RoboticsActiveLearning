# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# # 1. 读取图像并构造未归一化的“概率高度”（这里取暗色为高概率）
# img = Image.open('/home/xu/workspace/ME455/HW3/lincoln.jpg').convert('L')
# img_arr = np.asarray(img, dtype=np.float64)

# # 反向强度，让黑色部分（数值小）对应高概率
# h, w = img_arr.shape
# unnorm = 255.0 - img_arr
# unnorm[unnorm < 0] = 0.0

# # 归一化使得 sum(unnorm) = 1
# prob = unnorm / unnorm.sum()

# # 连续 PDF f(x,y) ≈ prob[i,j] * (h*w)，因为每个像素占据面积 1/(h*w)
# def f_xy(x, y):
#     i = min(int(y * h), h - 1)
#     j = min(int(x * w), w - 1)
#     return prob[i, j] * (h * w)

# # 最大值，用于 rejection 判别
# f_max = prob.max() * (h * w)

# # 2a. Uniform 提议分布下的拒绝采样
# def rejection_uniform(n_samples):
#     samples = []
#     while len(samples) < n_samples:
#         # 多生成一些候选，加速接受
#         N = (n_samples - len(samples)) * 3
#         xs = np.random.rand(N)
#         ys = np.random.rand(N)
#         us = np.random.rand(N) * f_max

#         # 计算 f(x,y)
#         fvals = np.array([f_xy(x, y) for x,y in zip(xs, ys)])
#         accept = us < fvals

#         for x0, y0 in zip(xs[accept], ys[accept]):
#             samples.append((x0, y0))
#             if len(samples) >= n_samples:
#                 break
#     return np.array(samples)

# # 2b. Truncated Gaussian 提议分布
# mu = np.array([0.5, 0.5])
# sigma = 0.2
# # 提议分布 pdf 最大值 p_max = 1/(2πσ²)
# p_max = 1.0 / (2 * np.pi * sigma**2)
# # 信封常数 c，使得 c·p(x) >= f(x)
# c = f_max / p_max

# def rejection_gaussian(n_samples):
#     samples = []
#     while len(samples) < n_samples:
#         N = (n_samples - len(samples)) * 5
#         xs = np.random.randn(N) * sigma + mu[0]
#         ys = np.random.randn(N) * sigma + mu[1]
#         # 仅保留落在 [0,1]×[0,1] 的
#         mask = (xs>=0)&(xs<=1)&(ys>=0)&(ys<=1)
#         xs, ys = xs[mask], ys[mask]
#         if len(xs)==0: 
#             continue

#         # 计算提议 pdf p(x,y)
#         ex = np.exp(-((xs-mu[0])**2 + (ys-mu[1])**2)/(2*sigma**2))
#         pvals = ex / (2 * np.pi * sigma**2)

#         us = np.random.rand(len(xs)) * c * pvals
#         fvals = np.array([f_xy(x, y) for x,y in zip(xs, ys)])
#         accept = us < fvals

#         for x0, y0 in zip(xs[accept], ys[accept]):
#             samples.append((x0, y0))
#             if len(samples) >= n_samples:
#                 break

#     return np.array(samples)

# # 采样
# n_pts = 5000
# pts_uni = rejection_uniform(n_pts)
# pts_gau = rejection_gaussian(n_pts)

# # 3. 绘图
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# for ax, pts, title in zip(
#     axes,
#     [pts_uni, pts_gau],
#     ['Uniform Proposal', 'Gaussian Proposal']
# ):
#     ax.imshow(img_arr, cmap='gray', origin='lower', extent=[0,1,0,1])
#     ax.scatter(pts[:,0], pts[:,1], s=2, alpha=0.5)
#     ax.set_title(f'Rejection Sampling ({title})')
#     ax.set_xlim(0,1); ax.set_ylim(0,1)
# plt.tight_layout()
# plt.show()






import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import multivariate_normal

# ------------------------------------------------------------
# 1.  Read image  →  grayscale intensities in [0,1]
#    (black = high probability, white = low probability)
# ------------------------------------------------------------
IMG_PATH = "lincoln.jpg"          # adjust if necessary
img      = Image.open(IMG_PATH).convert("L")      # grayscale
I        = 1.0 - np.asarray(img, dtype=np.float32) / 255.0  # invert
H, W     = I.shape
I_max    = I.max()                # == 1 after inversion

# Helper: evaluate image-based density at (x,y) ∈ [0,1]² using nearest-pixel lookup
def p_img(xy):
    x, y = xy[..., 0], xy[..., 1]
    j = np.clip((x * (W - 1)).astype(int), 0, W - 1)
    i = np.clip(((1 - y) * (H - 1)).astype(int), 0, H - 1)  # ← 关键改动
    return I[i, j]


# ------------------------------------------------------------
# 2.  Proposal distributions
# ------------------------------------------------------------
# A) Uniform:  q_u(x) = 1  on [0,1]²
def sample_uniform(n):
    return np.random.rand(n, 2)

def accept_uniform(x):
    """Accept if u ≤ p(x)   (since p(x) ≤ 1 and q_u(x)=1)"""
    return np.random.rand(len(x)) < p_img(x)

# B) Truncated Gaussian centred at (0.5,0.5), σ=0.15 (independent)
mu      = np.array([0.5, 0.5])
sigma   = 0.15
cov     = np.eye(2) * sigma**2
rv      = multivariate_normal(mean=mu, cov=cov)

def sample_gauss(n):
    """Rejection inside [0,1]² to keep support identical."""
    out = []
    while len(out) < n:
        cand = rv.rvs(size=n)               # oversample
        cand = cand[(cand >= 0).all(1) & (cand <= 1).all(1)]
        out.extend(cand.tolist())
    return np.asarray(out[:n])

# Pre-compute c = max_x [p(x)/q_g(x)] on a fine grid for efficiency
grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 1, 400))
grid_xy        = np.dstack([grid_x, grid_y])
w              = p_img(grid_xy) / (rv.pdf(grid_xy) + 1e-12)
c_gauss        = w.max() * 1.05              # 5 % safety margin

def accept_gauss(x):
    w = p_img(x) / (rv.pdf(x) * c_gauss)
    return np.random.rand(len(x)) < w

# ------------------------------------------------------------
# 3.  Generic rejection sampler
# ------------------------------------------------------------
def rejection_sampler(N, sampler, accept_fn, batch=5000):
    samples = []
    while len(samples) < N:
        cand = sampler(batch)
        keep = cand[accept_fn(cand)]
        samples.extend(keep.tolist())
    return np.asarray(samples[:N])

# ------------------------------------------------------------
# 4.  Draw 5 000 samples with each proposal
# ------------------------------------------------------------
N_SAMPLES = 5000
samples_uniform = rejection_sampler(N_SAMPLES, sample_uniform, accept_uniform)
samples_gauss   = rejection_sampler(N_SAMPLES, sample_gauss,   accept_gauss)

# ------------------------------------------------------------
# 5.  Visualise  ——  (PDF, Uniform samples, Gaussian samples)
# ------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

# 5-A  灰度密度函数
ax[0].imshow(I,               # ← 不再 flip
             cmap='gray_r',   # ← 反色表：黑＝高概率
             origin='upper',  # ← 让行 0 在顶部 → 与采样坐标系一致
             extent=[0, 1, 0, 1])
ax[0].set_title("Probability Density Function")
ax[0].set_xlabel("x"); ax[0].set_ylabel("y")

# 5-B  Uniform→accept
ax[1].scatter(samples_uniform[:, 0], samples_uniform[:, 1], s=2, alpha=0.6, color='tab:blue')
ax[1].set_title("Rejection sampling – Uniform proposal")

# 5-C  Gaussian→accept
ax[2].scatter(samples_gauss[:, 0], samples_gauss[:, 1], s=2, alpha=0.6, color='tab:orange')
ax[2].set_title("Rejection sampling – Gaussian proposal")

for a in ax:
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_aspect('equal')


plt.tight_layout(w_pad=2)
fig.subplots_adjust(top=0.9)   # ← 给标题一点呼吸空间
# plt.show()
plt.savefig("rejection_sampling.png", dpi=300)
plt.close()

