"""
Zernike 分解模块
===============

使用 `zernike` 包 (https://pypi.org/project/zernike/) 进行 Zernike 多项式展开：
- 基于包围圆中心用 scipy 亚像素平移对齐
- 使用 RZern 进行最小二乘拟合（Noll 归一化约定）
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import shift as scipy_shift
from zernike import RZern


def fit_zernike(
    img: np.ndarray,
    max_order: int = 4,
    pupil_radius_px: float | None = None,
    center_x: float | None = None,
    center_y: float | None = None,
) -> dict:
    """
    基于包围圆对光瞳图进行亚像素对齐后拟合 Zernike 多项式。

    流程:
        1. 若提供 center_x/y，用 scipy.ndimage.shift 将光斑中心亚像素平移至图像中心
        2. 按包围圆半径裁剪正方形区域
        3. 归一化强度后使用 ``zernike.RZern`` 拟合系数

    Args:
        img: 去暗场后的光强图 (2D array)
        max_order: 最大径向阶数 n（0: piston, 1: tilt, 2: defocus, ...）
        pupil_radius_px: 包围圆半径（像素）。若为 None 则自动取短边/2。
        center_x: 包围圆中心 X 坐标（像素）。若提供则做亚像素平移对齐。
        center_y: 包围圆中心 Y 坐标（像素）。同上。

    Returns:
        {
            "coeffs": np.ndarray,          # Zernike 系数（Noll 顺序）
            "noll_map": list[int],         # 1-based Noll 序号（coeffs[i] 对应 Noll noll_map[i]）
            "ntab": np.ndarray,            # 各系数的径向阶数 n
            "mtab": np.ndarray,            # 各系数的角向频率 m
            "reconstructed": np.ndarray,   # 波前重建图 (side x side)
            "residual": np.ndarray,        # 残差图 (side x side)
            "rmse": float,                 # 单位圆内重建 RMSE
            "mask": np.ndarray,            # 单位圆内 bool 掩膜 (side x side)
            "n_terms": int,                # 有效基函数数
            "side": int,                   # 裁剪后正方形边长
            "y0": int, "x0": int, "y1": int, "x1": int,  # 在原图中的裁剪范围
        }
    """
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape

    if pupil_radius_px is not None:
        radius = float(pupil_radius_px)
    else:
        radius = min(h, w) / 2.0

    # 1. 亚像素平移：将包围圆中心移到图像几何中心
    if (center_x is not None and center_y is not None
            and np.isfinite(center_x) and np.isfinite(center_y)):
        shift_y = (h - 1.0) / 2.0 - center_y
        shift_x = (w - 1.0) / 2.0 - center_x
        img_centered = scipy_shift(
            img, shift=(shift_y, shift_x),
            order=3, mode="constant", cval=0.0,
        )
    else:
        img_centered = img.copy()

    # 2. 裁剪正方形区域，边长 = 2 * radius
    side = int(round(2.0 * radius))
    side = max(side, 4)
    if side % 2 != 0:
        side += 1  # 保持偶数便于对称

    y0 = int(max(0, round((h - side) / 2.0)))
    x0 = int(max(0, round((w - side) / 2.0)))
    y1, x1 = y0 + side, x0 + side
    patch = img_centered[y0:y1, x0:x1]

    # 3. 归一化（使圆内强度之和 = 1）
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
    total = patch.sum()
    if total <= 0:
        raise ValueError("Pupil region contains no signal")
    patch_norm = patch / total

    # 4. 使用 zernike 包拟合
    cart = RZern(max_order)

    # 构建 [-1, 1] 网格
    ddx = np.linspace(-1.0, 1.0, side)
    ddy = np.linspace(-1.0, 1.0, side)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)  # ρ > 1 处设为 NaN，自动掩膜

    coeffs, res, rnk, sv = cart.fit_cart_grid(patch_norm)
    # coeffs[k] 对应 Noll 序号 k+1（1-based）

    # 5. 重建与残差
    reconstructed = cart.eval_grid(coeffs, matrix=True)

    # 单位圆掩膜
    rho = np.sqrt(xv**2 + yv**2)
    mask = rho <= 1.0

    valid_orig = patch_norm[mask]
    valid_recon = reconstructed[mask]
    rmse = float(np.sqrt(np.mean((valid_orig - valid_recon) ** 2)))

    # 6. Noll 序号映射（1-based）
    noll_map = list(range(1, cart.nk + 1))

    return {
        "coeffs": coeffs,
        "noll_map": noll_map,
        "ntab": cart.ntab.copy(),
        "mtab": cart.mtab.copy(),
        "reconstructed": reconstructed,
        "residual": patch_norm - reconstructed,
        "rmse": rmse,
        "mask": mask,
        "n_terms": cart.nk,
        "side": side,
        "y0": y0,
        "x0": x0,
        "y1": y1,
        "x1": x1,
    }


def zernike_order_label(noll_idx: int) -> str:
    """返回 Zernike 单项式的可读标签（Noll 1-based 序号）。

    Args:
        noll_idx: Noll 序号（1-based），如 1 → Piston, 4 → Defocus.

    Returns:
        模式名称字符串。
    """
    labels: dict[int, str] = {
        1: "Piston (n=0, m=0)",
        2: "Tilt Y (n=1, m=1)",
        3: "Tilt X (n=1, m=-1)",
        4: "Defocus (n=2, m=0)",
        5: "Astigmatism 0° (n=2, m=-2)",
        6: "Astigmatism 45° (n=2, m=2)",
        7: "Coma X (n=3, m=-1)",
        8: "Coma Y (n=3, m=1)",
        9: "Trefoil X (n=3, m=3)",
        10: "Trefoil Y (n=3, m=-3)",
        11: "Spherical (n=4, m=0)",
        12: "Secondary Astig 45° (n=4, m=2)",
        13: "Secondary Astig 0° (n=4, m=-2)",
        14: "Tetrafoil 0° (n=4, m=4)",
        15: "Tetrafoil 45° (n=4, m=-4)",
    }
    if noll_idx in labels:
        return labels[noll_idx]
    # 通用回退：根据 Noll → (n, m) 公式推算
    # n = ceil((-3 + sqrt(9 + 8*(k-1)))/2)  （Noll 逆映射近似）
    k = noll_idx
    n = int(np.ceil((-3.0 + np.sqrt(9.0 + 8.0 * (k - 1))) / 2.0))
    return f"n={n}, Noll#{k}"


def recommend_zernike_order(pupil_diameter_px: float) -> int:
    """根据光瞳直径推荐最大 Zernike 阶数。经验公式：n ≈ 直径/24 + 3，上限 10。"""
    return int(np.clip(round(pupil_diameter_px / 24.0 + 3), 4, 10))
