import numpy as np
import math

from .common import cartesian_to_polar


def zernike_radial_poly(m, n, rho):
    """计算Zernike径向多项式 R_m^n(rho)"""
    if (n - m) % 2:  # 如果 (n-m) 是奇数，则该多项式为0
        return np.zeros_like(rho)
    if abs(m) > n:
        return np.zeros_like(rho)

    poly = np.zeros_like(rho)
    for k in range(0, (n - abs(m)) // 2 + 1):
        coef = ((-1) ** k * math.factorial(n - k)) / (
            math.factorial(k) *
            math.factorial((n + abs(m)) // 2 - k) *
            math.factorial((n - abs(m)) // 2 - k)
        )
        poly += coef * (rho ** (n - 2 * k))
    return poly

def zernike_poly(m, n, rho, phi):
    """计算Zernike多项式 Z_m^n(rho, phi)"""
    R = zernike_radial_poly(m, n, rho)
    if m >= 0:
        return R * np.cos(m * phi)
    else:
        return R * np.sin(-m * phi)

def get_zernike_indices(j):
    """根据Noll索引 j (1-based) 计算对应的 (n, m)"""
    # 使用预计算的查找表，适用于前15个Zernike模式
    noll_to_nm = {
        1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0),
        5: (2, -2), 6: (2, 2), 7: (3, -1), 8: (3, 1),
        9: (3, -3), 10: (3, 3), 11: (4, 0), 12: (4, 2),
        13: (4, -2), 14: (4, 4), 15: (4, -4)
    }
    return noll_to_nm.get(j, (0, 0)) # 默认返回Piston

def fit_zernike(image, pupil_mask, m_max=4):
    """
    使用最小二乘法将图像拟合到Zernike多项式。
    :param image: 2D numpy array, 输入图像
    :param pupil_mask: 2D numpy boolean array, 有效区域掩码
    :param m_max: int, 最大拟合阶数 (Noll索引)
    :return: coefficients, zernike_modes
    """
    height, width = image.shape
    Y, X = np.mgrid[:height, :width]
    X = (X - (width - 1) / 2) / ((width - 1) / 2) # 归一化到 [-1, 1]
    Y = (Y - (height - 1) / 2) / ((height - 1) / 2) # 归一化到 [-1, 1]

    rho, phi = cartesian_to_polar(X, Y)

    # 仅在单位圆内和掩码区域内进行拟合
    valid_indices = (rho <= 1.0) & pupil_mask
    valid_rho = rho[valid_indices]
    valid_phi = phi[valid_indices]
    valid_image = image[valid_indices]

    # 构建设计矩阵 (A matrix)
    num_points = np.sum(valid_indices)
    num_modes = m_max
    A = np.zeros((num_points, num_modes))

    for j in range(1, num_modes + 1):
        n, m = get_zernike_indices(j)
        # Zernike多项式在单位圆上是正交的，但通常不标准正交。
        # 为了简单起见，我们直接使用多项式值。
        # 更精确的方法是使用标准正交基或预先计算归一化系数。
        Z = zernike_poly(m, n, valid_rho, valid_phi)
        A[:, j-1] = Z

    # 使用最小二乘法求解系数 coefficients = (A^T * A)^{-1} * A^T * b
    # 或者使用更稳健的 np.linalg.lstsq
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(A, valid_image, rcond=None)
    except np.linalg.LinAlgError:
        print("拟合过程中出现线性代数错误，返回零系数。")
        coeffs = np.zeros(num_modes)

    # 重构拟合图像
    fitted_image = np.zeros_like(image, dtype=float)
    zernike_modes = {}
    for j in range(1, num_modes + 1):
        n, m = get_zernike_indices(j)
        Z_full = zernike_poly(m, n, rho, phi)
        fitted_image += coeffs[j-1] * Z_full
        zernike_modes[j] = {'coeff': coeffs[j-1], 'n': n, 'm': m, 'name': ZERNIKE_NAME_MAP.get(j, f"Z^{n}_{{{m}}}")}

    fitted_image[~valid_indices] = np.nan # 非有效区域设为NaN用于可视化

    return coeffs, fitted_image, zernike_modes

# --- 2. 定义Zernike模式名称 ---
ZERNIKE_NAME_MAP = {
    1: "Piston",
    2: "Tilt X",
    3: "Tilt Y",
    4: "Defocus",
    5: "Astigmatism 45°",
    6: "Astigmatism 0°",
    7: "Coma Y",
    8: "Coma X",
    9: "Trefoil Y",
    10: "Trefoil X",
    11: "Primary Spherical",
    12: "Secondary Astigmatism",
    13: "Secondary Astigmatism",
    14: "Tetrafoil 0°",
    15: "Tetrafoil 45°"
}