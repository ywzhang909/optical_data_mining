from collections import namedtuple

import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def _double_erf_pulse(t, P0, A, t1, sigma1, t2, sigma2):
    return P0 + (A / 2.0) * (erf((t - t1) / sigma1) - erf((t - t2) / sigma2))

def _double_sigmoid(t, P0, A1, k, t1, k1, t2, k2):
    rise = 1 / (1 + np.exp(-k1 * (t - t1)))
    fall = 1 / (1 + np.exp(-k2 * (t - t2)))
    return P0 + A1 * (rise - k * fall)

def _gaussian(x, mu, sigma, A, b):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + b


class BaseFitFunc:
    ParamName = []

    def __init__(self, func, y_data, x_data=None):
        self.func = func
        self.y = y_data
        self.x = x_data if x_data is not None else np.arange(len(y_data))
        self._params = namedtuple(self.__class__.__name__ + 'Params', self.ParamName)

    def __call__(self, x):
        return self.func(x, *self._params)
    
    def bounds(self):
        pass
    
    def guess_params(self):
        pass
        
    def fit(self):
        p0 = self.guess_params()
        bounds_low, bounds_high = self.bounds()
        popt, pcov = curve_fit(
            self.func, self.x, self.y,
            p0=p0,
            bounds=(bounds_low, bounds_high)
        )
        self._params = self.params(*popt)
        return self._params
    
    def plot_fit_res(self, samples = 10_000):
        t_fine = np.linspace(np.min(self.x), np.max(self.x), samples)
        P_fine = self.func(t_fine, *self._params)
        plt.figure(figsize=(10, 5))
        plt.scatter(self.x, self.y, s=15, alpha=0.6, label='Measured Data')
        plt.plot(t_fine, P_fine, 'r-', linewidth=2, label='Fit Data')
        plt.title(f'{self.__class__.__name__} R2 score: {r2_score(self.y, self.func(self.x, *self._params)):.3f}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
    
    @property
    def params(self):
        return self._params
    

class DoubleErfPulse(BaseFitFunc):
    r"""
    对于**带上升沿和下降沿的功率曲线（如激光脉冲、光开关响应等）**，且**上升/下降时间未知**，推荐使用 **"平滑阶跃函数"组合模型**。这类函数既能描述平台区（恒定功率），又能用可调参数刻画上升/下降沿的**形状和时间位置**。

    **双误差函数模型（Double Error Function）**

    $
    P(t) = P_0 + \frac{A}{2} \left[
    \operatorname{erf}\left( \frac{t - t_1}{\sigma_1} \right)
    - \operatorname{erf}\left( \frac{t - t_2}{\sigma_2} \right)
    \right]
    $

    #### 🔍 参数含义：
    | 参数 | 物理意义 |
    |------|----------|
    | $P_0$ | 基线功率（脉冲前/后的背景） |
    | $A$ | 脉冲幅度（平台高度） |
    | $t_1$ | **上升沿中心时间**（50% 上升点附近） |
    | $\sigma_1$ | **上升沿陡峭度**（越小越陡，$\tau_{\text{rise}} \propto \sigma_1$） |
    | $t_2$ | **下降沿中心时间**（50% 下降点附近） |
    | $\sigma_2$ | **下降沿陡峭度** |
    """
    ParamName = ['P0', 'A', 't1', 'sigma1', 't2', 'sigma2']

    def __init__(self, y_data, x_data=None):
        super().__init__(_double_erf_pulse, y_data, x_data)
    
    def bounds(self):
        bounds_low = [
            np.min(self.y) - 1,   # P0
            0.1,                  # A > 0
            np.min(self.x),       # t1
            1e-6,                 # sigma1 > 0
            np.min(self.x),       # t2
            1e-6                  # sigma2 > 0
        ]
        bounds_high = [
            np.max(self.y),       # P0   
            (np.max(self.y) - np.min(self.y)),  # A
            np.max(self.x),       # t1
            (np.max(self.x) - np.min(self.x)) / 2,  # sigma1
            np.max(self.x),       # t2
            (np.max(self.x) - np.min(self.x)) / 2   # sigma2
        ]
        return bounds_low, bounds_high
    
    def guess_params(self):
        P_min, P_max = np.min(self.y), np.max(self.y)
        P0_est = P_min
        A_est = P_max - P_min
        
        # 找 50% 幅度对应的大概时间
        mid_level = P0_est + 0.5 * A_est
        idx_rise = np.where(self.y >= mid_level)[0][0]  # 第一个超过50%的点（上升）
        idx_fall = np.where(self.y >= mid_level)[0][-1] # 最后一个超过50%的点（下降）
        
        t1_est = self.x[idx_rise]
        t2_est = self.x[idx_fall]
        
        # 估计 sigma：假设上升/下降跨越 ~5个数据点
        dt = np.mean(np.diff(self.x))
        sigma1_est = sigma2_est = 2 * dt  # 初始猜测较平缓
        
        return [P0_est, A_est, t1_est, sigma1_est, t2_est, sigma2_est]
    
    @property
    def rise_time(self):
        t_fine = np.linspace(np.min(self.x), np.max(self.x), 10000)
        P_fine = self.func(t_fine, *self._params)
        def find_time_for_fraction(frac):
            target = self.params.P0 + frac * self.params.A
            # 在拟合曲线上插值找时间
            idx = np.argmin(np.abs(P_fine - target))
            return t_fine[idx]
        t_rise_10 = find_time_for_fraction(0.1)
        t_rise_90 = find_time_for_fraction(0.9)
        rise_time = t_rise_90 - t_rise_10

        return rise_time
    
    @property
    def fall_time(self):
        # 注意：下降沿是从高到低，所以要找下降段的90%和10%
        # 更准确做法：在 t > (t1+t2)/2 区域找
        t_mid = (self.params.t1 + self.params.t2) / 2
        t_fine = np.linspace(np.min(self.x), np.max(self.x), 10000)
        P_fine = self.func(t_fine, *self._params)
        mask_fall = t_fine > t_mid
        t_fall_90 = t_fine[mask_fall][np.argmin(np.abs(P_fine[mask_fall] - (self.params.P0 + 0.9*self.params.A)))]
        t_fall_10 = t_fine[mask_fall][np.argmin(np.abs(P_fine[mask_fall] - (self.params.P0 + 0.1*self.params.A)))]
        fall_time = t_fall_10 - t_fall_90  # 应为正数

        return fall_time
    

class DoubleSigmoid(BaseFitFunc):
    r"""
    双 Sigmoid 脉冲模型

    $
    P(t) = P_0 + A \left[
    \frac{1}{1 + e^{-k_1 (t - t_1)}}
    - \frac{1}{1 + e^{-k_2 (t - t_2)}}
    \right]
    $

    #### 🔍 参数说明：
    | 参数 | 含义 |
    |------|------|
    | $P_0$ | 基线功率（脉冲前/后的背景值） |
    | $A$ | 脉冲平台高度（幅度） |
    | $t_1$ | 上升沿中心时间（Sigmoid 中点，≈50% 上升点） |
    | $k_1$ | 上升沿陡峭度（越大越陡，$k_1 > 0$） |
    | $t_2$ | 下降沿中心时间（Sigmoid 中点，≈50% 下降点） |
    | $k_2$ | 下降沿陡峭度（越大越陡，$k_2 > 0$） |
    """

    ParamName = ['P0', 'A1', 'k', 't1', 'k1', 't2', 'k2']

    def __init__(self, y_data, x_data=None):
        super().__init__(_double_sigmoid, y_data, x_data)

    def bounds(self):
        avg_t = np.diff(self.x).mean()
        bounds_low = [
            np.min(self.y) - 0.5,      # P0
            0.1,              # A > 0
            1,
            np.min(self.x),            # t1
            0.1,              # k1 > 0
            np.min(self.x),            # t2
            0.1               # k2 > 0
        ]
        bounds_high = [
            np.max(self.y),            # P0
            np.max(self.y) - np.min(self.y),  # A
            2,
            np.max(self.x),            # t1
            10.0 / avg_t,    # k1 上限（非常陡）
            np.max(self.x),            # t2
            10.0 / avg_t     # k2 上限
        ]
        return bounds_low, bounds_high
    
    def guess_params(self):
        """
        从数据粗略估计双 Sigmoid 的初始参数
        """
        P_min, P_max = np.min(self.y), np.max(self.y)
        P0_est = P_min
        A_est = P_max - P_min
        
        # 找 50% 幅度对应的时间（近似 t1, t2）
        mid_level = P0_est + 0.5 * A_est
        above_mid = np.where(self.y >= mid_level)[0]
        
        if len(above_mid) == 0:
            raise ValueError("无法估计脉冲位置")
        
        t1_est = self.x[above_mid[0]]   # 第一个超过50%的点 → 上升中点
        t2_est = self.x[above_mid[-1]]  # 最后一个超过50%的点 → 下降中点
        
        # 估计 k：假设上升/下降跨越 ~5个时间单位
        dt = np.mean(np.diff(self.x))
        k1_est = k2_est = 2.0 / dt  # 初始陡峭度（经验值）
        
        return [P0_est, A_est, 1, t1_est, k1_est, t2_est, k2_est]


class Gaussian(BaseFitFunc):
    r"""
    高斯函数模型

    $
    P(t) = P_0 + A \exp\left[ - \frac{(t - t_0)^2}{2 \sigma^2} \right]
    $
    #### 🔍 参数说明：
    | 参数 | 含义 |
    |------|------|
    | $P_0$ | 基线功率（脉冲前/后的背景值） |
    | $A$ | 脉冲平台高度（幅度） |
    | $t_0$ | 高斯中心时间（Sigmoid 中点，≈50% 上升点） |
    | $\sigma$ | 高斯宽度（标准差） |
    """
    ParamName = ['P0', 'A', 'mu', 'sigma']

    def __init__(self, y_data, x_data=None):
        super().__init__(_gaussian, y_data, x_data)

    def bounds(self):
        bounds_low = [
            np.min(self.y) - 0.5,      # P0
            0.1,              # A > 0
            np.min(self.x),            # mu
            0.1               # sigma > 0
        ]
        bounds_high = [
            np.max(self.y),            # P0
            np.max(self.y) - np.min(self.y),  # A
            np.max(self.x),            # mu
            np.max(self.x) - np.min(self.x)  # sigma
        ]
        return bounds_low, bounds_high
    
    def guess_params(self):
        """
        从数据粗略估计高斯函数的初始参数
        """
        P_min, P_max = np.min(self.y), np.max(self.y)
        P0_est = P_min
        A_est = P_max - P_min
        mu_est = np.mean(self.x)
        sigma_est = np.std(self.x)
        return [P0_est, A_est, mu_est, sigma_est]
    
    @property
    def diameter(self):
        diameter = 2 * self.params.sigma
        return diameter

if __name__ == '__main__':
    dummy_data_x = np.linspace(0, 10, 100)
    dummy_data_y = _gaussian(dummy_data_x, 1, 1, 1,0) + np.random.normal(0, 0.1, size=dummy_data_x.shape)

    fit_func = Gaussian(dummy_data_x, dummy_data_y)
    fit_func.fit()
    print(fit_func.params)

    fit_func.plot_fit_res()
    print(fit_func.diameter)
