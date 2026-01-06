"""
Ryven算子定义文件
将notebooks/data.ipynb中的计算函数转换为Ryven节点
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass, median_filter, generic_filter
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.signal import lombscargle
from scipy.fft import fft2, ifftshift, fftshift, ifft2
from scipy.interpolate import RegularGridInterpolator, interp1d
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import r2_score, mean_squared_error
import cv2
import math
import matplotlib.pyplot as plt
from pathlib import Path
import swifter

try:
    from ryven.main import Node
    from ryven.main import register_node
    print("使用 ryven.main 导入方式")
except ImportError:
    try:
        from ryven.NENV import *
        from ryven.core.reaction import Reaction
        from ryven.gui.node_datas import *
        from ryven.gui.icons import *
        from ryven.Style import *
        print("使用 ryven.NENV 导入方式")
    except ImportError:
        # 定义基本的Node类以避免导入错误
        class Node:
            def __init__(self, params):
                self.params = params
            def update_event(self, inp=-1):
                pass
            def set_output_val(self, index, value):
                pass
            def input(self, index):
                return None
        
        class NodeInputBP:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
        
        class NodeOutputBP:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
        
        def register_node(cls):
            print(f"模拟注册节点: {cls.title}")
        
        print("使用模拟的Node类和register_node函数")

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 从data.py导入相关函数
from notebooks.data import (
    adaptive_background_subtraction,
    calculate_strehl_ratio_with_energy_conservation,
    calculate_bpp_from_pupil_and_focal,
    compute_wavefront_gradient_tie
)

class DataAnalysisNodes:
    """数据分析节点集合"""

    # === 功率数据分析节点 ===
    class PowerDataAnalysisNode(Node):
        """功率数据分析节点"""
        title = '功率数据分析'
        color = '#FF6B6B'
        
        init_inputs = [
            NodeInputBP('time_data', '时间数据'),
            NodeInputBP('power_data', '功率数据'),
            NodeInputBP('fit_method', '拟合方法', default='sigmoid')
        ]
        
        init_outputs = [
            NodeOutputBP('fitted_data', '拟合数据'),
            NodeOutputBP('parameters', '拟合参数'),
            NodeOutputBP('rise_time', '上升时间'),
            NodeOutputBP('fall_time', '下降时间')
        ]
        
        def update_event(self, inp=-1):
            time_data = self.input(0)
            power_data = self.input(1)
            fit_method = self.input(2)
            
            if time_data is not None and power_data is not None:
                if fit_method == 'sigmoid':
                    # 双Sigmoid拟合
                    result = self.fit_double_sigmoid(time_data, power_data)
                    self.set_output_val(0, result['fitted_curve'])
                    self.set_output_val(1, result['params'])
                    self.set_output_val(2, result['rise_time'])
                    self.set_output_val(3, result['fall_time'])
                else:
                    # 其他拟合方法
                    self.set_output_val(0, power_data)
                    self.set_output_val(1, {})
                    self.set_output_val(2, 0.0)
                    self.set_output_val(3, 0.0)
            else:
                self.set_output_val(0, None)
                self.set_output_val(1, {})
                self.set_output_val(2, 0.0)
                self.set_output_val(3, 0.0)
        
        def fit_double_sigmoid(self, t, P):
            """双Sigmoid拟合函数"""
            def double_sigmoid(t, P0, A1, k, t1, k1, t2, k2):
                rise = 1 / (1 + np.exp(-k1 * (t - t1)))
                fall = 1 / (1 + np.exp(-k2 * (t - t2)))
                return P0 + A1 * (rise - k * fall)
            
            def estimate_initial_params_sigmoid(t, P):
                P_min, P_max = np.min(P), np.max(P)
                P0_est = P_min
                A_est = P_max - P_min
                
                # 找 50% 幅度对应的时间（近似 t1, t2）
                mid_level = P0_est + 0.5 * A_est
                above_mid = np.where(P >= mid_level)[0]
                
                if len(above_mid) == 0:
                    raise ValueError("无法估计脉冲位置")
                
                t1_est = t[above_mid[0]]   # 第一个超过50%的点 → 上升中点
                t2_est = t[above_mid[-1]]  # 最后一个超过50%的点 → 下降中点
                
                # 估计 k：假设上升/下降跨越 ~5个时间单位
                dt = np.mean(np.diff(t))
                k1_est = k2_est = 2.0 / dt  # 初始陡峭度（经验值）
                
                return [P0_est, A_est, 1, t1_est, k1_est, t2_est, k2_est]
            
            def bounds_sigmoid(t, P):
                avg_t = np.diff(t).mean()
                bounds_low = [
                    np.min(P) - 0.5,      # P0
                    0.1,              # A > 0
                    1,
                    np.min(t),            # t1
                    0.1,              # k1 > 0
                    np.min(t),            # t2
                    0.1               # k2 > 0
                ]
                bounds_high = [
                    np.max(P),            # P0
                    np.max(P) - np.min(P),  # A
                    2,
                    np.max(t),            # t1
                    10.0 / avg_t,    # k1 上限（非常陡）
                    np.max(t),            # t2
                    10.0 / avg_t     # k2 上限
                ]
                return bounds_low, bounds_high
            
            p0 = estimate_initial_params_sigmoid(t, P)
            
            try:
                popt, pcov = curve_fit(
                    double_sigmoid, t, P,
                    p0=p0,
                    bounds=bounds_sigmoid(t, P),
                    maxfev=5000
                )
                P0, A1, k, t1, k1, t2, k2 = popt
                
                t_fine = np.linspace(np.min(t), np.max(t), 10000)
                P_fine = double_sigmoid(t_fine, *popt)
                
                def find_time_for_fraction(frac):
                    target = P0 + frac * A1
                    # 在拟合曲线上插值找时间
                    idx = np.argmin(np.abs(P_fine - target))
                    return t_fine[idx]
                
                t_rise_10 = find_time_for_fraction(0.1)
                t_rise_90 = find_time_for_fraction(0.9)
                rise_time = t_rise_90 - t_rise_10
                
                # 注意：下降沿是从高到低，所以要找下降段的90%和10%
                # 更准确做法：在 t > (t1+t2)/2 区域找
                t_mid = (t1 + t2) / 2
                mask_fall = t_fine > t_mid
                t_fall_90 = t_fine[mask_fall][np.argmin(np.abs(P_fine[mask_fall] - (P0 + 0.9*A1*k)))]
                t_fall_10 = t_fine[mask_fall][np.argmin(np.abs(P_fine[mask_fall] - (P0 + 0.1*A1*k)))]
                fall_time = t_fall_10 - t_fall_90  # 应为正数
                
                return {
                    'fitted_curve': P_fine,
                    'params': popt,
                    'rise_time': rise_time,
                    'fall_time': fall_time,
                    'r2_score': r2_score(P, double_sigmoid(t, *popt))
                }
            except Exception as e:
                print(f"拟合失败: {e}")
                return {
                    'fitted_curve': P,
                    'params': [],
                    'rise_time': 0.0,
                    'fall_time': 0.0,
                    'r2_score': 0.0
                }

    # === 图像预处理节点 ===
    class ImagePreprocessingNode(Node):
        """图像预处理节点"""
        title = '图像预处理'
        color = '#4ECDC4'
        
        init_inputs = [
            NodeInputBP('image', '图像'),
            NodeInputBP('method', '预处理方法', default='background_subtraction'),
            NodeInputBP('kernel_size', '核大小', dtype='int', default=21),
            NodeInputBP('sigma_factor', '阈值倍数', dtype='float', default=3.0)
        ]
        
        init_outputs = [
            NodeOutputBP('processed_image', '预处理图像'),
            NodeOutputBP('background', '背景'),
            NodeOutputBP('mask', '掩码')
        ]
        
        def update_event(self, inp=-1):
            image = self.input(0)
            method = self.input(1)
            kernel_size = int(self.input(2))
            sigma_factor = float(self.input(3))
            
            if image is not None:
                if method == 'background_subtraction':
                    denoised, background, mask = adaptive_background_subtraction(
                        image,
                        kernel_size=kernel_size,
                        sigma_factor=sigma_factor
                    )
                    self.set_output_val(0, denoised)
                    self.set_output_val(1, background)
                    self.set_output_val(2, mask)
                else:
                    # 其他预处理方法
                    self.set_output_val(0, image)
                    self.set_output_val(1, np.zeros_like(image))
                    self.set_output_val(2, np.ones_like(image, dtype=bool))
            else:
                self.set_output_val(0, None)
                self.set_output_val(1, None)
                self.set_output_val(2, None)

    # === D4σ特征提取节点 ===
    class D4SigmaFeatureNode(Node):
        """D4σ特征提取节点"""
        title = 'D4σ特征提取'
        color = '#45B7D1'
        
        init_inputs = [
            NodeInputBP('image', '图像'),
            NodeInputBP('pixel_size_um', '像素尺寸(μm)', dtype='float', default=1.0)
        ]
        
        init_outputs = [
            NodeOutputBP('d4sigma_x', 'X方向D4σ直径'),
            NodeOutputBP('d4sigma_y', 'Y方向D4σ直径'),
            NodeOutputBP('center_x', 'X中心坐标'),
            NodeOutputBP('center_y', 'Y中心坐标'),
            NodeOutputBP('total_power', '总功率')
        ]
        
        def update_event(self, inp=-1):
            image = self.input(0)
            pixel_size_um = float(self.input(1))
            
            if image is not None:
                # 计算D4σ特征
                features = self.calculate_d4sigma_features(image, pixel_size_um)
                self.set_output_val(0, features['d4sigma_x'])
                self.set_output_val(1, features['d4sigma_y'])
                self.set_output_val(2, features['center_x'])
                self.set_output_val(3, features['center_y'])
                self.set_output_val(4, features['total_power'])
            else:
                self.set_output_val(0, 0.0)
                self.set_output_val(1, 0.0)
                self.set_output_val(2, 0.0)
                self.set_output_val(3, 0.0)
                self.set_output_val(4, 0.0)
        
        def calculate_d4sigma_features(self, img, pixel_size_um=1.0):
            """计算D4σ特征"""
            img = np.array(img)
            if img.ndim == 3:
                img = np.mean(img, axis=2)  # 转换为灰度图
            
            # 计算总功率
            total_power = np.sum(img)
            
            if total_power == 0:
                return {
                    'd4sigma_x': 0.0,
                    'd4sigma_y': 0.0,
                    'center_x': 0.0,
                    'center_y': 0.0,
                    'total_power': 0.0
                }
            
            # 计算重心
            y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]]
            center_y = np.sum(y * img) / total_power
            center_x = np.sum(x * img) / total_power
            
            # 计算X方向的D4σ
            x_profile = np.sum(img, axis=0)  # 沿Y轴积分
            x_coords = np.arange(len(x_profile)) * pixel_size_um
            x_power = np.sum(x_profile)
            
            if x_power > 0:
                x_center = np.sum(x_coords * x_profile) / x_power
                x_second_moment = np.sum(x_profile * (x_coords - x_center)**2) / x_power
                x_width = 2 * np.sqrt(2 * x_second_moment)  # D4σ = 4 * sigma
            else:
                x_width = 0.0
            
            # 计算Y方向的D4σ
            y_profile = np.sum(img, axis=1)  # 沿X轴积分
            y_coords = np.arange(len(y_profile)) * pixel_size_um
            y_power = np.sum(y_profile)
            
            if y_power > 0:
                y_center = np.sum(y_coords * y_profile) / y_power
                y_second_moment = np.sum(y_profile * (y_coords - y_center)**2) / y_power
                y_width = 2 * np.sqrt(2 * y_second_moment)  # D4σ = 4 * sigma
            else:
                y_width = 0.0
            
            return {
                'd4sigma_x': x_width,
                'd4sigma_y': y_width,
                'center_x': center_x,
                'center_y': center_y,
                'total_power': total_power
            }

    # === 椭圆拟合节点 ===
    class EllipseFitNode(Node):
        """椭圆拟合节点"""
        title = '椭圆拟合'
        color = '#96CEB4'
        
        init_inputs = [
            NodeInputBP('image', '图像'),
            NodeInputBP('threshold', '阈值', dtype='float', default=0.5)
        ]
        
        init_outputs = [
            NodeOutputBP('ellipse_params', '椭圆参数'),
            NodeOutputBP('major_axis', '长轴'),
            NodeOutputBP('minor_axis', '短轴'),
            NodeOutputBP('eccentricity', '离心率')
        ]
        
        def update_event(self, inp=-1):
            image = self.input(0)
            threshold = float(self.input(1))
            
            if image is not None:
                # 进行椭圆拟合
                ellipse_params = self.fit_ellipse(image, threshold)
                self.set_output_val(0, ellipse_params)
                self.set_output_val(1, ellipse_params.get('major_axis', 0.0))
                self.set_output_val(2, ellipse_params.get('minor_axis', 0.0))
                self.set_output_val(3, ellipse_params.get('eccentricity', 0.0))
            else:
                self.set_output_val(0, {})
                self.set_output_val(1, 0.0)
                self.set_output_val(2, 0.0)
                self.set_output_val(3, 0.0)
        
        def fit_ellipse(self, img, threshold=0.5):
            """椭圆拟合"""
            img = np.array(img)
            if img.ndim == 3:
                img = np.mean(img, axis=2)  # 转换为灰度图
            
            # 标准化图像
            img_norm = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-10)
            
            # 创建二值掩码
            mask = img_norm > threshold
            
            # 寻找轮廓
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if len(contours) > 0:
                # 选择最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                
                if len(largest_contour) >= 5:  # 至少需要5个点才能拟合椭圆
                    try:
                        # 拟合椭圆
                        ellipse = cv2.fitEllipse(largest_contour)
                        (center_x, center_y), (axis1, axis2), angle = ellipse
                        
                        # 确保axis1是长轴
                        major_axis = max(axis1, axis2)
                        minor_axis = min(axis1, axis2)
                        
                        # 计算离心率
                        eccentricity = np.sqrt(1 - (minor_axis / (major_axis + 1e-10))**2)
                        
                        return {
                            'center_x': center_x,
                            'center_y': center_y,
                            'major_axis': major_axis,
                            'minor_axis': minor_axis,
                            'angle': angle,
                            'eccentricity': eccentricity
                        }
                    except:
                        pass
            
            return {
                'center_x': 0.0,
                'center_y': 0.0,
                'major_axis': 0.0,
                'minor_axis': 0.0,
                'angle': 0.0,
                'eccentricity': 0.0
            }

    # === Strehl比计算节点 ===
    class StrehlRatioNode(Node):
        """Strehl比计算节点"""
        title = 'Strehl比计算'
        color = '#FFEAA7'
        
        init_inputs = [
            NodeInputBP('pupil_image', '光瞳图像'),
            NodeInputBP('focus_image', '焦平面图像'),
            NodeInputBP('pixel_size_pupil_um', '光瞳像素尺寸(μm)', dtype='float', default=5.5),
            NodeInputBP('pixel_size_focus_um', '焦面像素尺寸(μm)', dtype='float', default=3.45),
            NodeInputBP('focal_length_mm', '焦距(mm)', dtype='float', default=100.0),
            NodeInputBP('wavelength_um', '波长(μm)', dtype='float', default=1.064)
        ]
        
        init_outputs = [
            NodeOutputBP('strehl_ratio', 'Strehl比'),
            NodeOutputBP('ideal_matched', '匹配理想光斑')
        ]
        
        def update_event(self, inp=-1):
            pupil_img = self.input(0)
            focus_img = self.input(1)
            pixel_size_pupil_um = float(self.input(2))
            pixel_size_focus_um = float(self.input(3))
            focal_length_mm = float(self.input(4))
            wavelength_um = float(self.input(5))
            
            if pupil_img is not None and focus_img is not None:
                try:
                    strehl, ideal_matched = calculate_strehl_ratio_with_energy_conservation(
                        pupil_img,
                        focus_img,
                        pixel_size_pupil_um=pixel_size_pupil_um,
                        pixel_size_focus_um=pixel_size_focus_um,
                        f_mm=focal_length_mm,
                        wavelength_um=wavelength_um
                    )
                    self.set_output_val(0, strehl)
                    self.set_output_val(1, ideal_matched)
                except Exception as e:
                    print(f"Strehl比计算失败: {e}")
                    self.set_output_val(0, 0.0)
                    self.set_output_val(1, None)
            else:
                self.set_output_val(0, 0.0)
                self.set_output_val(1, None)

    # === BPP和M²计算节点 ===
    class BPPM2Node(Node):
        """BPP和M²计算节点"""
        title = 'BPP和M²计算'
        color = '#DDA0DD'
        
        init_inputs = [
            NodeInputBP('pupil_diameter_mm', '光瞳直径(mm)', dtype='float'),
            NodeInputBP('focal_diameter_mm', '焦面直径(mm)', dtype='float'),
            NodeInputBP('focal_length_mm', '焦距(mm)', dtype='float', default=100.0),
            NodeInputBP('wavelength_nm', '波长(nm)', dtype='float', default=1064.0)
        ]
        
        init_outputs = [
            NodeOutputBP('bpp_mm_mrad', 'BPP (mm·mrad)'),
            NodeOutputBP('m2', 'M²'),
            NodeOutputBP('theta_mrad', '发散角(mrad)'),
            NodeOutputBP('all_results', '所有结果')
        ]
        
        def update_event(self, inp=-1):
            pupil_diameter = self.input(0)
            focal_diameter = self.input(1)
            focal_length = float(self.input(2))
            wavelength = float(self.input(3))
            
            if pupil_diameter is not None and focal_diameter is not None:
                try:
                    results = calculate_bpp_from_pupil_and_focal(
                        pupil_diameter_mm=pupil_diameter,
                        focal_diameter_mm=focal_diameter,
                        focal_length_mm=focal_length,
                        wavelength_nm=wavelength
                    )
                    self.set_output_val(0, results['BPP_mm_mrad'])
                    self.set_output_val(1, results['M2'])
                    self.set_output_val(2, results['theta_mrad'])
                    self.set_output_val(3, results)
                except Exception as e:
                    print(f"BPP/M²计算失败: {e}")
                    self.set_output_val(0, 0.0)
                    self.set_output_val(1, 0.0)
                    self.set_output_val(2, 0.0)
                    self.set_output_val(3, {})
            else:
                self.set_output_val(0, 0.0)
                self.set_output_val(1, 0.0)
                self.set_output_val(2, 0.0)
                self.set_output_val(3, {})

    # === 时间序列分析节点 ===
    class TimeSeriesAnalysisNode(Node):
        """时间序列分析节点"""
        title = '时间序列分析'
        color = '#98D8C8'
        
        init_inputs = [
            NodeInputBP('data', '时间序列数据'),
            NodeInputBP('analysis_type', '分析类型', default='acf_pacf')
        ]
        
        init_outputs = [
            NodeOutputBP('result', '分析结果'),
            NodeOutputBP('plot', '图表')
        ]
        
        def update_event(self, inp=-1):
            data = self.input(0)
            analysis_type = self.input(1)
            
            if data is not None:
                try:
                    if analysis_type == 'acf_pacf':
                        result = self.acf_pacf_analysis(data)
                    elif analysis_type == 'stationarity':
                        result = self.check_stationarity(data)
                    elif analysis_type == 'decompose':
                        result = self.decompose_series(data)
                    else:
                        result = data
                    
                    self.set_output_val(0, result)
                    self.set_output_val(1, self.create_plot(data, result, analysis_type))
                except Exception as e:
                    print(f"时间序列分析失败: {e}")
                    self.set_output_val(0, None)
                    self.set_output_val(1, None)
            else:
                self.set_output_val(0, None)
                self.set_output_val(1, None)
        
        def acf_pacf_analysis(self, series, lags=40):
            """ACF和PACF分析"""
            try:
                acf_vals = acf(series, nlags=lags)
                pacf_vals = pacf(series, nlags=lags)
                return {'acf': acf_vals, 'pacf': pacf_vals}
            except:
                return {'acf': [], 'pacf': []}
        
        def check_stationarity(self, series, cutoff_p=0.05):
            """检验平稳性"""
            try:
                result = adfuller(series)
                is_stationary = result[1] <= cutoff_p
                return {'is_stationary': is_stationary, 'p_value': result[1], 'adf_statistic': result[0]}
            except:
                return {'is_stationary': False, 'p_value': 1.0, 'adf_statistic': 0.0}
        
        def decompose_series(self, series):
            """时间序列分解"""
            try:
                decomposition = seasonal_decompose(series, model='additive', period=10)
                return {
                    'trend': decomposition.trend,
                    'seasonal': decomposition.seasonal,
                    'residual': decomposition.resid
                }
            except:
                return {'trend': None, 'seasonal': None, 'residual': None}
        
        def create_plot(self, data, result, analysis_type):
            """创建分析图表"""
            try:
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if analysis_type == 'acf_pacf':
                    if isinstance(result, dict) and 'acf' in result and 'pacf' in result:
                        x = range(len(result['acf']))
                        ax.plot(x, result['acf'], label='ACF', marker='o')
                        ax.plot(x, result['pacf'], label='PACF', marker='s')
                        ax.set_title('ACF and PACF')
                        ax.legend()
                    else:
                        ax.plot(data, label='Time Series')
                        ax.set_title('Time Series')
                        ax.legend()
                elif analysis_type == 'stationarity':
                    ax.plot(data, label='Time Series')
                    ax.set_title(f'Stationarity Test: {"Stationary" if result.get("is_stationary", False) else "Non-stationary"}')
                    ax.legend()
                else:
                    ax.plot(data, label='Time Series')
                    ax.set_title('Time Series')
                    ax.legend()
                
                ax.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                
                return fig
            except:
                return None

    # === 波前梯度分析节点 ===
    class WavefrontGradientNode(Node):
        """波前梯度分析节点"""
        title = '波前梯度分析'
        color = '#F78FB3'
        
        init_inputs = [
            NodeInputBP('focus_image', '焦点图像'),
            NodeInputBP('defocus_image', '离焦图像'),
            NodeInputBP('delta_z', '离焦距离', dtype='float', default=3.0),
            NodeInputBP('wavelength', '波长', dtype='float', default=1064e-9)
        ]
        
        init_outputs = [
            NodeOutputBP('gradient_map', '梯度图'),
            NodeInputBP('wavefront_reconstruction', '波前重建')
        ]
        
        def update_event(self, inp=-1):
            focus_img = self.input(0)
            defocus_img = self.input(1)
            delta_z = float(self.input(2))
            wavelength = float(self.input(3))
            
            if focus_img is not None and defocus_img is not None:
                try:
                    gradient_map = compute_wavefront_gradient_tie(
                        focus_img,
                        defocus_img,
                        delta_z,
                        wavelength
                    )
                    self.set_output_val(0, gradient_map)
                    self.set_output_val(1, None)  # 波前重建需要额外的泊松方程求解
                except Exception as e:
                    print(f"波前梯度分析失败: {e}")
                    self.set_output_val(0, None)
                    self.set_output_val(1, None)
            else:
                self.set_output_val(0, None)
                self.set_output_val(1, None)

    # === 数据合并节点 ===
    class DataMergeNode(Node):
        """数据合并节点"""
        title = '数据合并'
        color = '#F1C40F'
        
        init_inputs = [
            NodeInputBP('data1', '数据集1'),
            NodeInputBP('data2', '数据集2'),
            NodeInputBP('merge_on', '合并键', default='time'),
            NodeInputBP('merge_type', '合并类型', default='left')
        ]
        
        init_outputs = [
            NodeOutputBP('merged_data', '合并后数据')
        ]
        
        def update_event(self, inp=-1):
            data1 = self.input(0)
            data2 = self.input(1)
            merge_on = self.input(2)
            merge_type = self.input(3)
            
            if data1 is not None and data2 is not None:
                try:
                    if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
                        if merge_type == 'asof':
                            # 使用时间序列近似合并
                            merged = pd.merge_asof(
                                data1.sort_values(merge_on),
                                data2.sort_values(merge_on),
                                on=merge_on,
                                direction='nearest'
                            )
                        else:
                            # 标准合并
                            merged = pd.merge(data1, data2, on=merge_on, how=merge_type)
                        
                        self.set_output_val(0, merged)
                    else:
                        self.set_output_val(0, None)
                except Exception as e:
                    print(f"数据合并失败: {e}")
                    self.set_output_val(0, None)
            else:
                self.set_output_val(0, None)

    # === 节点注册函数 ===
    @classmethod
    def get_nodes(cls):
        """获取所有节点类"""
        return [
            cls.PowerDataAnalysisNode,
            cls.ImagePreprocessingNode,
            cls.D4SigmaFeatureNode,
            cls.EllipseFitNode,
            cls.StrehlRatioNode,
            cls.BPPM2Node,
            cls.TimeSeriesAnalysisNode,
            cls.WavefrontGradientNode,
            cls.DataMergeNode
        ]

    @classmethod
    def register_nodes(cls):
        """注册所有节点到Ryven"""
        for node_class in cls.get_nodes():
            register_node(node_class)
        print(f"已注册 {len(cls.get_nodes())} 个数据处理节点")

def setup_ryven_operators():
    """设置Ryven算子环境"""
    try:
        DataAnalysisNodes.register_nodes()
        print("Ryven算子环境设置完成!")
        return True
    except Exception as e:
        print(f"设置Ryven算子环境失败: {e}")
        return False

if __name__ == "__main__":
    setup_ryven_operators()