"""
AO光束质量分析Streamlit应用
===========================
上传一张光轴(axis)图片和一张光瞳(pupil)图片，自动计算相关特征量并可视化结果

功能:
- D4σ直径计算（一阶矩和二阶矩）
- PIB占比计算
- 高斯拟合直径
- FFT居中处理
- 斯特列尔比(Strehl Ratio)计算
- BPP (Beam Parameter Product) 计算
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import plotly.graph_objects as go
from loguru import logger
import sys



# 导入光束分析模块
from data_mining.optical_analysis import (
    d4sigma,
    pib_ratio,
    calculate_xy_diameters,
    calculate_bpp,
)

# 导入衍射计算模块
from data_mining.optical_analysis import (
    shift_to_center_fft,
    calculate_strehl_ratio_with_energy_conservation,
)

# 导入图像处理工具
from data_mining.optical_analysis import (
    read_image_to_numpy,
    subtract_dark_field,
)

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 配置loguru：移除默认handler，添加INFO级别handler
logger.remove()
logger.add(sys.stderr, level="DEBUG")


def plot_beam_visualization(img, title, pixel_size_um, features):
    """
    绘制光束可视化图：
    - 质心标记
    - D4σ圆
    - XY轴切面光强曲线
    
    Args:
        img: 原始图像
        title: 标题
        pixel_size_um: 像素尺寸（微米）
        features: 特征字典，包含center_x, center_y, D_x, D_y, avg_diameter
    
    Returns:
        fig: matplotlib图像
    """
    # 直接使用输入图像，不去噪
    img = np.asarray(img, dtype=np.float64)
    
    cx, cy = features['center_x'], features['center_y']
    # 使用平均sigma2作为半径
    r_pix = features['avg_diameter'] / pixel_size_um / 2  # 半径（像素）
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Centroid + D4σ circle
    ax1 = axes[0]
    im1 = ax1.imshow(img, cmap='hot', interpolation='bilinear')
    ax1.set_title(f'{title} - Centroid & D4σ', fontsize=12)
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Mark centroid
    ax1.plot(cx, cy, 'c+', markersize=15, markeredgewidth=2, label='Centroid')
    # Draw D4σ circle (using average diameter)
    circle = Circle(
        (cx, cy), 
        radius=r_pix,
        fill=False, 
        color='cyan', 
        linewidth=2,
        linestyle='--',
        label=f'D4σ (r={features["avg_diameter"]/2:.1f}μm)'
    )
    ax1.add_patch(circle)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlabel('X (pixel)')
    ax1.set_ylabel('Y (pixel)')
    
    # 2. X direction profile
    ax2 = axes[1]
    x_data = img[int(cy), :]
    x_pixels = np.arange(len(x_data))
    x_um = (x_pixels - cx) * pixel_size_um  # Convert to μm
    
    ax2.plot(x_um, x_data, 'b-', linewidth=1.5, label='X profile')
    ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.7, label='Centroid')
    max_val = np.max(x_data)
    if max_val > 0:
        ax2.axhline(y=max_val/np.e, color='r', linestyle='--', alpha=0.5, label='1/e peak')
    ax2.set_title(f'{title} - X Profile', fontsize=12)
    ax2.set_xlabel('X (μm)')
    ax2.set_ylabel('Intensity')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Y direction profile
    ax3 = axes[2]
    y_data = img[:, int(cx)]
    y_pixels = np.arange(len(y_data))
    y_um = (y_pixels - cy) * pixel_size_um  # Convert to μm
    
    ax3.plot(y_um, y_data, 'g-', linewidth=1.5, label='Y profile')
    ax3.axvline(x=0, color='gray', linestyle=':', alpha=0.7, label='Centroid')
    max_val = np.max(y_data)
    if max_val > 0:
        ax3.axhline(y=max_val/np.e, color='r', linestyle='--', alpha=0.5, label='1/e peak')
    ax3.set_title(f'{title} - Y Profile', fontsize=12)
    ax3.set_xlabel('Y (μm)')
    ax3.set_ylabel('Intensity')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_3d_visualization(img, title, zmin=None, zmax=None):
    """
    使用Plotly绘制3D表面图
    
    Args:
        img: 输入图像数组
        title: 标题
        zmin: 颜色轴最小值（用于统一尺度）
        zmax: 颜色轴最大值（用于统一尺度）
    
    Returns:
        plotly_fig: Plotly 3D图表
    """
    # 直接使用输入图像，不去噪
    img = np.asarray(img, dtype=np.float64)
    
    # 降采样以提高渲染速度
    step = max(1, min(img.shape[0], img.shape[1]) // 100)
    x = np.arange(0, img.shape[1], step)
    y = np.arange(0, img.shape[0], step)
    X, Y = np.meshgrid(x, y)
    Z = img[::step, ::step]
    
    # 统一尺度
    if zmin is None:
        zmin = Z.min()
    if zmax is None:
        zmax = Z.max()
    
    # 计算xyz范围，用于统一尺度
    x_range = X.max() - X.min()
    y_range = Y.max() - Y.min()
    z_range = zmax - zmin
    
    # 归一化Z到与XY相同的尺度范围，使xyz视觉比例一致
    if z_range > 0:
        target_range = (x_range + y_range) / 2
        Z_display = (Z - zmin) / z_range * target_range
    else:
        Z_display = Z - zmin
    
    # 计算xyz范围，用于统一尺度
    x_range = X.max() - X.min()
    y_range = Y.max() - Y.min()
    z_range = zmax - zmin
    
    # 归一化Z到与XY相同的尺度范围，使xyz视觉比例一致
    if z_range > 0:
        target_range = (x_range + y_range) / 2
        Z_display = (Z - zmin) / z_range * target_range
    else:
        Z_display = Z - zmin
    
    # 创建Plotly 3D表面图
    plotly_fig = go.Figure(data=[go.Surface(
        x=X, 
        y=Y, 
        z=Z_display,
        colorscale='Hot',
        cmin=zmin,
        cmax=zmax,
        colorbar=dict(title='Intensity'),
        hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>Intensity: %{z:.1f}<extra></extra>'
    )])
    
    plotly_fig.update_layout(
        title=f'{title} - 3D Surface',
        scene=dict(
            xaxis_title='X (pixel)',
            yaxis_title='Y (pixel)',
            zaxis_title='Intensity',
            aspectmode='data',
            aspectratio=dict(x=1, y=1, z=1),
        ),
        width=800,
        height=600,
        margin=dict(l=50, r=50, b=50, t=50)
    )
    
    return plotly_fig


def main():
    st.set_page_config(
        page_title="AO光束质量分析",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 AO光束质量分析")
    st.markdown("上传一张**光轴(axis)**图片和一张**光瞳(pupil)**图片，自动计算相关特征量并可视化结果")
    
    # 侧边栏 - 参数设置
    st.sidebar.header("参数设置")
    
    # 相机参数
    st.sidebar.subheader("相机参数")
    axis_pixel = st.sidebar.number_input(
        "光轴相机像素尺寸 (μm)", 
        value=2.9, 
        format="%.2f"
    ) * 1e-6
    pupil_pixel = st.sidebar.number_input(
        "光瞳相机像素尺寸 (μm)", 
        value=2.9*20, 
        format="%.2f"
    ) * 1e-6
    
    # 处理参数
    st.sidebar.subheader("处理参数")
    denoise_method = st.sidebar.selectbox(
        "去暗场方法", 
        ['none', 'median', 'min', '1_e', 'manual'],
        index=0,
        format_func=lambda x: {'none': '无', 'median': '中值', 'min': '最小值', '1_e': '1/e', 'manual': '手动输入'}[x]
    )
    
    # 手动输入阈值
    manual_threshold = None
    if denoise_method == 'manual':
        manual_threshold = st.sidebar.number_input("手动阈值", value=100.0, step=1.0)
    
    # 斯特列尔比参数
    st.sidebar.subheader("光学参数")
    wavelength = st.sidebar.number_input("波长 (nm)", value=1064, step=1)
    focal_length = st.sidebar.number_input("焦距 (mm)", value=3000, step=100)
    aperture_diameter = st.sidebar.number_input("入瞳直径 (mm)", value=100, step=10)
    
    # File upload
    st.header("File Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Axis Image (Axis)")
        axis_file = st.file_uploader(
            "上传光轴相机图片", 
            type=['tiff', 'tif', 'png', 'jpg', 'jpeg'],
            help="支持TIFF, PNG, JPG格式"
        )
    
    with col2:
        st.subheader("Pupil Image (Pupil)")
        pupil_file = st.file_uploader(
            "上传光瞳相机图片", 
            type=['tiff', 'tif', 'png', 'jpg', 'jpeg'],
            help="支持TIFF, PNG, JPG格式"
        )
    
    # 处理图片
    if axis_file is not None and pupil_file is not None:
        try:
            # 读取图片
            axis_img = read_image_to_numpy(axis_file)
            pupil_img = read_image_to_numpy(pupil_file)
            
            st.header("Image Preprocessing")
            
            # 去暗场处理
            axis_denoise, axis_black = subtract_dark_field(axis_img, denoise_method, manual_threshold)
            pupil_denoise, pupil_black = subtract_dark_field(pupil_img, denoise_method, manual_threshold)
            
            # 显示原始图像
            col1, col2 = st.columns(2)
            with col1:
                st.image(axis_img, caption="光轴原始图像", width=600)
            with col2:
                st.image(pupil_img, caption="光瞳原始图像", width=600)
            
            # Calculate D4σ features
            st.header("D4σ Feature Extraction")
            
            axis_features = d4sigma(axis_denoise, axis_pixel * 1e6)
            pupil_features = d4sigma(pupil_denoise, pupil_pixel * 1e6)
            
            # 显示D4σ结果
            col1, col2 = st.columns(2)
            with col1:
                st.metric("光轴 D4σ X", f"{axis_features['D_x']:.2f} μm")
                st.metric("光轴 D4σ Y", f"{axis_features['D_y']:.2f} μm")
                st.metric("光轴 平均直径", f"{axis_features['avg_diameter']:.2f} μm")
                st.metric("光轴 中心强度", f"{axis_features['center_intensity']:.2f}")
            
            with col2:
                st.metric("光瞳 D4σ X", f"{pupil_features['D_x']:.2f} μm")
                st.metric("光瞳 D4σ Y", f"{pupil_features['D_y']:.2f} μm")
                st.metric("光瞳 平均直径", f"{pupil_features['avg_diameter']:.2f} μm")
                st.metric("光瞳 中心强度", f"{pupil_features['center_intensity']:.2f}")
            
            # PIB ratio
            st.header("PIB Ratio Calculation")
            
            wavelength_m = wavelength * 1e-9
            focal_length_m = focal_length * 1e-3
            aperture_diameter_m = aperture_diameter * 1e-3
            
            axis_pib, is_overexposed = pib_ratio(
                axis_denoise,
                (axis_features['center_x'], axis_features['center_y']),
                wavelength_m=wavelength_m,
                focal_length_m=focal_length_m,
                aperture_diameter_m=aperture_diameter_m,
                pixel_size_m=axis_pixel,
            )
            st.metric("光轴 PIB占比", f"{axis_pib:.4f}")
            if is_overexposed:
                st.warning("⚠️ 图像可能过曝，PIB占比计算结果可能不准确")
            if is_overexposed:
                st.warning("⚠️ 图像可能过曝，PIB占比计算结果可能不准确")
            
            # Gaussian fitting
            st.header("Gaussian Fitting")
            
            axis_gaussian = calculate_xy_diameters(
                axis_denoise, 
                axis_features['center_x'], 
                axis_features['center_y'],
                axis_pixel * 1e6
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("光轴 高斯直径 X", f"{axis_gaussian['gaussian_dia_x(um)']:.2f} μm")
                st.metric("光轴 高斯直径 Y", f"{axis_gaussian['gaussian_dia_y(um)']:.2f} μm")
            
            # FFT centering - use original images (without denoise)
            # st.header("FFT Centering")
            
            axis_shifted = shift_to_center_fft(axis_img, axis_features['center_x'], axis_features['center_y'])
            pupil_shifted = shift_to_center_fft(pupil_img, pupil_features['center_x'], pupil_features['center_y'])
            
            # col1, col2 = st.columns(2)
            # 归一化图像以便显示
            # axis_shifted_display = normalize_image_for_display(axis_shifted)
            # pupil_shifted_display = normalize_image_for_display(pupil_shifted)
            
            # with col1:
            #     st.image(axis_shifted_display, caption="光轴 FFT居中", width=600)
            # with col2:
            #     st.image(pupil_shifted_display, caption="光瞳 FFT居中", width=600)
            
            # Strehl ratio
            st.header("Strehl Ratio")
            
            wavelength_m = wavelength * 1e-9
            focal_length_m = focal_length * 1e-3  # 转换为米
            
            strehl, ideal_matched = calculate_strehl_ratio_with_energy_conservation(
                pupil_shifted, 
                axis_shifted,
                f_m=3,
                wavelength_m=wavelength_m,
                focal_length_m=focal_length_m,
                input_pixel_size=pupil_pixel,  # 光瞳相机像素尺寸作为输入
                output_pixel_size=axis_pixel   # 光轴相机像素尺寸作为输出
            )
            
            st.metric("斯特列尔比", f"{strehl:.4f}")
            if strehl >= 0.8:
                st.success("Beam quality good (Strehl >= 0.8)")
            elif strehl >= 0.5:
                st.warning("Beam quality moderate (0.5 <= Strehl < 0.8)")
            else:
                st.error("Beam quality poor (Strehl < 0.5)")
            
            # 斯特列尔比可视化 (Plotly 3D)
            st.subheader("Strehl Ratio Visualization - 3D (Plotly)")
            
            # 计算统一尺度
            zmin = min(axis_shifted.min(), ideal_matched.min(), pupil_shifted.min())
            zmax = max(axis_shifted.max(), ideal_matched.max(), pupil_shifted.max())
            
            # 创建3个子图：实际光斑、理想光斑、光瞳
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.plotly_chart(plot_3d_visualization(axis_shifted, "Actual Focus", zmin, zmax), width='stretch')
            with col2:
                st.plotly_chart(plot_3d_visualization(ideal_matched, "Ideal Focus", zmin, zmax), width='stretch')
            with col3:
                st.plotly_chart(plot_3d_visualization(pupil_shifted, "Pupil", zmin, zmax), width='stretch')
            
            # BPP calculation
            st.header("BPP (Beam Parameter Product)")
            
            # 将直径转换为mm
            axis_diameter_mm = axis_features['avg_diameter'] * 1e-3
            pupil_diameter_mm = pupil_features['avg_diameter'] * 1e-3
            
            bpp_result = calculate_bpp(pupil_diameter_mm, axis_diameter_mm, focal_length)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("BPP", f"{bpp_result['BPP_mm_mrad']:.4f} mm·mrad")
            with col2:
                st.metric("发散角", f"{bpp_result['divergence_mrad']:.4f} mrad")
            
            # 理论衍射极限
            lambda_um = wavelength / 1000
            bpp_diffraction = lambda_um / np.pi
            st.metric("衍射极限 BPP", f"{bpp_diffraction:.4f} mm·mrad")
            
            M2 = bpp_result['BPP_mm_mrad'] / bpp_diffraction
            st.metric("M²", f"{M2:.4f}")
            
            # Axis visualization (2D)
            st.subheader("Axis Image Visualization - 2D")
            axis_fig = plot_beam_visualization(axis_img, "Axis", axis_pixel * 1e6, axis_features)
            st.pyplot(axis_fig)
            
            # Axis 3D visualization
            # st.subheader("Axis Image Visualization - 3D (Plotly)")
            # axis_3d_fig = plot_3d_visualization(axis_img, "Axis")
            # st.plotly_chart(axis_3d_fig, use_container_width=True)
            
            # Pupil visualization (2D)
            st.subheader("Pupil Image Visualization - 2D")
            pupil_fig = plot_beam_visualization(pupil_img, "Pupil", pupil_pixel * 1e6, pupil_features)
            st.pyplot(pupil_fig)
            
            # Pupil 3D visualization
            # st.subheader("Pupil Image Visualization - 3D (Plotly)")
            # pupil_3d_fig = plot_3d_visualization(pupil_img, "Pupil")
            # st.plotly_chart(pupil_3d_fig, use_container_width=True)
            
            # Results summary
            st.header("Results Summary")
            
            results = {
                "参数": [
                    "光轴 D4σ X (μm)",
                    "光轴 D4σ Y (μm)",
                    "光轴 平均直径 (μm)",
                    "光轴 高斯直径 X (μm)",
                    "光轴 高斯直径 Y (μm)",
                    "光轴 PIB占比",
                    "光瞳 D4σ X (μm)",
                    "光瞳 D4σ Y (μm)",
                    "光瞳 平均直径 (μm)",
                    "斯特列尔比",
                    "BPP (mm·mrad)",
                    "发散角 (mrad)",
                    "M²",
                ],
                "值": [
                    f"{axis_features['D_x']:.2f}",
                    f"{axis_features['D_y']:.2f}",
                    f"{axis_features['avg_diameter']:.2f}",
                    f"{axis_gaussian['gaussian_dia_x(um)']:.2f}",
                    f"{axis_gaussian['gaussian_dia_y(um)']:.2f}",
                    f"{axis_pib:.4f}",
                    f"{pupil_features['D_x']:.2f}",
                    f"{pupil_features['D_y']:.2f}",
                    f"{pupil_features['avg_diameter']:.2f}",
                    f"{strehl:.4f}",
                    f"{bpp_result['BPP_mm_mrad']:.4f}",
                    f"{bpp_result['divergence_mrad']:.4f}",
                    f"{M2:.4f}",
                ]
            }
            
            import pandas as pd
            results_df = pd.DataFrame(results)
            st.table(results_df)
            
            # 下载结果
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 下载结果CSV",
                csv,
                "ao_analysis_results.csv",
                "text/csv"
            )
            
        except Exception as e:
            st.error(f"Error processing images: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
    
    else:
        st.info("Please upload axis and pupil images to start analysis")
        
        # 显示示例
        st.header("📖 使用说明")
        st.markdown("""
        ### 分析流程
        
        1. **图片上传**: 上传光轴(axis)和光瞳(pupil)图片
        2. **参数设置**: 在侧边栏设置相机参数和处理参数
        3. **自动分析**: 系统将自动计算以下特征:
           - **D4σ直径**: 基于二阶矩的光束直径
           - **PIB占比**: 中心区域的能量占比
           - **高斯拟合**: 拟合高斯轮廓得到直径
           - **斯特列尔比**: 评价光束质量的重要指标
           - **BPP**: 光束参数乘积
           - **M²**: 光束质量因子
        
        4. **可视化**: 生成综合分析图表
        5. **导出**: 下载分析结果CSV
        """)


if __name__ == "__main__":
    main()
