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

import sys
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# 导入光束分析模块（analysis/ 是 ui/ 的子包）
from analysis.optical_analysis import (
    BeamAnalysisMetrics,
    HistoryManager,
    plot_3d_visualization,
    plot_beam_visualization,
    read_image_to_numpy,
)
from loguru import logger

# 配置loguru：移除默认handler，添加INFO级别handler
logger.remove()
logger.add(sys.stderr, level="DEBUG")


def main():
    st.set_page_config(page_title="AO光束质量分析", page_icon="🔬", layout="wide")

    st.title("🔬 AO光束质量分析")
    st.markdown("上传一张**光轴(axis)**图片和一张**光瞳(pupil)**图片，自动计算相关特征量并可视化结果")

    # 侧边栏 - 参数设置
    st.sidebar.header("参数设置")

    # 历史记录库设置
    st.sidebar.subheader("历史记录")
    history_base_dir = st.sidebar.text_input("历史记录目录", value=".history")
    history_manager = HistoryManager(base_dir=history_base_dir)

    # 相机参数
    st.sidebar.subheader("相机参数")
    axis_pixel_um = st.sidebar.number_input("光轴相机像素尺寸 (μm)", value=2.9, format="%.2f")
    pupil_pixel_um = st.sidebar.number_input("光瞳相机像素尺寸 (μm)", value=2.9 * 20, format="%.2f")

    # 处理参数
    st.sidebar.subheader("处理参数")
    denoise_method = st.sidebar.selectbox(
        "去暗场方法",
        ["none", "median", "min", "1_e", "manual"],
        index=0,
        format_func=lambda x: {
            "none": "无",
            "median": "中值",
            "min": "最小值",
            "1_e": "1/e",
            "manual": "手动输入",
        }[x],
    )

    # 手动输入阈值
    manual_threshold = None
    if denoise_method == "manual":
        manual_threshold = st.sidebar.number_input("手动阈值", value=100.0, step=1.0)

    # 斯特列尔比参数
    st.sidebar.subheader("光学参数")
    wavelength = st.sidebar.number_input("波长 (nm)", value=1064, step=1)
    focal_length = st.sidebar.number_input("焦距 (mm)", value=3000, step=100)
    aperture_diameter = st.sidebar.number_input("入瞳直径 (mm)", value=100, step=10)

    # 会话名称
    session_name = st.sidebar.text_input("会话名称（可选）", value="")

    # File upload
    st.header("File Upload")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Axis Image (Axis)")
        axis_file = st.file_uploader(
            "上传光轴相机图片",
            type=["tiff", "tif", "png", "jpg", "jpeg"],
            help="支持TIFF, PNG, JPG格式",
        )

    with col2:
        st.subheader("Pupil Image (Pupil)")
        pupil_file = st.file_uploader(
            "上传光瞳相机图片",
            type=["tiff", "tif", "png", "jpg", "jpeg"],
            help="支持TIFF, PNG, JPG格式",
        )

    # 处理图片
    if axis_file is not None and pupil_file is not None:
        try:
            # 读取图片
            axis_img = read_image_to_numpy(axis_file)
            pupil_img = read_image_to_numpy(pupil_file)

            # 初始化分析器
            analyzer = BeamAnalysisMetrics(
                wavelength_nm=wavelength,
                focal_length_mm=focal_length,
                aperture_diameter_mm=aperture_diameter,
            )

            # 开始新会话
            session_dir = history_manager.start_new_session(session_name)
            st.sidebar.success(f"会话已创建: {session_dir.name}")

            # 计算所有指标
            with st.spinner("正在计算光束分析指标..."):
                results = analyzer.calculate_all_metrics(
                    axis_img,
                    pupil_img,
                    axis_pixel_um,
                    pupil_pixel_um,
                    denoise_method,
                    manual_threshold,
                )

            # 保存分析结果到历史记录
            params = {
                "wavelength_nm": wavelength,
                "focal_length_mm": focal_length,
                "aperture_diameter_mm": aperture_diameter,
                "axis_pixel_um": axis_pixel_um,
                "pupil_pixel_um": pupil_pixel_um,
                "denoise_method": denoise_method,
                "manual_threshold": manual_threshold,
                "analysis_timestamp": datetime.now().isoformat(),
            }

            history_manager.save_analysis_results(results, axis_img, pupil_img, params)

            st.sidebar.success("分析结果已保存到历史记录")

            st.header("Image Preprocessing")

            # 显示原始图像
            col1, col2 = st.columns(2)
            with col1:
                st.image(axis_img, caption="光轴原始图像", width=600)
            with col2:
                st.image(pupil_img, caption="光瞳原始图像", width=600)

            # 显示D4σ结果
            st.header("D4σ Feature Extraction")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("光轴 D4σ X", f"{results['axis_D4σ_X_um']:.2f} μm")
                st.metric("光轴 D4σ Y", f"{results['axis_D4σ_Y_um']:.2f} μm")
                st.metric("光轴 平均直径", f"{results['axis_avg_diameter_um']:.2f} μm")
                st.metric("光轴 最大亮度", f"{float(np.max(axis_img)):.2f}")
                st.metric("光轴 椭圆度", f"{results['axis_ellipticity']:.4f}")
                st.metric("光轴 均匀度", f"{results['axis_uniformity']:.4f}")

            with col2:
                st.metric("光轴 短轴", f"{results['axis_short_axis']:.2f} pixel")
                st.metric("光轴 长轴", f"{results['axis_long_axis']:.2f} pixel")
                st.metric("光轴 角度", f"{results['axis_angle']:.2f} °")

            with col3:
                st.metric("光瞳 D4σ X", f"{results['pupil_D4σ_X_um']:.2f} μm")
                st.metric("光瞳 D4σ Y", f"{results['pupil_D4σ_Y_um']:.2f} μm")
                st.metric("光瞳 平均直径", f"{results['pupil_avg_diameter_um']:.2f} μm")
                st.metric("光瞳 最大亮度", f"{float(np.max(pupil_img)):.2f}")
                st.metric("光瞳 椭圆度", f"{results['pupil_ellipticity']:.4f}")
                st.metric("光瞳 均匀度", f"{results['pupil_uniformity']:.4f}")

            with col4:
                st.metric("光瞳 短轴", f"{results['pupil_short_axis']:.2f} pixel")
                st.metric("光瞳 长轴", f"{results['pupil_long_axis']:.2f} pixel")
                st.metric("光瞳 角度", f"{results['pupil_angle']:.2f} °")

            # PIB ratio
            st.header("PIB Ratio Calculation")
            st.metric("光轴 PIB占比", f"{results['axis_PIB_ratio']:.4f}")
            if results["is_overexposed_warning"]:
                st.warning("⚠️ 图像可能过曝，PIB占比计算结果可能不准确")

            # Gaussian fitting
            st.header("Gaussian Fitting")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "光轴 高斯直径 X",
                    f"{results['axis_gaussian_diameter_X_um']:.2f} μm",
                )
                st.metric(
                    "光轴 高斯直径 Y",
                    f"{results['axis_gaussian_diameter_Y_um']:.2f} μm",
                )

            # Strehl ratio
            st.header("Strehl Ratio")
            st.metric("斯特列尔比", f"{results['strehl_ratio']:.4f}")
            if results["strehl_ratio"] >= 0.8:
                st.success("Beam quality good (Strehl >= 0.8)")
            elif results["strehl_ratio"] >= 0.5:
                st.warning("Beam quality moderate (0.5 <= Strehl < 0.8)")
            else:
                st.error("Beam quality poor (Strehl < 0.5)")

            # 斯特列尔比可视化 (Plotly 3D)
            st.subheader("Strehl Ratio Visualization - 3D (Plotly)")

            # Get the images for visualization
            axis_shifted = results["axis_shifted_image"]
            pupil_shifted = results["pupil_shifted_image"]
            ideal_matched = results["ideal_matched_image"]

            # Validate the images and handle edge cases
            def validate_and_fix_image(img, name):
                """Validate image and fix common issues"""
                if img is None:
                    st.warning(f"{name} is None, using axis image as fallback")
                    return axis_shifted

                img_array = np.asarray(img, dtype=np.float64)

                if img_array.size == 0 or img_array.ndim != 2:
                    st.warning(f"{name} has invalid dimensions, using axis image as fallback")
                    return axis_shifted

                # Check if image is all zeros or nearly uniform (would appear black)
                if np.allclose(img_array, 0) or abs(img_array.max() - img_array.min()) < 1e-10:
                    st.warning(f"{name} has no contrast, generating Gaussian beam as fallback")
                    # Generate a simple Gaussian beam for visualization
                    h, w = img_array.shape
                    cy, cx = h // 2, w // 2
                    Y, X = np.ogrid[:h, :w]
                    # Create Gaussian with reasonable spread
                    sigma = min(h, w) / 8
                    gaussian_beam = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))
                    return gaussian_beam

                return img_array

            axis_validated = validate_and_fix_image(axis_shifted, "Axis Shifted Image")
            pupil_validated = validate_and_fix_image(pupil_shifted, "Pupil Shifted Image")
            ideal_validated = validate_and_fix_image(ideal_matched, "Ideal Matched Image")
            zmin, zmax = None, None
            # 计算统一尺度
            # zmin = min(
            #     axis_validated.min(),
            #     ideal_validated.min(),
            #     pupil_validated.min(),
            # )
            # zmax = max(
            #     axis_validated.max(),
            #     ideal_validated.max(),
            #     pupil_validated.max(),
            # )

            # 创建3个子图：实际光斑、理想光斑、光瞳
            col1, col2, col3 = st.columns(3)

            with col1:
                st.plotly_chart(
                    plot_3d_visualization(axis_validated, "Actual Focus", zmin, zmax),
                    use_container_width=True,
                )
            with col2:
                st.plotly_chart(
                    plot_3d_visualization(ideal_validated, "Ideal Focus", zmin, zmax),
                    use_container_width=True,
                )
            with col3:
                st.plotly_chart(
                    plot_3d_visualization(pupil_validated, "Pupil", zmin, zmax),
                    use_container_width=True,
                )

            st.sidebar.write(
                f"Pupil shifted shape: {pupil_shifted.shape}, range: [{pupil_shifted.min():.2f}, {pupil_shifted.max():.2f}]"
            )
            st.sidebar.write(
                f"Ideal matched shape: {ideal_matched.shape}, range: [{ideal_matched.min():.2f}, {ideal_matched.max():.2f}]"
            )

            # BPP calculation
            st.header("BPP (Beam Parameter Product)")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("BPP", f"{results['BPP_mm_mrad']:.4f} mm·mrad")
            with col2:
                st.metric("发散角", f"{results['divergence_mrad']:.4f} mrad")

            st.metric(
                "衍射极限 BPP",
                f"{results['diffraction_limit_BPP_mm_mrad']:.4f} mm·mrad",
            )
            st.metric("M²", f"{results['M2']:.4f}")

            # Axis visualization (2D) - Using Plotly now
            st.subheader("Axis Image Visualization - 2D")
            axis_fig = plot_beam_visualization(axis_img, "Axis", axis_pixel_um, results["axis_features"])
            st.plotly_chart(axis_fig, use_container_width=True)

            # Pupil visualization (2D) - Using Plotly now
            st.subheader("Pupil Image Visualization - 2D")
            pupil_fig = plot_beam_visualization(pupil_img, "Pupil", pupil_pixel_um, results["pupil_features"])
            st.plotly_chart(pupil_fig, use_container_width=True)

            # Results summary
            st.header("Results Summary")

            results_table = {
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
                    f"{results['axis_D4σ_X_um']:.2f}",
                    f"{results['axis_D4σ_Y_um']:.2f}",
                    f"{results['axis_avg_diameter_um']:.2f}",
                    f"{results['axis_gaussian_diameter_X_um']:.2f}",
                    f"{results['axis_gaussian_diameter_Y_um']:.2f}",
                    f"{results['axis_PIB_ratio']:.4f}",
                    f"{results['pupil_D4σ_X_um']:.2f}",
                    f"{results['pupil_D4σ_Y_um']:.2f}",
                    f"{results['pupil_avg_diameter_um']:.2f}",
                    f"{results['strehl_ratio']:.4f}",
                    f"{results['BPP_mm_mrad']:.4f}",
                    f"{results['divergence_mrad']:.4f}",
                    f"{results['M2']:.4f}",
                ],
            }

            results_df = pd.DataFrame(results_table)
            st.table(results_df)

            # 下载结果
            csv = results_df.to_csv(index=False)
            st.download_button("📥 下载结果CSV", csv, "ao_analysis_results.csv", "text/csv")

            # 历史记录管理
            st.sidebar.header("历史记录管理")
            sessions = history_manager.list_sessions()
            if sessions:
                st.sidebar.subheader("可用会话")
                for session in sessions[:10]:  # Show last 10 sessions
                    st.sidebar.text(f"- {session.name}")

                # Allow loading of a previous session
                st.sidebar.subheader("加载已有会话")
                session_options = [session.name for session in sessions]
                selected_session = st.sidebar.selectbox("选择会话", options=session_options)
                if st.sidebar.button("加载选中会话"):
                    selected_path = history_manager.base_dir / selected_session
                    session_data = history_manager.load_session_results(selected_path)
                    st.sidebar.json(session_data)
            else:
                st.sidebar.info("暂无历史记录")

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
           - **PIF占比**: 中心区域的能量占比
           - **高斯拟合**: 拟合高斯轮廓得到直径
           - **斯特列尔比**: 评价光束质量的重要指标
           - **BPP**: 光束参数乘积
           - **M²**: 光束质量因子
        
        4. **可视化**: 生成综合分析图表
        5. **导出**: 下载分析结果CSV
        6. **历史记录**: 结果自动保存到 .history 目录
        """)


if __name__ == "__main__":
    main()
