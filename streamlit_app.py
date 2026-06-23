"""
AO光束质量分析Streamlit应用
===========================
上传光轴(axis)图片和/或光瞳(pupil)图片，根据上传内容自动计算相关指标并可视化结果

功能:
- D4σ直径计算（一阶矩和二阶矩）
- PIB占比计算（仅axis）
- 高斯拟合直径（仅axis）
- FFT居中处理
- 斯特列尔比(Strehl Ratio)计算（axis + pupil）
- BPP (Beam Parameter Product) 计算（axis + pupil）
- 包围圆/椭圆计算与可视化（仅pupil）
- FTL (Flat-Topped Lorentz) 拟合（仅pupil 平顶光）
- 椭圆拟合（离心率、长短轴、倾角）
- Zernike 波前像差分解（可配置阶数）
- 光瞳类型自适应包围圆/椭圆/FTL/D4σ 均匀度边界
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from loguru import logger
from matplotlib.patches import Circle

# 将项目根目录添加到 sys.path（analysis 包位于根目录）
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# 导入光束分析模块
from analysis.image.common import get_profiles  # noqa: E402
from analysis.optical_analysis import (  # noqa: E402
    calculate_bpp,
    calculate_strehl_ratio_with_energy_conservation,
    calculate_xy_diameters,
    d4sigma,
    fit_flat_topped_lorentz,
    fitting_gaussian,
    pib_ratio,
    read_image_to_numpy,
    shift_to_center_fft,
    subtract_dark_field,
    fit_zernike,
    zernike_order_label,
)
from analysis.optical_analysis.image_utils import find_spot_border, find_spot_border_energy, ellipse_fit  # noqa: E402
from analysis.optical_analysis.uniform_analysis import (  # noqa: E402
    calculate_uniformity_metrics,
    plot_uniformity_analysis,
)
from analysis.optical_analysis.visualization import (  # noqa: E402
    plot_beam_visualization,
    plot_3d_visualization,
    plot_gaussian_cross_section,
    plot_ftl_polar,
    plot_ftl_angular,
    plot_ftl_q_polar,
    plot_zernike_barchart,
)

# 配置中文字体
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# 配置loguru：移除默认handler，添加INFO级别handler
logger.remove()
logger.add(sys.stderr, level="INFO")


def _render_pupil_type_analysis(
    pupil_denoise, pupil_features, pupil_border,
    pupil_type, pupil_img, pupil_pixel, border_valid,
):
    """
    根据光瞳类型渲染分析结果。

    高斯光 → 光瞳截面高斯拟合（展示截面和拟合高斯曲线、束腰位置与值）
    平顶光 → 均匀度分析（RMS 非均匀度、峰谷非均匀度、圆内均值）

    Returns:
        tuple: (uniformity_pupil, pupil_cross_section)
        两个值均为 dict 或 None。
    """
    uniformity_pupil = None
    pupil_cross_section = None
    if not border_valid:
        return uniformity_pupil, pupil_cross_section

    if pupil_type == "高斯光 (Gaussian)":
        # === 光瞳截面高斯拟合 ===
        st.subheader("🔦 光瞳截面高斯拟合")
        st.latex(r"f(x) = A \cdot \exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right) + b")
        cx_p, cy_p = pupil_features["center_x"], pupil_features["center_y"]
        pupil_profiles = get_profiles(pupil_denoise, (cx_p, cy_p), line_width=3)
        h_prof = pupil_profiles["horizontal"]
        v_prof = pupil_profiles["vertical"]

        (h_mu, h_sigma, h_A, h_b), _ = fitting_gaussian(h_prof)
        (v_mu, v_sigma, v_A, v_b), _ = fitting_gaussian(v_prof)

        if not np.isnan(h_sigma) and not np.isnan(v_sigma):
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.metric("X 方向束腰位置", f"{h_mu:.2f} px",
                          help="水平截面高斯拟合中心 (μ)，理想情况下应接近光斑质心。")
                st.metric("X 方向束腰 σ", f"{h_sigma:.2f} px",
                          help=f"水平截面高斯拟合 1σ 宽度 (像素)。光束高斯直径 (2σ) = {2 * h_sigma:.2f} px")
            with col_c2:
                st.metric("Y 方向束腰位置", f"{v_mu:.2f} px",
                          help="垂直截面高斯拟合中心 (μ)，理想情况下应接近光斑质心。")
                st.metric("Y 方向束腰 σ", f"{v_sigma:.2f} px",
                          help=f"垂直截面高斯拟合 1σ 宽度 (像素)。光束高斯直径 (2σ) = {2 * v_sigma:.2f} px")

            # 绘制截面 + 拟合曲线（委托渲染器）
            fig_gs = plot_gaussian_cross_section(
                h_prof, v_prof,
                h_mu, h_sigma, h_A, h_b,
                v_mu, v_sigma, v_A, v_b,
            )
            st.pyplot(fig_gs)

            pupil_cross_section = {
                "h_mu": h_mu, "h_sigma": h_sigma, "h_A": h_A, "h_b": h_b,
                "v_mu": v_mu, "v_sigma": v_sigma, "v_A": v_A, "v_b": v_b,
                "success": True,
            }
        else:
            st.warning("⚠️ 高斯拟合失败，截面数据信噪比可能过低。")
            pupil_cross_section = {"success": False}
    else:
        # === 均匀度分析（平顶光） ===
        st.subheader("光瞳均匀度分析")
        uniformity = calculate_uniformity_metrics(pupil_denoise, pupil_border)
        col_u1, col_u2, col_u3, col_u4 = st.columns(4)
        with col_u1:
            st.metric("RMS 非均匀度", f"{uniformity['rms_uniformity']:.4f}",
                      help="圆内光强的 RMS 非均匀度 = std / mean。值越接近 0，光斑越均匀。")
        with col_u2:
            st.metric("能量均匀度 γ", f"{uniformity['energy_uniformity']:.4f}",
                      help="γ = 1 − RMSE/mean。越接近 1 表示能量分布越均匀（参考：>0.95 为优）。")
        with col_u3:
            st.metric("峰谷非均匀度 (P-V)", f"{uniformity['pv']:.4f}",
                      help="峰谷非均匀度 = (max - min) / mean。值越小，说明光斑强度分布越平坦。")
        with col_u4:
            st.metric("圆内均值", f"{uniformity['mean_intensity']:.2f}",
                      help="包围圆内部所有像素的强度算术平均。反映该区域的平均光强能级。")

        uni_fig = plot_uniformity_analysis(pupil_img, pupil_border, uniformity, pupil_pixel * 1e6)
        st.pyplot(uni_fig)
        uniformity_pupil = uniformity

    return uniformity_pupil, pupil_cross_section


def main():
    st.set_page_config(page_title="光束质量分析", page_icon="🔬", layout="wide")

    st.title("🔬 光束质量分析")
    st.markdown(
        "上传一张**光轴**图片和一张**光瞳**图片，自动计算相关特征量并可视化结果  [帮助](https://github.com/ywzhang909/optical_data_mining/blob/streamlit-cloud/README.md)"
    )

    # 侧边栏 - 参数设置
    st.sidebar.header("⚙️ 参数设置")

    # 相机参数
    st.sidebar.subheader("📷 相机参数")
    axis_pixel = (
        st.sidebar.number_input(
            "光轴像素尺寸 (μm)",
            value=2.9,
            format="%.2f",
            help="聚焦面相机的单像素物理尺寸，用于像素→微米换算。",
        )
        * 1e-6
    )
    pupil_pixel = (
        st.sidebar.number_input(
            "光瞳像素尺寸 (μm)",
            value=2.9 * 20,
            format="%.2f",
            help="出射光瞳面相机的单像素物理尺寸，影响 D4σ / BPP / M² 换算。",
        )
        * 1e-6
    )

    # 处理参数
    st.sidebar.subheader("🔧 处理参数")
    denoise_method = st.sidebar.selectbox(
        "去暗场方法",
        ["none", "median", "min", "1_e", "manual"],
        index=0,
        format_func=lambda x: {
            "none": "无",
            "median": "中值滤波",
            "min": "最小值",
            "1_e": "1/e 阈值",
            "manual": "手动阈值",
        }[x],
        help="从图像中扣除背景噪声。\"中值滤波\"适用于均匀背景，\"1/e 阈值\"自动估计噪声水平，\"手动阈值\"直接置零低于设定值的像素。",
    )

    # 手动输入阈值
    manual_threshold = None
    if denoise_method == "manual":
        manual_threshold = st.sidebar.number_input(
            "手动阈值",
            value=100.0,
            step=1.0,
            help="当去暗场方式为\"手动阈值\"时，低于此强度的像素直接置零。",
        )

    # 光瞳类型
    st.sidebar.subheader("🔦 光瞳类型")
    pupil_type = st.sidebar.radio(
        "选择光瞳光斑类型",
        ["平顶光 (Flat-Top)", "高斯光 (Gaussian)"],
        index=0,
        help="选择光瞳光斑的强度分布模型。平顶光使用 FTL 模型拟合特征半径 R_FL 和平顶阶数 q；"
        "高斯光使用高斯函数拟合 X/Y 截面并计算束腰。",
    )

    # 均匀度计算边界
    st.sidebar.subheader("📐 均匀度边界")
    uniformity_boundary_type = st.sidebar.radio(
        "边界类型",
        ["包围圆 (Enclosing circle)", "包围椭圆 (Ellipse)", "FTL 特征半径", "二阶矩半径 (2nd moment)", "手动输入 (Manual)"],
        index=0,
        help="平顶光均匀度分析的光斑边界选取方式。包围圆/椭圆基于轮廓检测，FTL 基于拟合特征半径，二阶矩半径基于 D4σ。仅平顶光模式生效。",
    )

    # 手动边界输入
    manual_center_x: float | None = None
    manual_center_y: float | None = None
    manual_radius: float | None = None
    if uniformity_boundary_type == "手动输入 (Manual)":
        st.sidebar.caption("手动指定均匀度边界圆心和半径（像素坐标）")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            manual_center_x = st.number_input("中心 X", value=320.0, format="%.1f")
            manual_center_y = st.number_input("中心 Y", value=240.0, format="%.1f")
        with col2:
            manual_radius = st.number_input("半径 (px)", value=200.0, min_value=1.0, format="%.1f")

    # FTL 角度采样
    st.sidebar.subheader("🔦 FTL 角度采样")
    ftl_n_angles = st.sidebar.slider(
        "FTL 角度采样数",
        min_value=0,
        max_value=72,
        value=36,
        step=4,
        help="沿光瞳圆心向各方向逐角度采样并独立拟合 FTL。0 表示仅用角向平均。增大采样数可更好地探测光斑各向异性。",
    )

    # 显示单位
    display_unit = st.sidebar.selectbox(
        "显示单位",
        ["μm", "mm", "nm"],
        index=0,
        help="长度/直径值的显示单位。内部始终以 μm 计算，显示时按需换算。",
    )
    _unit_factor = {"μm": 1.0, "mm": 0.001, "nm": 1000.0}[display_unit]
    _unit_label = display_unit

    # 斯特列尔比参数
    st.sidebar.subheader("🔬 光学参数")
    wavelength = st.sidebar.number_input(
        "波长 λ (nm)",
        value=1064,
        step=1,
        help="激光波长，用于 PIB 衍射孔径、斯特列尔比、BPP 衍射极限等计算。",
    )
    focal_length = st.sidebar.number_input(
        "焦距 f (mm)",
        value=3000,
        step=100,
        help="聚焦透镜焦距。用于 PIB 孔径角计算和 BPP 换算（θ = D_pupil / f）。",
    )
    aperture_diameter = st.sidebar.number_input(
        "入瞳直径 D (mm)",
        value=100,
        step=10,
        help="系统入瞳口径，用于 PIB 的归一化衍射孔径。",
    )

    # Zernike 分解参数
    st.sidebar.subheader("🔭 Zernike 波前分析")
    max_zernike_order = st.sidebar.slider(
        "Zernike 最大阶数",
        min_value=4,
        max_value=10,
        value=6,
        step=1,
        help="Zernike 多项式最大径向阶数 n。阶数越高可拟合的像差模式越多（如彗差、球差等高阶项），但需要更大的有效采样孔径。",
    )

    st.header("📁 文件上传")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("光轴图片 (Axis)")
        axis_file = st.file_uploader(
            "点击或拖拽上传光轴相机图片",
            type=["tiff", "tif", "png", "jpg", "jpeg"],
            help="支持 TIFF / PNG / JPG 格式。光轴为聚焦面图像（像方焦平面），用于计算 D4σ、PIB、高斯拟合直径。",
        )

    with col2:
        st.subheader("光瞳图片 (Pupil)")
        pupil_file = st.file_uploader(
            "点击或拖拽上传光瞳相机图片",
            type=["tiff", "tif", "png", "jpg", "jpeg"],
            help="支持 TIFF / PNG / JPG 格式。光瞳为出射光瞳面图像（远场），用于计算 D4σ、包围圆、FTL/高斯拟合、Zernike 波前分析。",
        )

    has_axis = axis_file is not None
    has_pupil = pupil_file is not None

    if not has_axis and not has_pupil:
        st.info("请上传光轴(axis)和/或光瞳(pupil)图片以开始分析")

        st.header("📖 使用说明")
        st.markdown("""
        | 步骤 | 操作 |
        |------|------|
        | ① | 上传光轴(Axis)和/或光瞳(Pupil)图片（可单独上传） |
        | ② | 在侧边栏调参（相机参数、去暗场、光瞳类型等） |
        | ③ | 点击 **🚀 开始计算**，系统自动计算指标并生成图表 |
        | ④ | 导出结果汇总 CSV |

        **支持的计算范围：**
        - **仅光瞳** → D4σ、包围圆/椭圆拟合、FTL/高斯拟合、均匀度分析
        - **仅光轴** → D4σ、PIB占比、高斯拟合直径
        - **两者均有** → 完整分析：BPP、M²、斯特列尔比、Zernike 波前分解
        """)
    else:
        upload_parts = []
        if has_axis:
            upload_parts.append("光轴")
        if has_pupil:
            upload_parts.append("光瞳")
        st.success(f"已上传: {', '.join(upload_parts)}")

        notes = []
        if has_axis and not has_pupil:
            notes.append("- 当前仅可计算：**光轴** D4σ、PIB占比、高斯拟合")
            notes.append("- 如需斯特列尔比、BPP、M²，请同时上传**光瞳图片**")
        elif has_pupil and not has_axis:
            notes.append("- 当前仅可计算：**光瞳** D4σ、包围圆/椭圆拟合、FTL/高斯拟合、均匀度分析")
            notes.append("- 如需 BPP、M²、斯特列尔比，请同时上传**光轴图片**")
        if notes:
            st.info("📌 " + "；".join(notes))

        if st.button("🚀 开始计算", type="primary", width="stretch"):
            zernike_result = None
            if has_axis:
                axis_img = read_image_to_numpy(axis_file)
                axis_denoise, axis_black = subtract_dark_field(
                    axis_img, denoise_method, manual_threshold
                )
                axis_features = d4sigma(axis_denoise, axis_pixel * 1e6)

            if has_pupil:
                pupil_img = read_image_to_numpy(pupil_file)
                pupil_denoise, pupil_black = subtract_dark_field(
                    pupil_img, denoise_method, manual_threshold
                )

            st.header("💡 三维光强分布 — 3D Intensity Surface")
            imgs_to_plot = []
            if has_axis:
                imgs_to_plot.append((axis_denoise, "光轴 (Axis)"))
            if has_pupil:
                imgs_to_plot.append((pupil_denoise, "光瞳 (Pupil)"))

            if imgs_to_plot:
                if len(imgs_to_plot) == 1:
                    fig3d = plot_3d_visualization(imgs_to_plot[0][0], imgs_to_plot[0][1])
                    st.plotly_chart(fig3d, width='stretch')
                else:
                    cols3d = st.columns(len(imgs_to_plot))
                    for col, (img_data, title) in zip(cols3d, imgs_to_plot):
                        with col:
                            fig3d = plot_3d_visualization(img_data, title)
                            st.plotly_chart(fig3d, width='stretch')

            if has_pupil:
                pupil_features = d4sigma(pupil_denoise, pupil_pixel * 1e6)
                pupil_border = find_spot_border(pupil_denoise)
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.metric(
                        "D4σ X 直径",
                        f"{pupil_features['D_x'] * _unit_factor:.2f} {_unit_label}",
                        help="基于 X 方向二阶矩的 D4σ 直径（ISO 11146 约定）。",
                    )
                    st.metric(
                        "D4σ Y 直径",
                        f"{pupil_features['D_y'] * _unit_factor:.2f} {_unit_label}",
                        help="基于 Y 方向二阶矩的 D4σ 直径。",
                    )
                    st.metric(
                        "D4σ 平均直径",
                        f"{pupil_features['avg_diameter'] * _unit_factor:.2f} {_unit_label}",
                        help="D4σ X 与 D4σ Y 的几何平均，近似圆对称光斑口径。",
                    )
                with col_p2:
                    st.metric(
                        "中心强度",
                        f"{pupil_features['center_intensity']:.2f}",
                        help="质心处像素强度，评估信号强度和饱和/欠曝情况。",
                    )
                    st.metric(
                        "包围圆半径",
                        f"{pupil_border['border_radius']:.2f} px",
                        help="minEnclosingCircle 拟合出的包围光斑最小圆半径。",
                    )
                    st.metric(
                        "包围圆直径",
                        f"{pupil_border['border_radius'] * 2:.2f} px",
                        help="包围圆直径 = 2×半径。",
                    )
                    st.metric(
                        "离心率",
                        f"{pupil_border['eccentricity']:.4f}",
                        help="椭圆离心率（0=正圆，越接近1越扁）。",
                    )
                    st.metric(
                        "置信度",
                        f"{pupil_border['confidence']:.4f}",
                        help="轮廓与圆拟合置信度（0-1），越高越接近规则圆形。",
                    )

                # ----- 能量指标：峰值光强、总能量、能量集中度 -----
                _peak_intensity = float(np.max(pupil_denoise))
                _total_energy = float(np.sum(pupil_denoise))
                # 围困能量曲线：按径向距离排序计算累计能量占比
                _cx = pupil_features["center_x"]
                _cy = pupil_features["center_y"]
                _h_p, _w_p = pupil_denoise.shape
                _y_i, _x_i = np.indices((_h_p, _w_p))
                _R = np.sqrt((_x_i - _cx) ** 2 + (_y_i - _cy) ** 2)
                _sort_idx = np.argsort(_R.ravel())
                _sorted_R = _R.ravel()[_sort_idx]
                _sorted_E = pupil_denoise.ravel()[_sort_idx]
                _cum_E = np.cumsum(_sorted_E)
                _cum_norm = _cum_E / _cum_E[-1] if _cum_E[-1] > 0 else _cum_E
                # 提取关键能量占比对应半径
                _r50 = _sorted_R[np.searchsorted(_cum_norm, 0.50)] if np.searchsorted(_cum_norm, 0.50) < len(_sorted_R) else np.nan
                _r80 = _sorted_R[np.searchsorted(_cum_norm, 0.80)] if np.searchsorted(_cum_norm, 0.80) < len(_sorted_R) else np.nan
                _r95 = _sorted_R[np.searchsorted(_cum_norm, 0.95)] if np.searchsorted(_cum_norm, 0.95) < len(_sorted_R) else np.nan

                col_em1, col_em2 = st.columns([1, 1])
                with col_em1:
                    st.metric("峰值光强", f"{_peak_intensity:.2f}",
                              help="光斑内像素强度最大值，反映信号峰值水平。")
                with col_em2:
                    st.metric("总能量", f"{_total_energy:.2e}",
                              help="光斑内像素强度总和，反映光束总功率的相对值。")
                # 能量集中度饼图：按径向分区展示能量占比
                _ring_labels = []
                _ring_sizes = []
                _ring_colors = []
                if not np.isnan(_r50) and not np.isnan(_r80) and not np.isnan(_r95):
                    # 查找径向环对应累计能量端点
                    _idx_r50 = np.searchsorted(_cum_norm, 0.50)
                    _idx_r80 = np.searchsorted(_cum_norm, 0.80)
                    _idx_r95 = np.searchsorted(_cum_norm, 0.95)
                    # 环内能量 = 端点累计差
                    _e50 = _cum_norm[_idx_r50] if _idx_r50 < len(_cum_norm) else 0.50  # ≈ 0.50
                    _e80 = _cum_norm[_idx_r80] - _cum_norm[_idx_r50] if _idx_r80 < len(_cum_norm) else 0.30
                    _e95 = _cum_norm[_idx_r95] - _cum_norm[_idx_r80] if _idx_r95 < len(_cum_norm) else 0.15
                    _erem = 1.0 - _e50 - _e80 - _e95 if _e50 + _e80 + _e95 < 1.0 else 0.0
                    _ring_labels = [
                        f"核心\n(r≤{_r50:.0f}px)\n{_e50*100:.0f}%",
                        f"内环\n({_r50:.0f}<r≤{_r80:.0f}px)\n{_e80*100:.0f}%",
                        f"外环\n({_r80:.0f}<r≤{_r95:.0f}px)\n{_e95*100:.0f}%",
                        f"边缘\n(r>{_r95:.0f}px)\n{_erem*100:.0f}%",
                    ]
                    _ring_sizes = [_e50, _e80, _e95, _erem]
                    _ring_colors = ["#e74c3c", "#f39c12", "#3498db", "#95a5a6"]
                else:
                    _ring_labels = ["N/A"]
                    _ring_sizes = [1.0]
                    _ring_colors = ["#bdc3c7"]
                _fig_pie, _ax_pie = plt.subplots(figsize=(4.5, 3.5))
                _ax_pie.pie(
                    _ring_sizes, labels=_ring_labels, colors=_ring_colors,
                    startangle=90, textprops={"fontsize": 8},
                    wedgeprops={"linewidth": 1, "edgecolor": "white"},
                )
                _ax_pie.set_title("能量集中度 η(r) — 径向能量环分布", fontsize=10)
                st.pyplot(_fig_pie)

                # ===== 二、形状分析：椭圆拟合 =====
                st.subheader("二、形状分析 — 椭圆拟合")
                pupil_uint8 = (pupil_denoise - pupil_denoise.min()) / (pupil_denoise.max() - pupil_denoise.min()) * 255
                pupil_uint8 = pupil_uint8.astype(np.uint8)
                pupil_ellipse = ellipse_fit(pupil_uint8)
                if not np.isnan(pupil_ellipse.get("ellipticity", np.nan)):
                    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                    with col_e1:
                        st.metric("短轴", f"{pupil_ellipse['short_axis']:.2f} px",
                                  help="椭圆拟合的短轴长度（像素），反映光斑的最短径向尺度。")
                    with col_e2:
                        st.metric("长轴", f"{pupil_ellipse['long_axis']:.2f} px",
                                  help="椭圆拟合的长轴长度（像素），反映光斑的最长径向尺度。")
                    with col_e3:
                        st.metric("椭圆度", f"{pupil_ellipse['ellipticity']:.4f}",
                                  help="长轴/短轴比值，越接近 1 越圆。")
                    with col_e4:
                        st.metric("倾角", f"{pupil_ellipse['angle']:.1f}°",
                                  help="椭圆主轴相对水平方向的旋转角度。")
                    st.metric("椭圆内均匀度", f"{pupil_ellipse['uniformity']:.4f}",
                              help="椭圆轮廓内 std/mean，越小越均匀。")

                    # ----- 边界类型对比图（光瞳 + 所有边界圆） -----
                    _pixel_size_um = pupil_pixel * 1e6
                    _r2m = pupil_features["avg_diameter"] / _pixel_size_um / 2.0
                    _shape_boundaries: list[dict] = [
                        {
                            "label": "包围圆",
                            "cx": pupil_border["border_x"],
                            "cy": pupil_border["border_y"],
                            "radius": pupil_border["border_radius"],
                            "color": "yellow",
                            "linestyle": "-.",
                        },
                        {
                            "label": "椭圆",
                            "cx": pupil_ellipse["ellipse_center_x"],
                            "cy": pupil_ellipse["ellipse_center_y"],
                            "radius": np.sqrt(pupil_ellipse["short_axis"] * pupil_ellipse["long_axis"]) / 2.0,
                            "color": "lime",
                            "linestyle": "--",
                        },
                        {
                            "label": "二阶矩",
                            "cx": pupil_features["center_x"],
                            "cy": pupil_features["center_y"],
                            "radius": _r2m,
                            "color": "orange",
                            "linestyle": ":",
                        },
                    ]
                    if manual_center_x is not None and manual_center_y is not None and manual_radius is not None:
                        _shape_boundaries.append({
                            "label": f"手动({manual_center_x:.0f},{manual_center_y:.0f})",
                            "cx": manual_center_x,
                            "cy": manual_center_y,
                            "radius": manual_radius,
                            "color": "red",
                            "linestyle": "-",
                        })
                    _border_ok = not any(np.isnan(v) for v in (
                        pupil_border["border_x"], pupil_border["border_y"], pupil_border["border_radius"]))
                    if _border_ok:
                        _fig = plot_beam_visualization(pupil_img, "Pupil", _pixel_size_um, pupil_features)
                        for _b in _shape_boundaries:
                            _fig.axes[0].add_patch(
                                Circle((_b["cx"], _b["cy"]), radius=_b["radius"],
                                       fill=False, color=_b["color"], linewidth=1.5, linestyle=_b["linestyle"],
                                       label=f"{_b['label']} (c=({_b['cx']:.0f},{_b['cy']:.0f}), r={_b['radius']:.1f}px)"))
                        _fig.axes[0].legend(loc="upper right", fontsize=8)
                        st.pyplot(_fig)
                    else:
                        st.warning("⚠️ 包围圆检测失败，无法绘制边界对比图。")
                else:
                    st.warning("⚠️ 椭圆拟合失败，无法提取参数。")

                # ===== 三、强度分布：光瞳类型拟合 + 均匀度 =====
                st.subheader("三、强度分布 — 光瞳类型拟合与均匀度")
                pupil_ftl_result = None
                pupil_gaussian_result = None
                pupil_cross_section = None
                if pupil_type == "高斯光 (Gaussian)":
                    pupil_gaussian_result = calculate_xy_diameters(
                        pupil_denoise,
                        pupil_features["center_x"],
                        pupil_features["center_y"],
                        pupil_pixel * 1e6,
                    )
                    col_gp1, col_gp2 = st.columns(2)
                    with col_gp1:
                        st.metric(
                            "高斯半腰 X",
                            f"{pupil_gaussian_result['gaussian_dia_x(um)'] * _unit_factor:.2f} {_unit_label}",
                            help="X 方向高斯拟合半高宽直径 (2σ)。",
                        )
                    with col_gp2:
                        st.metric(
                            "高斯半腰 Y",
                            f"{pupil_gaussian_result['gaussian_dia_y(um)'] * _unit_factor:.2f} {_unit_label}",
                            help="Y 方向高斯拟合半高宽直径 (2σ)。",
                        )
                else:
                    pupil_ftl_result = fit_flat_topped_lorentz(
                        pupil_denoise,
                        pupil_features["center_x"],
                        pupil_features["center_y"],
                        pixel_size=pupil_pixel * 1e6,
                        n_angles=ftl_n_angles,
                    )
                    if pupil_ftl_result["success"]:
                        st.latex(r"I(R) = \frac{I_0}{\left[1 + (R/R_{\text{FL}})^q\right]^{1 + 2/q}}")
                        col_f1, col_f2, col_f3 = st.columns(3)
                        with col_f1:
                            st.metric(
                                "R_FL (特征半径)",
                                f"{pupil_ftl_result['R_FL'] * _unit_factor:.2f} {_unit_label}",
                                help="FTL 拟合特征半径，决定光斑整体尺度。",
                            )
                            st.metric(
                                "R_FL 误差",
                                f"±{pupil_ftl_result['R_FL_error'] * _unit_factor:.2f} {_unit_label}",
                                help="全局 FTL 拟合 R_FL 的标准误差，反映拟合不确定性。",
                            )
                        with col_f2:
                            st.metric(
                                "q (平顶阶数)",
                                f"{pupil_ftl_result['q']:.2f}",
                                help="q→∞ 接近理想平顶，q=2 为洛伦兹线型。",
                            )
                            st.metric(
                                "q 误差",
                                f"±{pupil_ftl_result['q_error']:.2f}",
                                help="全局 FTL 拟合平顶阶数 q 的标准误差。值越大说明径向轮廓对 q 越不敏感。",
                            )
                        with col_f3:
                            st.metric(
                                "拟合中心强度 I₀",
                                f"{pupil_ftl_result['I0']:.2f}",
                                help="FTL 模型在 R=0 处的拟合峰值强度，反映光斑中心能量密度。",
                            )
                            st.metric(
                                "R_FL / D4σ 半径比",
                                f"{pupil_ftl_result['R_FL'] / (pupil_features['avg_diameter'] / 2):.4f}",
                                help="特征半径与 D4σ 半半径之比，反映光斑轮廓形态。",
                            )
                    else:
                        st.warning(f"⚠️ FTL 拟合失败: {pupil_ftl_result['message']}")

                # ===== 计算所有可用边界类型（用于可视化对比） =====
                all_boundaries: list[dict] = []

                # 1. 包围圆 — 始终可用
                all_boundaries.append({
                    "label": "包围圆",
                    "cx": pupil_border["border_x"],
                    "cy": pupil_border["border_y"],
                    "radius": pupil_border["border_radius"],
                    "color": "yellow",
                    "linestyle": "-.",
                })

                # 2. 椭圆 — 始终可计算
                try:
                    energy_border = find_spot_border_energy(pupil_denoise, edge_method="ellipse")
                    all_boundaries.append({
                        "label": "椭圆",
                        "cx": energy_border["border_x"],
                        "cy": energy_border["border_y"],
                        "radius": energy_border["border_radius"],
                        "color": "lime",
                        "linestyle": "--",
                    })
                except Exception:
                    pass

                # 3. 二阶矩半径 — 始终可计算
                pixel_size_um_pupil = pupil_pixel * 1e6
                radius_2m = pupil_features["avg_diameter"] / pixel_size_um_pupil / 2.0
                all_boundaries.append({
                    "label": "二阶矩",
                    "cx": pupil_features["center_x"],
                    "cy": pupil_features["center_y"],
                    "radius": radius_2m,
                    "color": "orange",
                    "linestyle": ":",
                })

                # 4. FTL 特征半径 — 仅平顶光模式且拟合成功
                if pupil_type == "平顶光 (Flat-Top)" and pupil_ftl_result is not None and pupil_ftl_result.get("success"):
                    all_boundaries.append({
                        "label": "FTL",
                        "cx": pupil_features["center_x"],
                        "cy": pupil_features["center_y"],
                        "radius": pupil_ftl_result["R_FL_pixels"],
                        "color": "magenta",
                        "linestyle": "-",
                    })

                # 5. 手动输入 — 仅用户提供了有效值
                if manual_center_x is not None and manual_center_y is not None and manual_radius is not None:
                    all_boundaries.append({
                        "label": f"手动({manual_center_x:.0f},{manual_center_y:.0f})",
                        "cx": manual_center_x,
                        "cy": manual_center_y,
                        "radius": manual_radius,
                        "color": "red",
                        "linestyle": "-",
                    })

                # ===== 根据所选边界类型设置 uniformity_border =====
                uniformity_border = dict(pupil_border)
                if pupil_type == "平顶光 (Flat-Top)":
                    if uniformity_boundary_type == "包围椭圆 (Ellipse)":
                        eb = next((b for b in all_boundaries if b["label"] == "椭圆"), None)
                        if eb is not None:
                            uniformity_border["border_radius"] = eb["radius"]
                    elif uniformity_boundary_type == "FTL 特征半径":
                        eb = next((b for b in all_boundaries if b["label"] == "FTL"), None)
                        if eb is not None:
                            uniformity_border["border_radius"] = eb["radius"]
                        else:
                            st.warning("⚠️ FTL 拟合未成功，均匀度边界回退到包围圆半径")
                    elif uniformity_boundary_type == "二阶矩半径 (2nd moment)":
                        eb = next((b for b in all_boundaries if b["label"] == "二阶矩"), None)
                        if eb is not None:
                            uniformity_border["border_radius"] = eb["radius"]
                    elif uniformity_boundary_type == "手动输入 (Manual)":
                        eb = next((b for b in all_boundaries if b["label"].startswith("手动")), None)
                        if eb is not None:
                            uniformity_border["border_radius"] = eb["radius"]
                            uniformity_border["border_x"] = eb["cx"]
                            uniformity_border["border_y"] = eb["cy"]

                uniformity_pupil, pupil_cross_section = _render_pupil_type_analysis(
                    pupil_denoise, pupil_features, uniformity_border,
                    pupil_type, pupil_img, pupil_pixel,
                    not (
                        np.isnan(pupil_border["border_x"])
                        or np.isnan(pupil_border["border_y"])
                        or np.isnan(pupil_border["border_radius"])
                    ),
                )
                if uniformity_pupil is not None:
                    st.caption(
                        f"边界: {uniformity_boundary_type}"
                        f" | 圆心: ({uniformity_pupil['cx']:.1f}, {uniformity_pupil['cy']:.1f})"
                        f" | 半径: {uniformity_pupil['radius']:.2f} px"
                    )

                if pupil_type == "平顶光 (Flat-Top)" and pupil_ftl_result is not None:
                    if ftl_n_angles > 0 and pupil_ftl_result.get("angular_R_FL_pixels"):
                        st.subheader("FTL 角度采样 — 各向异性分析")
                        ang_rfl = pupil_ftl_result["angular_R_FL_pixels"]
                        ang_q = pupil_ftl_result.get("angular_q", [])
                        ang_theta = pupil_ftl_result.get("angular_theta_deg", [])

                        if ang_rfl and not all(np.isnan(ang_rfl)):
                            ang_rfl_arr = np.array(ang_rfl, dtype=np.float64)
                            ang_q_arr = np.array(ang_q, dtype=np.float64)
                            ang_theta_arr = np.deg2rad(np.array(ang_theta, dtype=np.float64))

                            col_ftl1, col_ftl2, col_ftl3, col_ftl4 = st.columns(4)
                            valid_rfl = ang_rfl_arr[~np.isnan(ang_rfl_arr)]
                            with col_ftl1:
                                st.metric("平均 R_FL",
                                          f"{np.nanmean(valid_rfl) * _unit_factor:.2f} {_unit_label}",
                                          help="各角度 FTL 特征半径 R_FL(θ) 的角向均值。与全局径向平均拟合值接近，反映光斑总体尺度。")
                            with col_ftl2:
                                st.metric("R_FL 标准差",
                                          f"{np.nanstd(valid_rfl) * _unit_factor:.2f} {_unit_label}",
                                          help="R_FL(θ) 角向标准差。值越大说明光斑半径的方向依赖性越强（各向异性越显著）。")
                            with col_ftl3:
                                st.metric("R_FL 最大",
                                          f"{np.nanmax(valid_rfl) * _unit_factor:.2f} {_unit_label}",
                                          help="R_FL(θ) 最大值，对应光斑的最长径向尺度方向。")
                            with col_ftl4:
                                eftl = pupil_ftl_result.get("ellipticity_from_ftl", np.nan)
                                st.metric("FTL 椭圆度", f"{eftl:.4f}" if not np.isnan(eftl) else "N/A",
                                          help="基于 R_FL(θ) 最大/最小值比 (= 长轴/短轴)，=1 表示圆对称，>1 表示椭圆。")

                            valid_q = ang_q_arr[~np.isnan(ang_q_arr)]
                            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
                            with col_q1:
                                st.metric("平均 q",
                                          f"{np.nanmean(valid_q):.2f}" if len(valid_q) > 0 else "N/A",
                                          help="各角度平顶阶数 q(θ) 的角向均值。q → ∞ 为理想平顶，q ≈ 2 为洛伦兹线型。")
                            with col_q2:
                                st.metric("q 标准差",
                                          f"{np.nanstd(valid_q):.2f}" if len(valid_q) > 0 else "N/A",
                                          help="q(θ) 角向标准差。反映光斑边缘陡峭程度的方向一致性，越大说明各方向轮廓形态差异越大。")
                            with col_q3:
                                st.metric("q 最小",
                                          f"{np.nanmin(valid_q):.2f}" if len(valid_q) > 0 else "N/A",
                                          help="q(θ) 最小值，对应光斑边缘最平缓（最接近洛伦兹型）的方向。")
                            with col_q4:
                                st.metric("q 最大",
                                          f"{np.nanmax(valid_q):.2f}" if len(valid_q) > 0 else "N/A",
                                          help="q(θ) 最大值，对应光斑边缘最陡峭（最接近理想平顶）的方向。")

                            col_polar1, col_polar2 = st.columns(2)
                            with col_polar1:
                                st.markdown(f"**R_FL(θ) 极坐标图** (n={ftl_n_angles}) :gray[R_FL(θ) 为各角度方向 FTL 拟合的特征半径，反映光斑的径向尺度在各方向上的分布。半径轴大小表示该方向光斑延伸范围，图形越接近正圆表示光斑越圆对称；半径波动越大说明各向异性越显著，椭圆拉伸方向对应半径极大值方向。]")
                                st.pyplot(
                                    plot_ftl_polar(ang_theta_arr, ang_rfl_arr * _unit_factor, _unit_label)
                                )
                            with col_polar2:
                                if len(valid_q) > 0:
                                    st.markdown("**FTL 平顶阶数 q(θ) 极坐标图** :gray[q(θ) 控制 FTL 模型边缘滚降的陡峭程度（阶数），q → ∞ 为理想平顶（阶跃边缘），q ≈ 2 为洛伦兹型（缓变边缘）。q 越大边缘越陡、顶部越平坦。各向异性越显著，q(θ) 的角向波动越大。]")
                                    st.pyplot(plot_ftl_q_polar(ang_theta_arr, ang_q_arr))

                if has_axis and has_pupil:
                    st.subheader("四、光束质量 — BPP & M²")
                    axis_diameter_mm = axis_features["avg_diameter"] * 1e-3
                    pupil_diameter_mm = pupil_features["avg_diameter"] * 1e-3
                    bpp_result = calculate_bpp(
                        pupil_diameter_mm, axis_diameter_mm, focal_length
                    )
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        st.metric("BPP", f"{bpp_result['BPP_mm_mrad']:.4f} mm·mrad",
                                  help="Beam Parameter Product = 束腰半径 × 远场发散角。衡量光束质量的核心指标，越接近衍射极限越好。")
                    with col_b2:
                        st.metric(
                            "发散角",
                            f"{bpp_result['divergence_mrad']:.4f} mrad",
                            help="远场发散半角 (θ = D_focal / 2f)，基于焦斑直径与焦距计算。",
                        )
                    lambda_um = wavelength / 1000
                    bpp_diffraction = lambda_um / np.pi
                    st.metric("衍射极限 BPP", f"{bpp_diffraction:.4f} mm·mrad",
                              help="理想衍射受限光束的 BPP (= λ/π)。作为基准与实测 BPP 对比求 M²。")
                    M2 = bpp_result["BPP_mm_mrad"] / bpp_diffraction
                    st.metric("M²", f"{M2:.4f}",
                              help="光束质量因子 = BPP / BPP_衍射极限。M² = 1 为理想衍射受限，越大光束质量越差。")

            st.header("光轴分析 — Axis Analysis")

            if has_axis:
                st.subheader("一、尺寸特征 — D4σ 直径")
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.metric(
                        "D4σ X 直径",
                        f"{axis_features['D_x'] * _unit_factor:.2f} {_unit_label}",
                        help="基于 X 方向二阶矩的 D4σ 直径（ISO 11146），反映焦斑 X 向尺寸。",
                    )
                    st.metric(
                        "D4σ Y 直径",
                        f"{axis_features['D_y'] * _unit_factor:.2f} {_unit_label}",
                        help="基于 Y 方向二阶矩的 D4σ 直径（ISO 11146），反映焦斑 Y 向尺寸。",
                    )
                with col_a2:
                    st.metric(
                        "D4σ 平均直径",
                        f"{axis_features['avg_diameter'] * _unit_factor:.2f} {_unit_label}",
                        help="D4σ X 与 D4σ Y 的几何平均，近似圆对称焦斑等效口径。",
                    )
                    st.metric(
                        "中心强度",
                        f"{axis_features['center_intensity']:.2f}",
                        help="质心处像素强度。用于判断图像是否饱和或欠曝。",
                    )

                axis_fig = plot_beam_visualization(
                    axis_img, "Axis", axis_pixel * 1e6, axis_features
                )
                st.pyplot(axis_fig)

                st.subheader("二、功率内桶比 — PIB")
                axis_pib, is_overexposed = pib_ratio(
                    axis_denoise,
                    (axis_features["center_x"], axis_features["center_y"]),
                    wavelength_m=wavelength * 1e-9,
                    focal_length_m=focal_length * 1e-3,
                    aperture_diameter_m=aperture_diameter * 1e-3,
                    pixel_size_m=axis_pixel,
                )
                st.metric("PIB 占比", f"{axis_pib:.4f}",
                          help="Power In Bucket：衍射孔径内能量与总能量之比。基于入瞳直径 D 和波长计算衍射极限孔径。值越接近 1 说明能量越集中。")
                if is_overexposed:
                    st.warning("⚠️ 图像存在过曝像素，PIB 占比可能偏高（饱和像素低估总能量）。")

                st.subheader("三、强度分布 — 光轴高斯拟合")
                st.latex(r"f(x) = A \cdot \exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right) + b")
                axis_gaussian = calculate_xy_diameters(
                    axis_denoise,
                    axis_features["center_x"],
                    axis_features["center_y"],
                    axis_pixel * 1e6,
                )
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.metric(
                        "高斯直径 X",
                        f"{axis_gaussian['gaussian_dia_x(um)'] * _unit_factor:.2f} {_unit_label}",
                        help="X 方向截面高斯拟合的半高宽直径 (2σ)，反映焦斑 X 向高斯宽度。",
                    )
                    st.metric(
                        "高斯直径 Y",
                        f"{axis_gaussian['gaussian_dia_y(um)'] * _unit_factor:.2f} {_unit_label}",
                        help="Y 方向截面高斯拟合的半高宽直径 (2σ)，反映焦斑 Y 向高斯宽度。",
                    )

            if has_axis and has_pupil:
                axis_shifted = shift_to_center_fft(
                    axis_img, axis_features["center_x"], axis_features["center_y"]
                )
                pupil_shifted = shift_to_center_fft(
                    pupil_img, pupil_features["center_x"], pupil_features["center_y"]
                )

                st.header("五、波前质量 — 斯特列尔比 (Strehl Ratio)")
                try:
                    strehl, ideal_matched = calculate_strehl_ratio_with_energy_conservation(
                        pupil_shifted,
                        axis_shifted,
                        f_m=3,
                        wavelength_m=wavelength * 1e-9,
                        focal_length_m=focal_length * 1e-3,
                        input_pixel_size=pupil_pixel,
                        output_pixel_size=axis_pixel,
                    )
                    st.metric("斯特列尔比", f"{strehl:.4f}",
                              help="Strehl Ratio = 实际焦斑峰值强度 / 理想衍射受限焦斑峰值强度（能量守恒归一化）。Strehl ≥ 0.8 为衍射受限。")
                    if strehl >= 0.8:
                        st.success("✅ 光束质量优秀 (Strehl ≥ 0.8) — 接近衍射受限")
                    elif strehl >= 0.5:
                        st.warning("⚠️ 光束质量中等 (0.5 ≤ Strehl < 0.8) — 存在明显像差")
                    else:
                        st.error("❌ 光束质量较差 (Strehl < 0.5) — 像差严重")

                    st.subheader("Strehl 三维重建 — 实际 / 理想 / 光瞳")
                    zmin = min(axis_shifted.min(), ideal_matched.min(), pupil_shifted.min())
                    zmax = max(axis_shifted.max(), ideal_matched.max(), pupil_shifted.max())
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        st.plotly_chart(
                            plot_3d_visualization(axis_shifted, "实际焦斑", zmin, zmax),
                            width="stretch",
                        )
                    with col_s2:
                        st.plotly_chart(
                            plot_3d_visualization(ideal_matched, "理想焦斑", zmin, zmax),
                            width="stretch",
                        )
                    with col_s3:
                        st.plotly_chart(
                            plot_3d_visualization(pupil_shifted, "光瞳", zmin, zmax),
                            width="stretch",
                        )
                except Exception as e:
                    logger.exception(e)
                    st.warning(f"⚠️ 斯特列尔比计算失败: {e}")

                # ===== 六、波前像差 — Zernike 分解 =====
                st.header("六、波前像差 — Zernike 多项式分解")
                try:
                    zernike_result = fit_zernike(
                        pupil_denoise,
                        max_order=max_zernike_order,
                        pupil_radius_px=pupil_border["border_radius"],
                        center_x=pupil_border.get("border_x"),
                        center_y=pupil_border.get("border_y"),
                    )
                    zernike_coeffs = zernike_result["coeffs"]
                    zernike_noll = zernike_result["noll_map"]

                    col_z1, col_z2 = st.columns([2, 1])
                    sorted_indices = np.argsort(np.abs(zernike_coeffs))[::-1]
                    top_k = min(10, len(sorted_indices))
                    top_idx = sorted_indices[:top_k]
                    with col_z1:
                        st.caption("各阶 Zernike 系数条形图（按 |系数| 降序）")
                        fig_bar = plot_zernike_barchart(zernike_coeffs, zernike_noll, top_k=10)
                        st.pyplot(fig_bar)

                    with col_z2:
                        st.caption("前 5 阶系数")
                        for rank, idx in enumerate(top_idx[:5], 1):
                            noll_n = zernike_noll[idx]
                            label = zernike_order_label(noll_n)
                            val = zernike_coeffs[idx]
                            st.metric(
                                f"#{rank} {label}",
                                f"{val:.6f}",
                            )
                        st.caption(f"RMSE 重建误差: {zernike_result['rmse']:.6f}")
                        st.caption(f"有效基函数数: {zernike_result['n_terms']}")

                    with st.expander("📋 Zernike 系数完整表"):
                        zernike_table_data = {
                            "Noll 序号": [],
                            "模式名称": [],
                            "系数值": [],
                            "绝对值": [],
                        }
                        for i, noll_n in enumerate(zernike_noll):
                            zernike_table_data["Noll 序号"].append(noll_n)
                            zernike_table_data["模式名称"].append(zernike_order_label(noll_n))
                            zernike_table_data["系数值"].append(f"{zernike_coeffs[i]:.6f}")
                            zernike_table_data["绝对值"].append(f"{abs(zernike_coeffs[i]):.6f}")
                        st.dataframe(zernike_table_data, width='stretch')

                except Exception as e:
                    st.warning(f"⚠️ Zernike 分解失败: {e}")

            st.header("📊 结果汇总 — Results Summary")
            results = {"参数": [], "值": []}

            if has_axis:
                results["参数"] += [
                    f"光轴 D4σ X ({_unit_label})",
                    f"光轴 D4σ Y ({_unit_label})",
                    f"光轴 平均直径 ({_unit_label})",
                    "PIB",
                    f"光轴 高斯直径 X ({_unit_label})",
                    f"光轴 高斯直径 Y ({_unit_label})",
                ]
                results["值"] += [
                    f"{axis_features['D_x'] * _unit_factor:.2f}",
                    f"{axis_features['D_y'] * _unit_factor:.2f}",
                    f"{axis_features['avg_diameter'] * _unit_factor:.2f}",
                    f"{axis_pib:.4f}",
                    f"{axis_gaussian['gaussian_dia_x(um)'] * _unit_factor:.2f}",
                    f"{axis_gaussian['gaussian_dia_y(um)'] * _unit_factor:.2f}",
                ]

            if has_pupil:
                results["参数"] += [
                    f"光瞳 D4σ X ({_unit_label})",
                    f"光瞳 D4σ Y ({_unit_label})",
                    f"光瞳 D4σ 平均 ({_unit_label})",
                    "包围圆半径 (px)",
                    "包围圆直径 (px)",
                    "离心率",
                    "置信度",
                ]
                results["值"] += [
                    f"{pupil_features['D_x'] * _unit_factor:.2f}",
                    f"{pupil_features['D_y'] * _unit_factor:.2f}",
                    f"{pupil_features['avg_diameter'] * _unit_factor:.2f}",
                    f"{pupil_border['border_radius']:.2f}",
                    f"{pupil_border['border_radius'] * 2:.2f}",
                    f"{pupil_border['eccentricity']:.4f}",
                    f"{pupil_border['confidence']:.4f}",
                ]
                if not np.isnan(pupil_ellipse.get("ellipticity", np.nan)):
                    results["参数"] += ["短轴 (px)", "长轴 (px)", "椭圆度", "倾角 (°)", "椭圆内均匀度"]
                    results["值"] += [
                        f"{pupil_ellipse['short_axis']:.2f}",
                        f"{pupil_ellipse['long_axis']:.2f}",
                        f"{pupil_ellipse['ellipticity']:.4f}",
                        f"{pupil_ellipse['angle']:.1f}",
                        f"{pupil_ellipse['uniformity']:.4f}",
                    ]
                if uniformity_pupil is not None:
                    results["参数"] += [
                        "均匀度边界类型",
                        "均匀度边界圆心 X (px)",
                        "均匀度边界圆心 Y (px)",
                        "均匀度边界半径 (px)",
                        "RMS 非均匀度",
                        "能量均匀度 γ",
                        "峰谷非均匀度",
                        "圆内均值",
                    ]
                    results["值"] += [
                        uniformity_boundary_type,
                        f"{uniformity_pupil['cx']:.1f}",
                        f"{uniformity_pupil['cy']:.1f}",
                        f"{uniformity_pupil['radius']:.2f}",
                        f"{uniformity_pupil['rms_uniformity']:.4f}",
                        f"{uniformity_pupil['energy_uniformity']:.4f}",
                        f"{uniformity_pupil['pv']:.4f}",
                        f"{uniformity_pupil['mean_intensity']:.2f}",
                    ]
                elif pupil_cross_section is not None and pupil_cross_section["success"]:
                    results["参数"] += [
                        "X 束腰位置 (px)", "X 束腰 σ (px)",
                        "Y 束腰位置 (px)", "Y 束腰 σ (px)",
                    ]
                    results["值"] += [
                        f"{pupil_cross_section['h_mu']:.2f}",
                        f"{pupil_cross_section['h_sigma']:.2f}",
                        f"{pupil_cross_section['v_mu']:.2f}",
                        f"{pupil_cross_section['v_sigma']:.2f}",
                    ]
                if pupil_type == "高斯光 (Gaussian)" and pupil_gaussian_result is not None:
                    results["参数"] += [
                        f"光瞳 高斯半腰 X ({_unit_label})",
                        f"光瞳 高斯半腰 Y ({_unit_label})",
                    ]
                    results["值"] += [
                        f"{pupil_gaussian_result['gaussian_dia_x(um)'] * _unit_factor:.2f}",
                        f"{pupil_gaussian_result['gaussian_dia_y(um)'] * _unit_factor:.2f}",
                    ]
                elif pupil_type == "平顶光 (Flat-Top)" and pupil_ftl_result is not None and pupil_ftl_result["success"]:
                    results["参数"] += [
                        f"R_FL ({_unit_label})",
                        f"R_FL 误差 ({_unit_label})",
                        "q (平顶阶数)",
                        "q 误差",
                        "拟合中心强度 I₀",
                    ]
                    results["值"] += [
                        f"{pupil_ftl_result['R_FL'] * _unit_factor:.2f}",
                        f"{pupil_ftl_result['R_FL_error'] * _unit_factor:.2f}",
                        f"{pupil_ftl_result['q']:.2f}",
                        f"{pupil_ftl_result['q_error']:.2f}",
                        f"{pupil_ftl_result['I0']:.2f}",
                    ]
                    if ftl_n_angles > 0:
                        ang_q_arr = np.array(pupil_ftl_result.get("angular_q", []), dtype=np.float64)
                        valid_q = ang_q_arr[~np.isnan(ang_q_arr)]
                        if len(valid_q) > 0:
                            results["参数"] += [
                                "q 角向均值", "q 角向标准差", "q 角向最小", "q 角向最大",
                            ]
                            results["值"] += [
                                f"{np.nanmean(valid_q):.2f}",
                                f"{np.nanstd(valid_q):.2f}",
                                f"{np.nanmin(valid_q):.2f}",
                                f"{np.nanmax(valid_q):.2f}",
                            ]

            if has_axis and has_pupil:
                results["参数"] += [
                    "BPP (mm·mrad)",
                    "发散角 (mrad)",
                    "M²",
                    "斯特列尔比",
                ]
                results["值"] += [
                    f"{bpp_result['BPP_mm_mrad']:.4f}",
                    f"{bpp_result['divergence_mrad']:.4f}",
                    f"{M2:.4f}",
                    f"{strehl:.4f}",
                ]
                if zernike_result is not None:
                    results["参数"] += ["Zernike RMSE", "Zernike 基函数数"]
                    results["值"] += [
                        f"{zernike_result['rmse']:.6f}",
                        f"{zernike_result['n_terms']}",
                    ]

            import pandas as pd

            results_df = pd.DataFrame(results)
            st.table(results_df)

            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 下载结果CSV", csv, "ao_analysis_results.csv", "text/csv"
            )


if __name__ == "__main__":
    main()
