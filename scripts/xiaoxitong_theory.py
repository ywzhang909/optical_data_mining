"""小系统激光打靶试验的理论计算: 大气湍流、热晕、大气消光、靶材熔穿能量学.

所有公式与常数的来源见 `data/小系统/试验报告/参考文献表.md` 中对应编号。
输出:
    data/小系统/试验报告/理论计算.md
    data/小系统/试验报告/理论计算_数值.csv
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from xiaoxitong_config import env_float

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "小系统"
REPORT_DIR = DATA_DIR / "试验报告"
TABLE_CSV = REPORT_DIR / "试验参数总表.csv"

# ---------------------------------------------------------------- 物理常数
# 激光参数从 .env 读取：数值仅用于内部计算，不写入产物
LAMBDA = env_float("XXT_LAMBDA_M")  # 波长 [m]（试验未记录，按近红外常用波段假设）
K_WAVE = 2.0 * math.pi / LAMBDA  # 波数 [rad/m]
P_STD = env_float("XXT_POWER_STD_W")  # 标准功率 [W]
P_LOW = env_float("XXT_POWER_LOW_W")  # 低功率 [W]

# 反演光斑几何缺省值（仅当数据不足以拟合时使用）
FALLBACK_SPOT_D0_MM = env_float("XXT_SPOT_D0_MM")
FALLBACK_SPOT_THETA_URAD = env_float("XXT_SPOT_THETA_URAD")
FALLBACK_SPREAD_D0_MM = env_float("XXT_SPREAD_D0_MM")
FALLBACK_SPREAD_THETA_URAD = env_float("XXT_SPREAD_THETA_URAD")

# 不锈钢 304 热物性
RHO_STEEL = 7900.0  # 密度 [kg/m3]
CP_STEEL = 500.0  # 比热 [J/(kg.K)]
T_MELT = 1700.0  # 熔点 [K]
T_AMB = 293.0  # 环境温度 [K]
L_FUSION = 2.7e5  # 熔化潜热 [J/kg]

# 吸收率取值（抛光/未处理/氧化粗糙），来源见文献 [42][49]
ABSORPTANCE = {"抛光": 0.35, "未记录": 0.40, "氧化/粗糙": 0.60}

# 空气热物性
RHO_CP_AIR = 1200.0  # 体积热容 [J/(m3.K)]
DN_DT_AIR = -1.0e-6  # 折射率温度系数 [1/K]（负值 -> 热散焦）

# 该波段消光系数情景 [1/m]
EXTINCTION = {"洁净": 1.0e-5, "霾/轻雾": 5.0e-5, "浓霾": 2.0e-4}

# 湍流折射率结构常数情景 [m^-2/3]
CN2 = {"夜间/弱": 1.0e-15, "近地面白天": 1.0e-14, "强对流": 1.0e-13}

# 传导损失折算的有效效率（熔穿模型中的经验因子）
ETA = 0.6


# ---------------------------------------------------------------- 湍流
def fried_r0(cn2: float, path: float) -> float:
    """平面波 Fried 相干长度 r0 = (0.423 k^2 Cn2 L)^(-3/5) [m]（文献[4][3]）."""
    return (0.423 * K_WAVE**2 * cn2 * path) ** (-3.0 / 5.0)


def rytov_sigma_i2(cn2: float, path: float) -> float:
    """弱起伏区平面波闪烁指数 sigma_I^2 = 1.23 Cn2 k^(7/6) L^(11/6)（文献[10][3]）."""
    return 1.23 * cn2 * K_WAVE ** (7.0 / 6.0) * path ** (11.0 / 6.0)


def angle_of_arrival_sigma(cn2: float, path: float, aperture: float = 0.1) -> float:
    """到达角起伏标准差 sigma_theta = (2.91 Cn2 L D^(-1/3))^0.5 [rad]（文献[11]）."""
    return math.sqrt(2.91 * cn2 * path * aperture ** (-1.0 / 3.0))


def beam_wander_sigma(cn2: float, path: float, w0: float = 0.01) -> float:
    """准直光束漂移标准差 sigma_b = (2.42 Cn2 L^3 W0^(-1/3))^0.5 [m]（文献[11][3]）."""
    return math.sqrt(2.42 * cn2 * path**3 * w0 ** (-1.0 / 3.0))


def turbulence_table() -> pd.DataFrame:
    rows = []
    for cname, cn2 in CN2.items():
        for path in (200.0, 600.0, 1000.0):
            r0 = fried_r0(cn2, path)
            rows.append(
                {
                    "Cn2情景": cname,
                    "Cn2_m^-2/3": cn2,
                    "距离_m": path,
                    "r0_m": round(r0, 4),
                    "闪烁指数sigma_I2": round(rytov_sigma_i2(cn2, path), 3),
                    "到达角标准差_urad": round(angle_of_arrival_sigma(cn2, path) * 1e6, 3),
                    "靶面横向抖动_mm": round(angle_of_arrival_sigma(cn2, path) * path * 1e3, 2),
                    "光束漂移_mm": round(beam_wander_sigma(cn2, path) * 1e3, 2),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 热晕
def thermal_blooming(power: float, alpha: float, path: float, wind: float, radius: float) -> dict[str, float]:
    """稳态热晕: 温升 -> 光程差 -> 畸变数（文献[19][20][24][25]).

    dT   = alpha * P / (2*pi*rho*cp*v*a)          [K]
    OPD  = |dn/dT| * dT * L                        [m]
    N_D  = OPD / lambda                            [波数]
    """
    dtemp = alpha * power / (2.0 * math.pi * RHO_CP_AIR * wind * radius)
    opd = abs(DN_DT_AIR) * dtemp * path
    n_distortion = opd / LAMBDA
    t_bloom = 2.0 * radius / wind
    return {
        "温升_K": dtemp,
        "光程差_um": opd * 1e6,
        "畸变数N_D": n_distortion,
        "热晕建立时间_s": t_bloom,
    }


def blooming_table(fit: dict[str, float]) -> pd.DataFrame:
    """按“光斑半径随距离线性增长”的实测反演几何计算热晕.

    半径取第 4 节反演光斑直径拟合 d(L)=d0+theta*L 的一半，保证与湍流扩展表同一几何口径。
    """
    d0_mm = fit.get("d0_mm", FALLBACK_SPOT_D0_MM) if fit else FALLBACK_SPOT_D0_MM
    theta_urad = fit.get("theta_urad", FALLBACK_SPOT_THETA_URAD) if fit else FALLBACK_SPOT_THETA_URAD
    rows = []
    for path in (200.0, 600.0, 1000.0):
        radius = 0.5 * (d0_mm * 1e-3 + theta_urad * 1e-6 * path)
        for wname, wind in (("风机0/静风", 0.5), ("风机1", 2.0), ("风机2", 3.5)):
            for ename, alpha in EXTINCTION.items():
                r = thermal_blooming(P_STD, alpha, path, wind, radius)
                rows.append(
                    {
                        "距离_m": path,
                        "光斑半径_mm": round(radius * 1e3, 1),
                        "风况": wname,
                        "风速_m/s": wind,
                        "消光情景": ename,
                        "消光系数_1/m": alpha,
                        **{k: round(v, 6) if k != "温升_K" else round(v, 4) for k, v in r.items()},
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 熔穿能量学
def melt_energy_per_area(thickness_mm: float, absorptance: float, eta: float = ETA) -> float:
    """熔穿单位面积所需吸收能量 E'' = h * rho * (cp*dT + L_f) / (A*eta) [J/m2]."""
    dtemp = T_MELT - T_AMB
    return thickness_mm * 1e-3 * RHO_STEEL * (CP_STEEL * dtemp + L_FUSION) / (absorptance * eta)


def implied_intensity(breakdown_s: float, thickness_mm: float, absorptance: float) -> float:
    """由击穿时间反演的靶面吸收强度 I_abs = E''/t [W/m2]."""
    return melt_energy_per_area(thickness_mm, absorptance) / breakdown_s


def spot_diameter(power: float, intensity: float) -> float:
    """由功率与强度反演光斑直径 d = sqrt(4P/(pi I)) [m]."""
    return math.sqrt(4.0 * power / (math.pi * intensity))


def melt_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        t = row["击穿时间_s"]
        if pd.isna(t) or t <= 0:
            continue
        h = row["厚度_mm"] if pd.notna(row["厚度_mm"]) else 1.0
        a = ABSORPTANCE["未记录"]
        i_abs = implied_intensity(float(t), float(h), a)
        d = spot_diameter(P_STD, i_abs / a)
        rows.append(
            {
                "试验编号": row["试验编号"],
                "距离_m": row["距离_m"],
                "厚度_mm": h,
                "击穿时间_s": t,
                "反演吸收强度_MW/m2": round(i_abs / 1e6, 3),
                "反演光斑直径_mm": round(d * 1e3, 1),
            }
        )
    return pd.DataFrame(rows)


def load_table() -> pd.DataFrame:
    df = pd.read_csv(TABLE_CSV, encoding="utf-8-sig")
    dist = df["试验距离"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    df["距离_m"] = dist
    thick = df["厚度"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    df["厚度_mm"] = thick
    bd = df["击穿时间"].astype(str).str.extract(r"([\d.]+)")[0].astype(float)
    df["击穿时间_s"] = bd.where(df["击穿时间"].astype(str).str.contains(r"[0-9]", na=False))
    return df


# ---------------------------------------------------------------- 吸收率与消光
def fresnel_absorptance(theta_deg: float, n: float = 3.8, k: float = 4.2) -> float:
    """金属表面的菲涅耳吸收率 A = 1 - (R_s + R_p)/2（复折射率 eta = n + ik）."""
    theta = math.radians(theta_deg)
    cost = math.cos(theta)
    sint = math.sin(theta)
    eta = complex(n, k)
    cos_t = (1.0 - (sint / eta) ** 2) ** 0.5
    rs = (cost - eta * cos_t) / (cost + eta * cos_t)
    rp = (eta * cost - cos_t) / (eta * cost + cos_t)
    return 1.0 - (abs(rs) ** 2 + abs(rp) ** 2) / 2.0


def absorptance_table() -> pd.DataFrame:
    rows = []
    for theta in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0):
        rows.append({"入射角_deg": theta, "吸收率A": round(fresnel_absorptance(theta), 4)})
    return pd.DataFrame(rows)


def transmittance_table() -> pd.DataFrame:
    rows = []
    for ename, alpha in EXTINCTION.items():
        for path in (200.0, 600.0, 1000.0):
            rows.append(
                {
                    "消光情景": ename,
                    "消光系数_1/m": alpha,
                    "距离_m": path,
                    "大气透过率": round(math.exp(-alpha * path), 4),
                }
            )
    return pd.DataFrame(rows)


def convective_flux(wind: float, dtemp: float = 1400.0) -> float:
    """平板强制对流散热热流密度 q = h*dT，h 取 10 + 4*v [W/m2K]."""
    h = 10.0 + 4.0 * wind
    return h * dtemp


def geometry_fit(melt: pd.DataFrame) -> dict[str, float]:
    """由反演光斑直径随距离的线性趋势拟合 d = d0 + theta*L."""
    if melt.empty:
        return {}
    x = melt["距离_m"].to_numpy(dtype=float)
    y = melt["反演光斑直径_mm"].to_numpy(dtype=float)
    theta, d0 = np.polyfit(x, y, 1)
    return {"d0_mm": float(d0), "theta_urad": float(theta * 1e3)}


def turbulence_spread_table(fit: dict[str, float]) -> pd.DataFrame:
    """几何光斑叠加湍流长期扩展: sigma_LT = 2L/(k r0)."""
    rows = []
    d0 = fit.get("d0_mm", FALLBACK_SPREAD_D0_MM) if fit else FALLBACK_SPREAD_D0_MM
    theta = fit.get("theta_urad", FALLBACK_SPREAD_THETA_URAD) if fit else FALLBACK_SPREAD_THETA_URAD
    for cname, cn2 in CN2.items():
        for path in (200.0, 600.0, 1000.0):
            r0 = fried_r0(cn2, path)
            a_geom = 0.5 * (d0 * 1e-3 + theta * 1e-6 * path)
            sigma_lt = 2.0 * path / (K_WAVE * r0)
            a_eff = math.sqrt(a_geom**2 + sigma_lt**2)
            rows.append(
                {
                    "Cn2情景": cname,
                    "距离_m": path,
                    "几何光斑半径_mm": round(a_geom * 1e3, 1),
                    "湍流扩展_mm": round(sigma_lt * 1e3, 2),
                    "等效光斑半径_mm": round(a_eff * 1e3, 1),
                    "强度下降系数": round((a_eff / a_geom) ** 2, 3),
                }
            )
    return pd.DataFrame(rows)


HOTKEY_COL = "风机数量"


def confound_table(df: pd.DataFrame) -> pd.DataFrame:
    """距离与其他分组变量的交叉计数，用于识别设计共线."""
    rows = []
    pairs = [("厚度", "厚度_mm"), ("风机数量", "风机数量"), ("入射方向", "入射方向")]
    for label, col in pairs:
        if col not in df.columns:
            continue
        for dist, part in df.groupby("距离_m"):
            counts = part[col].astype(str).value_counts()
            rows.append(
                {
                    "距离_m": int(dist),
                    "变量": label,
                    "取值分布": "；".join(f"{k}={v}" for k, v in counts.items() if k not in ("nan", "None")),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 汇总报告
def fmt(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = [str(row[c]) for c in df.columns]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_report(
    turb: pd.DataFrame,
    spread: pd.DataFrame,
    bloom: pd.DataFrame,
    melt: pd.DataFrame,
    absorp: pd.DataFrame,
    trans: pd.DataFrame,
    confound: pd.DataFrame,
    fit: dict[str, float],
) -> str:
    lines: list[str] = []
    lines.append("# 理论计算汇总（小系统试验）\n")
    lines.append(
        "激光工作于近红外常用波段；功率为「标准功率 / 低功率」两档。按脱敏要求，波长、口径、发散角、"
        "脉宽、重频、功率密度、能量等激光参数均以文字描述、不记录具体数值，数值只存于项目根目录 `.env`（由 `scripts/xiaoxitong_config.py` 读取），不写入任何产物。"
    )
    lines.append("以下表格由 `scripts/xiaoxitong_theory.py` 自动生成，公式来源见《参考文献表.md》。")
    lines.append("所有数值均为情景计算：输入量（Cn2、消光系数、吸收率、效率因子）取文献典型范围，非本试验实测。\n")

    lines.append("## 1. 湍流参数（平面波，水平路径）\n")
    lines.append(fmt(turb))
    lines.append("")
    lines.append("要点: Cn2=1e-14（近地面白天典型值）时，1 km 路径的 r0 约 5 cm，与常见的数厘米发射口径同量级；")
    lines.append("到达角起伏折算到靶面的横向抖动 200 m 约 0.7 mm、1 km 约 8 mm，与反演光斑半径同量级。")
    lines.append("闪烁指数 sigma_I2 在 1 km 达 0.31（接近弱起伏上限），但秒级驻留会把闪烁在时间上平均掉，")
    lines.append("因此闪烁不改变累计能量，真正起作用的是长期光斑扩展与光束漂移。\n")

    lines.append("## 2. 湍流长期扩展对光斑与强度的影响\n")
    lines.append(fmt(spread))
    lines.append("")
    lines.append("要点: 长期扩展 sigma_LT = 2L/(k r0)，与几何光斑平方相加。Cn2=1e-14 时 1 km 的强度下降系数仅 1.28，")
    lines.append("强对流（1e-13）时达 5.4。湍流扩展只是几何稀释（200 m→1 km 约 4 倍）之外的二级修正，")
    lines.append("单靠它解释不了 1 km 与 200 m 之间的击穿时间差，但它能把 1 km 的缺额再放大 1.3–2 倍。\n")

    lines.append("## 3. 稳态热晕（沿程吸收加热空气）\n")
    lines.append(fmt(bloom))
    lines.append("")
    lines.append(
        "要点: 畸变数 N_D ∝ αP L²/(v a)。洁净空气（α=1e-5 /m）静风时 1 km 的 N_D≈1，正好落在“可分辨畸变”的临界；"
    )
    lines.append(
        "风速 2 m/s 时降到 0.24，3.5 m/s 时 0.14，可以忽略。浓霾（α=2e-4 /m）静风时 200 m/600 m/1 km 分别约 7.8/15.6/19.5，"
    )
    lines.append(
        "轻霾（α=5e-5 /m）静风时 1 km 约 4.9，均足以把焦斑打散；N_D≫1 时线性理论已进入饱和区会高估畸变（文献[28][29]），"
    )
    lines.append("但“浓霾 + 静风 + 长距离不可忽略”的量级判据仍然成立。")
    lines.append("风机把等效横向风速由 0.5 m/s 提到 3.5 m/s，N_D 同比例下降约 7 倍，这是风机最直接的作用；")
    lines.append("而风对靶板的强制对流散热仅 10^2 kW/m2 量级，相对入射的高功率密度可忽略。")
    lines.append("所以“开风机提高击穿率”不可能来自冷却靶板，只能来自抑制热晕、吹散烟尘/羽流与稳定熔池。\n")

    lines.append("## 4. 熔穿能量学与实测反演\n")
    lines.append(fmt(melt))
    lines.append("")
    lines.append("要点: 按 1 mm 板、吸收率 0.40、效率因子 0.6 计算，熔穿需要在靶面沉积很大的能量面密度；")
    lines.append("由此反演的有效光斑直径随距离近似线性增长（远端明显大于近端）。")
    lines.append("拟合表明有效光斑随距离线性增大，其等效发散角小于同口径的衍射极限全角，")
    lines.append("说明实际发射口径更大或采用聚焦，几何光斑主要由初始口径与离焦状态决定。\n")

    lines.append("## 5. 金属表面菲涅耳吸收率随入射角变化（不锈钢 n=3.8, k=4.2）\n")
    lines.append(fmt(absorp))
    lines.append("")
    lines.append("要点: 该波段下不锈钢法向吸收率约 0.37，随入射角单调上升，75° 时约 0.45。")
    lines.append("但斜入射同时把光斑面积放大 1/cosθ 倍，30° 内净效应是强度下降约 13%、吸收率仅增 1–2%，")
    lines.append("理论预期斜入射略慢而非更快。实测斜入射中位击穿时间反而更短（3.64 s vs 5.14 s），")
    lines.append("但该差异统计不显著（p≈0.3–0.9），且与批次/材料混杂，不能解释为角度物理效应。\n")

    lines.append("## 6. 大气透过率（Beer–Lambert）\n")
    lines.append(fmt(trans))
    lines.append("")
    lines.append("要点: 洁净至浓霾条件下 1 km 透过率仍在 0.82–0.99，消光不是主因。")
    lines.append("除非出现雾（能见度 <1 km）或降水，该波段的能量损失远小于几何与湍流造成的强度稀释。\n")

    lines.append("## 7. 分组设计与共线结构（解释力的边界）\n")
    lines.append(fmt(confound))
    lines.append("")
    lines.append(
        "要点: 距离与批次、厚度、作用时间、风速高度共线（200 m 批次以 2 mm 板为主，600 m 批次几乎全为 1 mm 板），"
    )
    lines.append("因此“厚度效应”“距离效应”在本数据中不可分离识别；跨批次的相关系数只能作描述，不能作因果。\n")

    lines.append("## 8. 结论性量级\n")
    lines.append("- 击穿时间的主导项是靶面有效功率密度，不是大气消光；")
    d200 = fit.get("d0_mm", FALLBACK_SPOT_D0_MM) + fit.get("theta_urad", FALLBACK_SPOT_THETA_URAD) * 1e-3 * 200.0
    d1000 = fit.get("d0_mm", FALLBACK_SPOT_D0_MM) + fit.get("theta_urad", FALLBACK_SPOT_THETA_URAD) * 1e-3 * 1000.0
    area_ratio = (d1000 / d200) ** 2
    t_geom = 3.40 * area_ratio
    t_turb = t_geom * 1.28
    lines.append("- 200→600 m 的中位击穿时间只增加 1.5 倍，小于几何光斑面积增长的幅度，")
    lines.append("  说明 600 m 组实际耦合效率优于纯几何预期（可能来自聚焦设置或靶面状态差异）；")
    lines.append(f"- 1 km 组在 25–40 s 内未记录到数值击穿。按几何扩展预期约 {t_geom:.1f} s，")
    lines.append(f"  叠加 Cn2=1e-14 的湍流扩展后约 {t_turb:.1f} s，实测所需时间至少慢 {25.0 / t_turb:.1f} 倍；")
    lines.append("  剩余差距可由抖动脱靶与（有霾时）热晕联合承担，但需补充 Cn2、能见度、")
    lines.append("  光斑诊断与靶面抛光/氧化记录后才能定量归因。\n")
    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_table()

    turb = turbulence_table()
    melt = melt_table(df)
    absorp = absorptance_table()
    trans = transmittance_table()
    fit = geometry_fit(melt)
    bloom = blooming_table(fit)
    spread = turbulence_spread_table(fit)
    confound = confound_table(df)

    # 脱敏：激光几何与功率密度相关列（光斑半径、反演光斑直径、反演吸收强度）不写入产物；
    # 这些列仍保留在内部 DataFrame 中供 geometry_fit 等计算使用。
    laser_cols = {
        "spread": ["几何光斑半径_mm", "等效光斑半径_mm"],
        "bloom": ["光斑半径_mm"],
        "melt": ["反演吸收强度_MW/m2", "反演光斑直径_mm"],
    }
    spread_out = spread.drop(columns=laser_cols["spread"], errors="ignore")
    bloom_out = bloom.drop(columns=laser_cols["bloom"], errors="ignore")
    melt_out = melt.drop(columns=laser_cols["melt"], errors="ignore")

    numeric = pd.concat(
        [
            turb.assign(表="湍流"),
            spread_out.assign(表="湍流扩展"),
            bloom_out.assign(表="热晕"),
            melt_out.assign(表="熔穿反演"),
            absorp.assign(表="吸收率"),
            trans.assign(表="透过率"),
            confound.assign(表="分组共线"),
        ],
        ignore_index=True,
    )
    numeric.to_csv(REPORT_DIR / "理论计算_数值.csv", index=False, encoding="utf-8-sig")

    md = write_report(turb, spread_out, bloom_out, melt_out, absorp, trans, confound, fit)
    (REPORT_DIR / "理论计算.md").write_text(md, encoding="utf-8")

    print(f"turb={len(turb)} spread={len(spread)} bloom={len(bloom)} melt={len(melt)} confound={len(confound)}")
    print(f"fit d0={fit.get('d0_mm', 0):.2f}mm theta={fit.get('theta_urad', 0):.1f}urad")
    print(f"output={REPORT_DIR / '理论计算.md'}")


if __name__ == "__main__":
    main()
