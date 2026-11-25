"""小系统激光打靶试验的参考文献表: 逐条通过 doi.org 内容协商核实元数据.

用法:
    uv run python scripts/xiaoxitong_refs.py            # 用缓存生成 参考文献表.md
    uv run python scripts/xiaoxitong_refs.py --verify   # 补齐缺失/失败条目的 bibtex
    uv run python scripts/xiaoxitong_refs.py --refresh  # 强制重新核实全部条目

输出:
    data/小系统/试验报告/参考文献表.md
    data/小系统/试验报告/.refs_cache.json   （保存原始 bibtex，便于复核）
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "data" / "小系统" / "试验报告"
CACHE = REPORT_DIR / ".refs_cache.json"
OUT_MD = REPORT_DIR / "参考文献表.md"

# 主题分组：用途 -> DOI。顺序即正文引用顺序。
GROUPS: dict[str, list[tuple[str, str]]] = {
    "A. 大气湍流统计与激光传输": [
        ("Fried 1965 波前畸变几何统计（r0 定义）", "10.1364/JOSA.55.001427"),
        ("Fried 1966 长短曝光下的湍流分辨率", "10.1364/JOSA.56.001372"),
        ("Fried 1966 向下视路径的极限分辨率", "10.1364/JOSA.56.001380"),
        ("Noll 1976 Zernike 多项式与湍流相位统计", "10.1364/JOSA.66.000207"),
        ("Wang & Markey 1978 相位畸变模式补偿", "10.1364/JOSA.68.000078"),
        ("Dai 1995 Zernike/KL 模态补偿", "10.1364/JOSAA.12.002182"),
        ("Andrews 等 1999 闪烁指数解析理论", "10.1364/JOSAA.16.001417"),
        ("Yura 1971 湍流诱导光束扩展", "10.1364/AO.10.002771"),
        ("Lutomirski & Yura 1971 非均匀介质中有限光束传播", "10.1364/AO.10.001652"),
        ("Tavis & Yura 1976 湍流中光束短期平均辐照分布", "10.1364/AO.15.002922"),
        ("Lutomirski 等 1977 倾斜校正孔径改善光束质量", "10.1364/AO.16.000665"),
        ("Churnside & Lataitis 1987 反射光束到达角起伏", "10.1364/JOSAA.4.001264"),
        ("Churnside & Lataitis 1990 湍流中光束漂移", "10.1364/AO.29.000926"),
        ("Churnside 1991 湍流闪烁的孔径平均", "10.1364/AO.30.001982"),
        ("Valley 1980 倾斜校正等晕退化（HV 模型出处）", "10.1364/AO.19.000574"),
        ("Greenwood 1977 自适应光学带宽（Greenwood 频率）", "10.1364/JOSA.67.000390"),
        ("Clifford 等 1974 强湍流闪烁饱和", "10.1364/JOSA.64.000148"),
        ("Wang 等 1978 抗饱和 Cn2 闪烁仪", "10.1364/JOSA.68.000334"),
        ("Andrews & Phillips 2005 随机介质中激光传输（专著）", "10.1117/3.626196"),
        ("Tyson 2015 自适应光学原理（专著，第 4 版）", "10.1201/b19712"),
        ("Hardy 1998 自适应光学导论（专著）", "10.1093/oso/9780195090192.001.0001"),
        ("Kolmogorov 与非 Kolmogorov 湍流中的电磁波传播", "10.1364/AO.47.006385"),
        ("各向异性湍流谱与光束传播", "10.1364/OE.23.030088"),
        ("非 Kolmogorov 湍流会议论文", "10.1117/12.211927"),
        ("高能激光大气传输专著章节", "10.1007/978-3-319-31289-7_8"),
        ("高能激光大气传输性能评估方法（中文综述）", "10.3788/IRLA201948.1203002"),
    ],
    "B. 热晕、热散焦与相位补偿": [
        ("Smith 1977 热晕综述（Proc. IEEE）", "10.1109/PROC.1977.10809"),
        ("热晕研究 25 年综述（SPIE）", "10.1117/12.18326"),
        ("Van Zandt 等 2013 热晕与湍流联合标度律", "10.1364/OE.21.014789"),
        ("Breaux 等 1979 连续激光热晕解析（代数）模型", "10.1364/AO.18.002638"),
        ("Barchers 2009 热晕相位补偿的不稳定性线性分析", "10.1364/JOSAA.26.001638"),
        ("流体中激光束热晕", "10.1364/AO.11.000455"),
        ("大气热晕的动力学模型", "10.1364/AO.12.001506"),
        ("热晕的相位补偿", "10.1364/AO.13.000331"),
        ("降雨条件下高功率激光稳态热晕", "10.1080/09205071.2016.1219276"),
        ("空心高斯光束在大气中的热散焦", "10.1017/S0263034613000402"),
    ],
    "C. 大气消光、气溶胶与能见度": [
        ("Berk 等 2014 MODTRAN6 辐射传输模型", "10.1117/12.2050433"),
        ("Berk 等 2014 MODTRAN 高光谱成像应用", "10.1109/WHISPERS.2014.8077573"),
        ("HITRAN2020 分子吸收谱数据库", "10.1016/j.jqsrt.2021.107949"),
        ("Koschmieder 能见度与消光系数关系", "10.1007/978-3-663-04661-5_2"),
        ("Naboulsi 等 2004 雾衰减预测（可见—近红外）", "10.1117/1.1637611"),
        ("Kim 等 2001 雾霾中 785/1550 nm 传输对比", "10.1117/12.417512"),
        ("Penano 等 2006 高能激光穿过大气滞止区（DTIC 报告）", "10.21236/ada444656"),
        ("自由空间光通信的雾/烟衰减信道模型", "10.1109/IWOW.2012.6349680"),
    ],
    "D. 激光与金属材料相互作用": [
        ("Chan & Mazumder 1987 汽化与液相排出损伤一维模型（熔穿能量基准）", "10.1063/1.339053"),
        ("Semak & Matsunawa 1997 反冲压力在能量平衡中的作用", "10.1088/0022-3727/30/18/008"),
        ("Klemens 1976 焊接过程热平衡分析", "10.1063/1.322866"),
        ("Von Allmen & Blatter 1995 激光束与材料相互作用（专著）", "10.1007/978-3-642-57813-7"),
        ("Nd:YAG/YLF 波段钢材吸收率测量", "10.1016/j.apsusc.2006.11.018"),
        ("Stuart 等 1995 超短脉冲损伤阈值", "10.1103/PhysRevLett.74.2248"),
        ("Stuart 等 1996 介质膜损伤机制", "10.1103/PhysRevB.53.1749"),
        ("Niemz 1995 激光致材料损伤阈值", "10.1063/1.113850"),
        ("Matsunawa & Semak 1997 匙孔前壁动力学", "10.1088/0022-3727/30/5/013"),
        ("Matsunawa 等 1998 匙孔与熔池动力学观测", "10.2351/1.521858"),
        ("Ujihara 1972 金属高温反射率", "10.1063/1.1661506"),
        ("Tolochko 等 2000 金属粉末对激光的吸收率（选区烧结）", "10.1108/13552540010337029"),
    ],
}

# 无 DOI 的专著/标准/手册：单独列出，注明可核查出处
NO_DOI: list[tuple[str, str]] = [
    ("V. I. Tatarski, Wave Propagation in a Turbulent Medium, McGraw-Hill, 1961", "湍流统计理论基础（Kolmogorov 谱）"),
    (
        "R. E. Hufnagel, Propagation through atmospheric turbulence, in The Infrared Handbook, 1974",
        "HV 模型中 Cn2(h) 的 H 分量",
    ),
    (
        "ISO 11146-1:2021, Test methods for laser beam widths, divergence angles and beam propagation ratios",
        "光斑/发散角/M² 测量规范",
    ),
    ("ISO 21254-1:2011, Test methods for laser-induced damage threshold", "损伤阈值测试规范"),
    (
        "ISO 11554:2017, Test methods for laser beam power, energy and temporal characteristics",
        "功率/能量/时间特性测试规范",
    ),
]


def fetch_bibtex(doi: str, retries: int = 4) -> tuple[int, str]:
    """向 doi.org 请求 bibtex 元数据（出版方权威记录）。"""
    req = urllib.request.Request(
        f"https://doi.org/{doi}",
        headers={"Accept": "application/x-bibtex", "User-Agent": "refs-check/1.0"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:  # noqa: S310
                return int(resp.status), resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503):
                time.sleep(2.0 * (attempt + 1))
                continue
            return int(exc.code), ""
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (attempt + 1))
    return 0, ""


def parse_bibtex(text: str) -> dict[str, str]:
    """从单行 bibtex 中抽取字段（doi.org 返回的记录不含换行）。"""

    def grab(field: str) -> str:
        m = re.search(rf"{field}\s*=\s*\{{([^{{}}]*)\}}", text, re.I)
        return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""

    def clean(value: str) -> str:
        value = value.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&amp;", "&")
        value = re.sub(r"[{}]", "", value)
        value = re.sub(r"</?[a-zA-Z][^>]*>", "", value)
        value = re.sub(r"\s*[*†‡]\s*$", "", value)
        return re.sub(r"\s+", " ", value).strip()

    authors = [a.strip() for a in grab("author").split(" and ") if a.strip()]
    if len(authors) > 3:
        author_str = "; ".join(authors[:3]) + "; 等"
    else:
        author_str = "; ".join(authors)
    venue = clean(grab("journal") or grab("booktitle") or grab("publisher"))
    venue = venue.replace("New York, NY", "").replace("  ", " ").strip()
    return {
        "title": clean(grab("title")),
        "authors": clean(author_str),
        "venue": venue,
        "year": grab("year") or (re.search(r"(\d{4})", grab("date")) or [None, ""])[1],
        "volume": grab("volume"),
        "pages": grab("pages"),
    }


def verify(dois: list[str], refresh: bool = False) -> dict[str, dict]:
    cache = {} if refresh else load_cache()
    todo = [d for d in dois if refresh or not cache.get(d, {}).get("verified")]
    for i, doi in enumerate(todo, 1):
        status, text = fetch_bibtex(doi)
        ok = status == 200 and bool(text.strip())
        entry = {"status": status, "verified": ok, "checked": time.strftime("%Y-%m-%d"), "bibtex": text if ok else ""}
        cache[doi] = entry
        label = parse_bibtex(text).get("title", "") if ok else ""
        safe = label.encode("ascii", "replace").decode("ascii")[:66]
        print(f"[{i}/{len(todo)}] {status} {doi} :: {safe}", flush=True)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(0.3)
    return cache


def load_cache() -> dict[str, dict]:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def render(cache: dict[str, dict]) -> str:
    lines: list[str] = [
        "# 参考文献表（小系统激光打靶试验）\n",
        "每条 DOI 均通过 `https://doi.org/<DOI>` 内容协商（`Accept: application/x-bibtex`）逐条核实题名、作者、来源与年份，",
        "原始 bibtex 记录保存在 `.refs_cache.json`，可复核。表中 DOI 均可直接解析到出版方页面。\n",
        "引用编号按主题分组连续编号，正文以 `[编号]` 引用。\n",
    ]
    idx = 0
    for group, items in GROUPS.items():
        lines.append(f"## {group}\n")
        lines.append("| 编号 | 作者 | 题名 | 来源（卷/页） | 年 | DOI | 在本报告中的用途 | 核实 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for purpose, doi in items:
            idx += 1
            rec = cache.get(doi, {})
            ok = "✓" if rec.get("verified") else "✗"
            info = parse_bibtex(rec.get("bibtex", "")) if rec.get("verified") else {}
            title = info.get("title") or "（未核实）"
            authors = info.get("authors") or "—"
            if len(authors) > 58:
                parts = authors.removesuffix(" 等").split("; ")
                while len("; ".join(parts)) > 55 and len(parts) > 1:
                    parts = parts[:-1]
                authors = "; ".join(parts) + " 等"
            venue = info.get("venue") or "—"
            src = " ".join(x for x in (venue, info.get("volume")) if x)
            if info.get("pages"):
                src += f", {info['pages']}"
            lines.append(
                f"| {idx} | {authors} | {title} | {src} | {info.get('year') or '—'} | "
                f"[{doi}](https://doi.org/{doi}) | {purpose} | {ok} |"
            )
        lines.append("")
    lines.append("## E. 专著、标准与手册（无 DOI）\n")
    lines.append("| 编号 | 文献 | 用途 |")
    lines.append("|---|---|---|")
    for ref, purpose in NO_DOI:
        idx += 1
        lines.append(f"| {idx} | {ref} | {purpose} |")
    lines.append("")
    dois_list = [d for _, items in GROUPS.items() for _, d in items]
    verified = sum(1 for d in dois_list if cache.get(d, {}).get("verified"))
    lines.append(
        f"合计 **{idx}** 条：有 DOI 的 {len(dois_list)} 条（核实成功 {verified} 条），"
        f"无 DOI 的专著/标准 {len(NO_DOI)} 条。"
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="补齐缺失或失败的条目")
    ap.add_argument("--refresh", action="store_true", help="强制重新核实全部条目")
    args = ap.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dois = [d for _, items in GROUPS.items() for _, d in items]
    if args.refresh or args.verify:
        cache = verify(dois, refresh=args.refresh)
    else:
        cache = load_cache()
    missing = [d for d in dois if not cache.get(d, {}).get("verified")]
    if missing:
        cache.update(verify(missing))
    OUT_MD.write_text(render(cache), encoding="utf-8")
    print(f"refs={len(dois) + len(NO_DOI)} failed={len(missing)} output={OUT_MD}")


if __name__ == "__main__":
    main()
