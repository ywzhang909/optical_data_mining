#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spots Data Pipeline
根目录下的简单数据处理管线，用于从指定输入源（本地目录/ZIP等）读取光斑 TIFF 数据，
计算基础特征（总功率、质心、二阶矩等），并输出到多种格式（JSON、Excel、SQL等）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

try:
    from data_mining.image.common import read_tiff_to_numpy
except Exception:
    # fallback if package path differs during quick experiments
    from src.data_mining.image.common import read_tiff_to_numpy  # type: ignore

from scipy.ndimage import center_of_mass


class SpotsDataPipeline:
    """简单的 Spots 数据处理管线：从目录读取 TIFF，计算统计特征，输出到多种格式"""

    def __init__(self, input_dir: str, output_dir: Optional[str] = None, output_format: str = 'parquet'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir
        self.output_format = output_format.lower()

    def _collect_tiff_paths(self) -> List[Path]:
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        paths = sorted([p for p in self.input_dir.rglob('*') if p.is_file() and p.suffix.lower() in {'.tiff', '.tif', '.tifff'}})
        return paths

    def _process_one(self, path: Path) -> Dict[str, Any]:
        img = read_tiff_to_numpy(str(path))
        if img is None:
            raise ValueError(f"Failed to read TIFF: {path}")
        total_power = float(np.sum(img))
        cy, cx = center_of_mass(img)

        h, w = img.shape if img.ndim == 2 else (img.shape[0], img.shape[1])
        y = np.arange(h)
        x = np.arange(w)
        yy, xx = np.meshgrid(y, x, indexing='ij')  # yy.shape==(h,w), xx==(h,w)
        # 二阶矩（带权重的方差）
        mu_xx = np.sum(((xx - cx) ** 2) * img) / total_power if total_power > 0 else 0.0
        mu_yy = np.sum(((yy - cy) ** 2) * img) / total_power if total_power > 0 else 0.0
        sigma_x2 = float(mu_xx)
        sigma_y2 = float(mu_yy)
        result = {
            'path': str(path),
            'total_power': total_power,
            'centroid_x': float(cx),
            'centroid_y': float(cy),
            'sigma_x2': sigma_x2,
            'sigma_y2': sigma_y2,
        }
        return result

    def run(self) -> pd.DataFrame:
        paths = self._collect_tiff_paths()
        records: List[Dict[str, Any]] = []
        for p in paths:
            try:
                rec = self._process_one(p)
                records.append(rec)
            except Exception as e:
                # 跳过错误，但记录错误信息以便追踪
                records.append({
                    'path': str(p),
                    'error': str(e)
                })
        df = pd.DataFrame(records)
        return df

    def save(self, df: pd.DataFrame) -> Dict[str, Any]:
        # 基本输出，支持 json/excel/ Parquet 等
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = None
        if self.output_format in {'json'}:
            out_path = str(self.output_dir / 'spots_output.json')
            df.to_json(out_path, orient='records', lines=False, force_ascii=False)
        elif self.output_format in {'excel', 'xlsx'}:
            out_path = str(self.output_dir / 'spots_output.xlsx')
            df.to_excel(out_path, index=False)
        elif self.output_format in {'parquet'}:
            out_path = str(self.output_dir / 'spots_output.parquet')
            try:
                df.to_parquet(out_path, index=False)
            except Exception:
                # Fallback to CSV if parquet engine is unavailable
                out_path = str(self.output_dir / 'spots_output.csv')
                df.to_csv(out_path, index=False)
        else:
            # default to CSV
            out_path = str(self.output_dir / 'spots_output.csv')
            df.to_csv(out_path, index=False)
        return {'output_path': out_path, 'rows': len(df)}

def main():
    # 简单示例：当前目录的 input_spots 下读 TIFF，输出到 output_spots
    input_dir = os.environ.get('SPOTS_INPUT_DIR', './spots_input')
    out_dir = os.environ.get('SPOTS_OUTPUT_DIR', './spots_output')
    pipeline = SpotsDataPipeline(input_dir, out_dir, output_format='parquet')
    df = pipeline.run()
    stats = pipeline.save(df)
    print(f"Processed {stats['rows']} spots, output at {stats['output_path']}")

if __name__ == '__main__':
    main()
