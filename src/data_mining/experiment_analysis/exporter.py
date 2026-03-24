from typing import Optional, Dict, Any
import pandas as pd
from pathlib import Path
from .base_processor import BaseProcessor, ProcessingResult, ProcessorConfig
from .models import AggregationResult

class DataExporterConfig(ProcessorConfig):
    model_config = ConfigDict(extra='forbid')
    output_format: str = Field(default='csv', description="输出格式: csv, excel, json")
    output_path: Optional[str] = Field(None, description="输出路径")
    include_metadata: bool = Field(default=True, description="包含元数据")

class DataExporter(BaseProcessor[Dict[str, Any]]):
    config_class = DataExporterConfig
    
    def process(self, config: DataExporterConfig, **kwargs) -> ProcessingResult:
        try:
            data = kwargs.get('data')
            if data is None:
                return ProcessingResult(
                    success=False,
                    message="缺少输入数据",
                    error="Data is required"
                )
            
            output_path = config.output_path or f"output/experiment_analysis.{config.output_format}"
            
            # 确保输出目录存在
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 根据格式导出数据
            if config.output_format == 'csv':
                if isinstance(data, pd.DataFrame):
                    data.to_csv(output_path, index=False, encoding='utf-8')
                else:
                    pd.DataFrame([data]).to_csv(output_path, index=False, encoding='utf-8')
            
            elif config.output_format == 'excel':
                if isinstance(data, pd.DataFrame):
                    data.to_excel(output_path, index=False)
                else:
                    pd.DataFrame([data]).to_excel(output_path, index=False)
            
            elif config.output_format == 'json':
                import json
                if isinstance(data, pd.DataFrame):
                    data_dict = data.to_dict('records')
                else:
                    data_dict = data
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data_dict, f, ensure_ascii=False, indent=2)
            
            else:
                return ProcessingResult(
                    success=False,
                    message=f"不支持的输出格式: {config.output_format}",
                    error=f"Unsupported output format: {config.output_format}"
                )
            
            result = {
                'output_path': output_path,
                'format': config.output_format,
                'size': Path(output_path).stat().st_size,
                'metadata': {
                    'output_format': config.output_format,
                    'include_metadata': config.include_metadata
                } if config.include_metadata else None
            }
            
            return ProcessingResult(
                success=True,
                message=f"数据导出成功: {output_path}",
                data=result
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"数据导出失败: {str(e)}",
                error=str(e)
            )
