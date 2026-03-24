from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from .base_processor import BaseProcessor, ProcessingResult, ProcessorConfig
from .models import DatabaseConfig, Experiment

class DataLoaderConfig(ProcessorConfig):
    model_config = ConfigDict(extra='forbid')
    
    processor_name: str = Field(default="DataLoader", description="处理器名称")
    
    table_name: str = Field(..., description="数据表名")
    use_cache: bool = Field(default=True, description="使用缓存")
    cache_path: Optional[str] = Field(None, description="缓存路径")

class DataLoader(BaseProcessor[pd.DataFrame]):
    config_class = DataLoaderConfig
    
    def validate_input(self, data=None, **kwargs) -> tuple[bool, str]:
        input_data = kwargs.get('config', data)
        if input_data is None:
            return False, "Config is required"
        if not isinstance(input_data, DataLoaderConfig):
            return False, "Config must be a DataLoaderConfig"
        return True, ""
    
    def process(self, config: DataLoaderConfig, **kwargs) -> ProcessingResult:
        try:
            if load_dotenv('.env.prod'):
                db_config = DatabaseConfig(
                    db_user=os.getenv("DB_USER"),
                    db_password=os.getenv("DB_PASSWORD"),
                    db_host=os.getenv("DB_HOST"),
                    db_port=int(os.getenv("DB_PORT")),
                    db_name=os.getenv("DB_NAME")
                )
            else:
                db_config = DatabaseConfig(
                    db_user="root",
                    db_password="123456",
                    db_host="localhost",
                    db_port=3306,
                    db_name="data_mining"
                )
            
            url = f'mysql+pymysql://{db_config.db_user}:{db_config.db_password}@{db_config.db_host}:{db_config.db_port}/{db_config.db_name}'
            engine = create_engine(url)
            
            df = pd.read_sql_table(config.table_name, con=engine)
            
            df['date'] = pd.to_datetime(df['experiment_num'], format='%Y%m%d')
            df.set_index('experiment_id', inplace=True)
            
            df = self._extract_extra_fields(df)
            
            return ProcessingResult(
                success=True,
                message=f"成功加载数据，共 {len(df)} 条记录",
                data=df
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"数据加载失败: {str(e)}",
                error=str(e)
            )
    
    def _extract_extra_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'experiment_extra' in df.columns and df['experiment_extra'].notna().any():
            extra_data = df['experiment_extra'].apply(pd.Series)
            df = pd.concat([df, extra_data], axis=1)
        return df
