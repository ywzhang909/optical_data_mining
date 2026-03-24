from typing import Optional
import pandas as pd
from sqlalchemy import create_engine
from .base_output import OutputSink

class SQLOutput(OutputSink):
    def __init__(self, connection_url: Optional[str] = None, db_config: Optional[object] = None):
        super().__init__()
        self.engine = None
        if connection_url:
            self.engine = create_engine(connection_url)
        elif db_config is not None:
            # expect db_config has fields: db_user, db_password, db_host, db_port, db_name
            url = f"mysql+pymysql://{db_config.db_user}:{db_config.db_password}@{db_config.db_host}:{db_config.db_port}/{db_config.db_name}"
            self.engine = create_engine(url)

    def write(self, df: pd.DataFrame, table_name: str) -> dict:
        if self.engine is None:
            raise ValueError("No database engine configured for SQLOutput")
        df.to_sql(table_name, con=self.engine, if_exists='append', index=False)
        return {"table": table_name, "rows": len(df)}
