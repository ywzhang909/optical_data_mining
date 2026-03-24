import pandas as pd

class ExcelOutput:
    def write(self, data, path: str) -> None:
        if isinstance(data, pd.DataFrame):
            data.to_excel(path, index=False)
