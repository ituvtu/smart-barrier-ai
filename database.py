import pandas as pd
from sqlalchemy import create_engine, text
import os

class DataManager:
    def __init__(self):
        self.allowed_plates = set()
        self.source_info = "Empty"

    def load_source(self, file_obj=None, db_url=None, sql_query="SELECT plate FROM users"):
        try:
            new_plates = []

            if file_obj is not None:
                file_path = file_obj.name
                self.source_info = f"File: {os.path.basename(file_path)}"

                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(file_path)
                elif file_path.endswith('.json'):
                    df = pd.read_json(file_path)
                else:
                    return False, "Unsupported file format"

                target_col = None
                candidates = ['plate', 'number', 'license', 'num', 'id']

                for col in df.columns:
                    if str(col).lower() in candidates:
                        target_col = col
                        break

                if not target_col:
                    target_col = df.columns[0]

                new_plates = df[target_col].astype(str).tolist()

            elif db_url:
                self.source_info = "Remote DB"
                engine = create_engine(db_url)
                with engine.connect() as conn:
                    result = conn.execute(text(sql_query))
                    new_plates = [row[0] for row in result]

            self.allowed_plates = set(str(p).strip().upper().replace(' ', '') for p in new_plates)
            return True, f"Loaded {len(self.allowed_plates)} plates"

        except Exception as e:
            return False, str(e)
