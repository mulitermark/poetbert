import pandas as pd
import json

def write_data(df: pd.DataFrame, out_file: str):
    with open(out_file, 'w', encoding='utf-8') as wf:
        df.to_json(wf, orient='records', force_ascii=False)

def read_json_dataframe(data_path: str):
    with open(data_path, 'r', encoding='utf-8') as rf:
        return json.load(rf)