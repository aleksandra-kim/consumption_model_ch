import pandas as pd
from pathlib import Path

dirpath = Path(__file__).parent.resolve()


def get_consumption_df():
    filepath = dirpath / "es8b01452_si_002.xlsx"
    df = pd.read_excel(filepath, sheet_name='Overview & LCA-Modeling', header=2)
    return df
