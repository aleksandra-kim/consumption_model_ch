import pandas as pd
from pathlib import Path

dirpath = Path(__file__).parent.resolve()


def get_consumption_df():
    filepath = dirpath / "es8b01452_si_002.xlsx"
    df = pd.read_excel(filepath, sheet_name='Overview & LCA-Modeling', header=2)
    return df


def get_agribalyse_df():
    df = pd.read_excel(dirpath / "agribalyse_replaced_with_ecoinvent.xlsx")
    return df
