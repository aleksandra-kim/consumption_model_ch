import pandas as pd
from pathlib import Path
import json

dirpath = Path(__file__).parent.resolve()


def get_consumption_df():
    filepath = dirpath / "es8b01452_si_002.xlsx"
    df = pd.read_excel(filepath, sheet_name='Overview & LCA-Modeling', header=2)
    return df


def get_agribalyse_df():
    df = pd.read_excel(dirpath / "agribalyse_replaced_with_ecoinvent.xlsx")
    return df


def get_ecoinvent_steam_pepper_migration_data():
    return json.load(
        open(dirpath / "migrations" / "ecoinvent-3.5-3.6-3.7.1-3.8.json")
    )


def get_ecoinvent_rice_migration_data():
    return json.load(
        open(dirpath / "migrations" / "ecoinvent-3.6-3.7.1-3.8.json")
    )


def get_ecoinvent_marine_fish_migration_data():
    return json.load(
        open(dirpath / "migrations" / "ecoinvent-3.8.json")
    )


def get_exiobase_migration_data():
    return json.load(
        open(dirpath / "migrations" / "exiobase-3.8.1.json")
    )


def get_exiobase_row_locations():
    return json.load(
        open(dirpath / "migrations" / "exiobase-row-locations.json")
    )


def get_exiobase_linking_data():
    return json.load(
        open(dirpath / "migrations" / "exiobase-linking.json")
    )


def get_exiobase_margins_data(exiobase_year, sheet_name):
    df = pd.read_excel(
        dirpath / "CH_{}.xls".format(exiobase_year),
        sheet_name=sheet_name,
        skiprows=11
    )
    return df
