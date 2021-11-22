from bw2io import Migration

from .data import (
    get_ecoinvent_steam_pepper_migration_data,
    get_ecoinvent_rice_migration_data,
    get_ecoinvent_marine_fish_migration_data,
    get_exiobase_migration_data,
    get_exiobase_row_locations,
)


def create_consumption_db_migrations():
    """Function that creates migrations for the consumption database. TODO refactor this section"""

    Migration("ecoinvent-35-36-37-38-change-names").write(
        get_ecoinvent_steam_pepper_migration_data(),
        "Change names of ecoinvent activities",
    )
    Migration("ecoinvent-36-371-38-rice-nonbasmati").write(
        get_ecoinvent_rice_migration_data(),
        "Change names and reference products in ecoinvent"
    )
    Migration("ecoinvent-38-marine-fish").write(
        get_ecoinvent_marine_fish_migration_data(),
        "Change names of ecoinvent reference products",
    )
    Migration("exiobase-381-change-names").write(
        get_exiobase_migration_data(),
        "Change names of some exiobase 3.8.1 activities",
    )
    Migration("exiobase-row-locations").write(
        get_exiobase_row_locations(),
        "Change RoW locations in exiobase activities",
    )
