import bw2data as bd
import bw2calc as bc
import bw2io as bi

import numpy as np
from copy  import deepcopy
import json
from pathlib import Path

# Local files
from consumption_model_ch.import_databases import (
    import_ecoinvent,
    create_ecoinvent_33_project,
    import_consumption_db,
    add_consumption_activities,
    add_consumption_categories,
    add_consumption_sectors,
    add_archetypes_consumption,
)
from consumption_model_ch.utils_consumption_db import CONSUMPTION_DB_NAME

if __name__ == "__main__":

    with open('global_settings.json', 'rb') as f:
        settings = json.load(f)
    which_pc = settings['which_pc']

    ### TODO -> give paths to databases files
    if which_pc == 'local':
        path_base = Path('/Users/akim/Documents/LCA_files/')
    elif which_pc == 'merlin':
        path_base = Path('/data/user/kim_a/LCA_files/')
    DIRPATH = Path(__file__).parent.resolve()

    ei33_path = path_base / 'ecoinvent_33_cutoff/datasets'
    ei33_name = 'ecoinvent 3.3 cutoff'
    ei371_path = path_base / 'ecoinvent_371_cutoff/datasets'
    ei371_name = 'ecoinvent 3.7.1 cutoff'
    heia_path = path_base / "heia"
    habe_path = path_base / 'HABE_2017/'
    habe_year = "151617"
    # archetypes_path = heia_path / "hh_archetypes_weighted_ipcc_{}.csv".format(habe_year)

    project = "GSA for protocol paper 3"
    # try:
    #     bd.projects.delete_project(project, delete_dir=True)
    # except:
    #     pass
    bd.projects.set_current(project)

    co_name = CONSUMPTION_DB_NAME

    bi.bw2setup()
    import_ecoinvent(ei371_path, ei371_name)

    # current_project = deepcopy(bd.projects.current)
    # bd.projects.set_current(ei33_name)  # Temporarily switch to ecoinvent 3.3 project
    # bi.bw2setup()

    create_ecoinvent_33_project(ei33_path)
    exclude_dbs = [
        'heia',
        'Agribalyse 1.2',
        'Agribalyse 1.3 - {}'.format(ei371_name),
        'exiobase 2.2'
    ]
    # try:
    #     del bd.databases[CONSUMPTION_DB_NAME]
    # except:
    #     pass
    # co = import_consumption_db(
    #     habe_path,
    #     exclude_dbs=exclude_dbs,
    #     ei_name=ei371_name,
    # )
    # add_consumption_activities(co_name, habe_path, habe_year, option='aggregated',)
    # add_consumption_categories(co_name)
    # add_consumption_sectors(co_name)

    co = bd.Database("CH consumption 1.0")
    demand_act = [act for act in co if "Food" in act["name"]]
    assert len(demand_act) == 1
    demand_act = demand_act[0]
    demand = {demand_act: 1}
    uncertain_method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    lca = bc.LCA(demand, uncertain_method, use_distributions=False)
    lca.lci()
    lca.lcia()
    print(lca.score)

    print()