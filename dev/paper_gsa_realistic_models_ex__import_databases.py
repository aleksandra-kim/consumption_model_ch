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
    import_exiobase_3,
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
    ei38_path = path_base / 'ecoinvent_38_cutoff/datasets'
    ei38_name = 'ecoinvent 3.8 cutoff'
    heia_path = path_base / "heia"
    habe_path = path_base / 'HABE_2017/'
    habe_year = "091011"
    # co_path = DIRPATH.parent / "consumption_model_ch" / "data" / 'es8b01452_si_002.xlsx'
    archetypes_path = heia_path / "hh_archetypes_weighted_ipcc_{}.csv".format(habe_year)
    ex381_path = path_base / 'exiobase_381_monetary/IOT_2015_pxp/'
    ex381_name = "Exiobase 3.8.1 Monetary 2015"
    sut_path = path_base / 'exiobase_SUT/data/'

    project = "GSA for realistic models exiobase"
    # try:
    #     bd.projects.delete_project(project, delete_dir=True)
    # except:
    #     pass
    bd.projects.set_current(project)

    co_name = CONSUMPTION_DB_NAME

    bi.bw2setup()
    import_ecoinvent(ei38_path, ei38_name)
    import_exiobase_3(ex381_path, ex381_name)
    # current_project = deepcopy(bd.projects.current)
    # bd.projects.set_current(ei33_name)  # Temporarily switch to ecoinvent 3.3 project
    # bi.bw2setup()

    create_ecoinvent_33_project(ei33_path)
    exclude_dbs = [
        'heia',
        'Agribalyse 1.2',
        'Agribalyse 1.3 - {}'.format(ei38_name),
    ]
    try:
        del bd.databases[CONSUMPTION_DB_NAME]
    except:
        pass
    co = import_consumption_db(
        habe_path,
        exclude_databases=exclude_dbs,
        ei_name=ei38_name,
        exiobase_path=ex381_path,
        sut_path=sut_path,
    )
    add_consumption_activities(co_name, habe_path, habe_year, option='aggregated',) #TODO check if this option is correct
    add_consumption_categories(co_name)
    add_consumption_sectors(co_name)
    add_archetypes_consumption(co_name, archetypes_path)

    co = bd.Database("CH consumption 1.0")
    demand_act = [act for act in co if "Food" in act["name"]]
    assert len(demand_act) == 1
    demand_act = demand_act[0]
    demand = {demand_act: 1}
    uncertain_method = ("IPCC 2013", "climate change", "GWP 100a")
    lca = bc.LCA(demand, uncertain_method)
    lca.lci()
    lca.lcia()
    print(lca.score)

    consumption_acts = [act for act in co if "consumption" in act['name']]
    for act in consumption_acts:
        lca = bc.LCA({act: 1}, uncertain_method)
        lca.lci()
        lca.lcia()
        print(act['name'], lca.score)

