import bw2data as bd
import bw2calc as bc
import bw2io as bi
from pathlib import Path
# from gwp_uncertainties import add_bw_method_with_gwp_uncertainties

# Local files
from consumption_model_ch.import_databases import (
    import_ecoinvent,
    import_exiobase_3,
    import_consumption_db,
)
from consumption_model_ch.consumption_fus import (
    add_consumption_activities,
    add_consumption_categories,
    add_consumption_sectors,
    add_archetypes_consumption,
)

if __name__ == "__main__":

    delete_consumption_db = True
    add_functional_units = True
    habe_year = '151617'

    path_base = Path('/Users/ajakobs/Documents/data/')
    path_base_2 = Path('/Users/ajakobs/Documents/CCL/consumption_model_ch/consumption_model_ch/data/functional_units/')
    # directory_habe = path_base / 'HABE/131_EBM 2009-2011/HABE091011_161128UOe/'
    directory_habe = path_base / 'HABE/131_EBM 2015-2017/HABE151617_191111UOe/'
    fp_ecoinvent_38 = path_base / 'ecoinvent/ecoinvent 3.8_cutoff_ecoSpold02/datasets/'
    fp_ecoinvent_33 = path_base / 'ecoinvent/ecoinvent 3.3_cutoff_ecoSpold02/datasets/'
    fp_exiobase = path_base / 'EXIOBASE/EX_3_8_1/IOT_2015_pxp/'
    fp_archetypes = path_base_2 / "hh_archetypes_weighted_working_tables_091011.csv"

    # fp_gsa_project = path_base / "brightway2-project-GSA-backup.16-November-2021-11-50AM.tar.gz"
    # directory_habe = path_base / 'HABE_2017/'
    # fp_ecoinvent_38 = path_base / "ecoinvent_38_cutoff" / "datasets"
    # fp_ecoinvent_33 = path_base / 'ecoinvent_33_cutoff/datasets'
    # fp_exiobase = path_base / "exiobase_381_monetary" / "IOT_2015_pxp"
    # fp_archetypes = path_base / "heia" / "hh_archetypes_weighted_ipcc_091011.csv"

    project = "CCL dev"
    bd.projects.set_current(project)

    # Import all databases
    bi.bw2setup()
    ei38_name = "ecoinvent 3.8"
    import_ecoinvent(fp_ecoinvent_38, ei38_name)
    ex38_name = "exiobase 3.8.1 monetary"
    import_exiobase_3(fp_exiobase, ex38_name)

    exclude_databases = [
        'heia',
        'Agribalyse 1.2',
        'Agribalyse 1.3 - {}'.format(ei38_name),
    ]
    co_name = "CH consumption 1.0"
    if delete_consumption_db and co_name in bd.databases:
        del bd.databases[co_name]
    import_consumption_db(directory_habe, habe_year, co_name, fp_ecoinvent_33, exclude_databases, fp_exiobase)
    co = bd.Database(co_name)

    # Add functional units
    option = 'aggregated'
    if add_functional_units:
        add_consumption_activities(co_name, habe_year, directory_habe, option=option)
        add_consumption_categories(co_name)
        add_consumption_sectors(co_name, habe_year)
        add_archetypes_consumption(co_name, fp_archetypes)

    # Add uncertainties to GWP values
    method = ('IPCC 2013', 'climate change', 'GWP 100a')  # ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    # if method not in bd.methods:
    #    add_bw_method_with_gwp_uncertainties()

    # LCA
    co_average_act_name = 'ch hh average consumption {}'.format(option)
    hh_average = [act for act in co if co_average_act_name == act['name']]
    assert len(hh_average) == 1
    demand_act = hh_average[0]
    lca = bc.LCA({demand_act: 1}, method)
    lca.lci()
    lca.lcia()
    print(demand_act['name'], lca.score)

    # sectors = [act for act in co if "sector" in act['name'].lower()]
    # sum_ = 0
    # for demand_act in sectors:
    #     lca = bc.LCA({demand_act: 1}, method)
    #     lca.lci()
    #     lca.lcia()
    #     print(demand_act['name'], lca.score)
    #     sum_ += lca.score
    # print(sum_)

    archetypes = [act for act in co if "archetype" in act['name'].lower()]
    for demand_act in archetypes:
        lca = bc.LCA({demand_act: 1}, method)
        lca.lci()
        lca.lcia()
        print(demand_act['name'], lca.score)

