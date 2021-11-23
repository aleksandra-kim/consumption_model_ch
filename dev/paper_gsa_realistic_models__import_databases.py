import bw2data as bd
import bw2calc as bc
import bw2io as bi
from pathlib import Path
from gwp_uncertainties import add_bw_method_with_gwp_uncertainties

# Local files
from consumption_model_ch.import_databases import (
    import_consumption_db,
)
from consumption_model_ch.consumption_fus import (
    add_consumption_activities,
    add_consumption_categories,
    add_consumption_sectors,
)

delete_consumption_db = False
add_functional_units = True

path_base = Path('/Users/akim/Documents/LCA_files/')
fp_gsa_project = path_base / "brightway2-project-GSA-backup.16-November-2021-11-50AM.tar.gz"
directory_habe = path_base / 'HABE_2017/'
fp_ei33 = path_base / 'ecoinvent_33_cutoff/datasets'

project = "GSA"

# Restore GSA project
if project not in bd.projects:
    bi.restore_project_directory(fp_gsa_project)

bd.projects.set_current(project)

# Import all databases
ei38_name = "ecoinvent"
exclude_databases = [
    'heia',
    'Agribalyse 1.2',
    'Agribalyse 1.3 - {}'.format(ei38_name),
    'exiobase 2.2',
]
co_name = "CH consumption 1.0"
if delete_consumption_db:
    del bd.databases[co_name]
import_consumption_db(directory_habe, fp_ei33, co_name, exclude_databases)
co = bd.Database(co_name)

# Add functional units
option = 'aggregated'
if add_functional_units:
    add_consumption_activities(co_name, option=option,)
    add_consumption_categories(co_name)
    add_consumption_sectors(co_name)

# Add uncertainties to GWP values
method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
if method not in bd.methods:
    add_bw_method_with_gwp_uncertainties()

# LCA
co_average_act_name = 'ch hh average consumption {}'.format(option)
hh_average = [act for act in co if co_average_act_name == act['name']]
assert len(hh_average) == 1
demand_act = hh_average[0]
lca = bc.LCA({demand_act: 1}, method)
lca.lci()
lca.lcia()
print(demand_act['name'], lca.score)

# food = [act for act in co if "Food" in act['name']]
# assert len(food) == 1
# demand_act = food[0]

# transport = [act for act in co if "Transport" in act['name']]
# assert len(transport) == 1
# demand_act = transport[0]

sectors = [act for act in co if "sector" in act['name'].lower()]
sum_ = 0
for demand_act in sectors:
    lca = bc.LCA({demand_act: 1}, method)
    lca.lci()
    lca.lcia()
    print(demand_act['name'], lca.score)
    sum_ += lca.score
print(sum_)

