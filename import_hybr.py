import brightway2 as bw
import numpy as np
import json, os

#Local files
from import_databases import *

with open('global_settings.json', 'rb') as f:
    settings = json.load(f)
which_pc = settings['which_pc']

if which_pc == 'local':
    # Older databases
    ei33_path = '/Users/akim/Documents/LCA_files/ecoinvent 3.3 cutoff/datasets'
    ex22_path = '/Users/akim/Documents/LCA_files/exiobase 2.2/mrIOT_IxI_fpa_coefficient_version2.2.2'
    ag12_path = '/Users/akim/Documents/LCA_files/agribalyse 1.2/AGRIBALYSEv1.2_Complete.xml'
    ei36_path = '/Users/akim/Documents/LCA_files/ecoinvent 3.6 cutoff/datasets'
    # Newer databases
    ei371_path = '/Users/akim/Documents/LCA_files/ecoinvent 3.7.1 cutoff/datasets'
    # ex34_path = '/Users/akim/Documents/LCA_files/exiobase 3.4/IOT_2011_pxp/' #TODO
    ag13_path = '/Users/akim/Documents/LCA_files/agribalyse 1.3/Agribalyse CSV FINAL_no links_Nov2016v3.CSV'
    habe_path = '/Users/akim/Documents/LCA_files/HABE_2017/'
    co_path = 'data/es8b01452_si_002.xlsx'
elif which_pc == 'merlin':
    ex22_path = '/data/user/kim_a/LCA_files/exiobase_22/mrIOT_IxI_fpa_coefficient_version2.2.2'
    ei33_path = '/data/user/kim_a/LCA_files/ecoinvent_33_cutoff/datasets'
    ei36_path = '/data/user/kim_a/LCA_files/ecoinvent_36_cutoff/datasets'
    ag13_path = '/data/user/kim_a/LCA_files/agribalyse_13/Agribalyse CSV FINAL_no links_Nov2016v3.CSV'
    habe_path = '/data/user/kim_a/LCA_files/HABE_2017'
    co_path = 'data/es8b01452_si_002.xlsx'

project = "GSA for hybrid"
bw.projects.set_current(project)

ei36_name = 'ecoinvent 3.6 cutoff'
ei371_name = 'ecoinvent 3.7.1 cutoff'
ei_option = "37"
if ei_option == "36":
    ei_name = ei36_name
    ei_path = ei36_path
elif ei_option == "37":
    ei_name = ei371_name
    ei_path = ei371_path

try:
    del bw.databases["Agribalyse 1.3 - {} - new biosphere".format(ei_name)]
    del bw.databases["Agribalyse 1.3 - {}".format(ei_name)]
    del bw.databases["CH consumption 1.0"]
except:
    pass

co_name = CONSUMPTION_DB_NAME

bw.bw2setup()
import_ecoinvent(ei_path, ei_name)
# import_agribalyse_13(ag13_path, ei_name)
# create_ecoinvent_33_project(ei33_path)
exclude_dbs = [
    'heia',
    'EXIOBASE 2.2',
    "Agribalyse 1.3 - {}".format(ei_name),
    "Agribalyse 1.3 - {} - new biosphere".format(ei_name),
]
import_consumption_db(co_path, habe_path, exclude_dbs=exclude_dbs, ei_name=ei_name,)
add_consumption_activities(co_name, habe_path)
add_consumption_categories(co_name, co_path)
add_consumption_sectors(co_name)

co = bw.Database('CH consumption 1.0')
demand_act = co.search('average consumption')[0]
# demand_act = [act for act in co if 'Food and non-alcoholic beverages sector' in act['name']][0]
demand = {demand_act: 1}
method = ('IPCC 2013', 'climate change', 'GWP 100a')

lca = bw.LCA(demand, method)
lca.lci()
lca.lcia()
print(demand_act, lca.score)