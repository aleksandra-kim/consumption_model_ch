import bw2data as bd
import bw2calc as bc
import bw2io as bi

import numpy as np
from copy  import deepcopy
import json
from pathlib import Path

# Local files
from consumption_model_ch import ConsumptionDbImporter


path_base = Path('/Users/akim/Documents/LCA_files/')
habe_directory = path_base / 'HABE_2017/'
bd.projects.set_current('GSA')
ei38_name = "ecoinvent"
exclude_databases = [
    'heia',
    'Agribalyse 1.2',
    'Agribalyse 1.3 - {}'.format(ei38_name),
    'exiobase 2.2',
]
co = ConsumptionDbImporter(
    habe_directory,
    exclude_databases=exclude_databases,
    replace_agribalyse_with_ecoinvent=True,
)
co.match_database()
co.match_database(db_name='biosphere3', fields=('name', 'category', 'unit', 'location'))
co.match_database(db_name=ei38_name, fields=('name', 'unit', 'reference product', 'location'))
co.apply_strategies()
co.statistics()

print()

