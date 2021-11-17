import bw2data as bd
import bw2io as bi

import pandas as pd
import numpy as np
from copy import deepcopy
import os, json
from pathlib import Path
import re
import string

from .importers import ConsumptionDbImporter

# Local files
from .utils_allocation import modify_exchanges
from .utils_consumption_db import (
    CONSUMPTION_DB_NAME,
    N_ACT_RELEVANT,
    create_df_bw,
    get_units_habe,
    complete_columns,
    append_activity,
    append_exchanges,
    update_all_db,
    link_exiobase,
)

DATADIR = Path(__file__).parent.resolve() / "data"


# ################
# ## Ecoinvent ###
# ################

def import_ecoinvent(ei_path, ei_name):
    if ei_name in bd.databases:
        print(ei_name + " database already present!!! No import is needed")
    else:
        ei = bi.SingleOutputEcospold2Importer(ei_path, ei_name)
        ei.apply_strategies()
        ei.match_database(db_name='biosphere3', fields=('name', 'category', 'unit', 'location'))
        ei.statistics()
        ei.write_database()

def create_ecoinvent_33_project(ei33_path, ei33_name="ecoinvent 3.3 cutoff"):
    current_project = deepcopy(bd.projects.current)
    bd.projects.set_current(ei33_name)  # Temporarily switch to ecoinvent 3.3 project
    bi.bw2setup()
    import_ecoinvent(ei33_path, ei33_name)
    bd.projects.set_current(current_project)  # Switch back

# #################
# ## Exiobase 3 ###
# #################
        
def import_exiobase_3(ex3_path, ex3_name):
    if ex3_name in bd.databases:
        print("{} database already present!!! No import is needed".format(ex3_name))
    else:
        ex = bi.Exiobase3MonetaryImporter(ex3_path, ex3_name)  # give path to IOT_year_pxp folder
        ex.apply_strategies()
        ex.write_database()


# #################
# ## Agribalyse ###
# #################

def import_agribalyse_13(ag13_path, ei_name, ag13_name='Agribalyse 1.3'):

    if ei_name not in bd.databases:
        print('Cannot find database: ' + ei_name)
        return

    ag13_ei_name = "{} - {}".format(ag13_name, ei_name)
    if ag13_ei_name in bd.databases:
        print(ag13_ei_name + " database already present!!! No import is needed")
    else:
        # 1. Apply standard BW migrations and strategies
        ag13_ei = bi.SimaProCSVImporter(ag13_path, ag13_ei_name)
        ag13_ei.apply_strategies()
        # Apply all migrations with previous versions of ecoinvent
        ag13_ei.migrate('simapro-ecoinvent-3.3')
        # Update US locations
        from bw2io.strategies.locations import update_ecoinvent_locations
        ag13_ei = update_ecoinvent_locations(ag13_ei)
        # Biosphere flows
        ag13_ei_new_biosphere_name = "{} - new biosphere".format(ag13_ei_name)
        bd.Database(ag13_ei_new_biosphere_name).register()
        ag13_ei.add_unlinked_flows_to_biosphere_database(ag13_ei_new_biosphere_name)
        # Add unlinked waste flows as new activities
        ag13_ei.add_unlinked_activities()
        ag13_ei.match_database(ei_name, fields=('reference product', 'location', 'unit', 'name'))

        # 2. Define some of the migrations manually.
        # - Most of them are minor changes in names of activities and reference products
        # - Activities with `multiplier` field address unit conversions and differences in reference products
        agribalyse13_ecoinvent_names = json.load(open(DATADIR / "migrations" / "agribalyse-1.3.json"))
        bi.Migration("agribalyse13-ecoinvent-names").write(
            agribalyse13_ecoinvent_names,
            description="Change names of some activities"
        )
        ag13_ei.migrate('agribalyse13-ecoinvent-names')
        ag13_ei.match_database(ei_name, fields=('reference product','location', 'unit', 'name'))

        # 3. Allocate by production volume
        # - Create mapping between unlinked exchanges and ecoinvent activities that constitute each unlinked exchange.
        # - No need to link exchanges manually, since allocation is geographic. Example:
        #   (market for lime, GLO) is split by production volume into (market for lime, RoW) & (market for lime, RER).
        # - Mapping is a list of dictionaries, where each dictionary corresponds to an unlinked exchange.
        # - The key is the name of the unlinked exchange and the values are ecoinvent activities codes.

        def create_location_mapping(agribalyse_13_db, ecoinvent_name):

            ecoinvent_db = bd.Database(ecoinvent_name)

            unlinked_list = list(agribalyse_13_db.unlinked)
            len_unlinked = len(unlinked_list)

            mapping_ = [0]*len_unlinked
            for u in range(len_unlinked):
                new_el = {}
                name = unlinked_list[u]['name']
                loc = unlinked_list[u]['location']
                acts_codes = [act['code'] for act in ecoinvent_db if name == act['name']]
                new_el[(name, loc)] = acts_codes
                mapping_[u] = new_el

            return mapping_

        mapping = create_location_mapping(ag13_ei, ei_name)
        agg = modify_exchanges(ag13_ei, mapping, ei_name)

        # 4. Change uncertainty info
        import stats_arrays as sa
        changed = []
        for i, act in enumerate(agg.data):
            excs = act.get('exchanges', [])
            for j, exc in enumerate(excs):
                if exc.get('uncertainty type', False) == sa.LognormalUncertainty.id and \
                        np.allclose(exc.get('amount'), exc.get('loc')):
                    # Option A. loc is chosen s.t. amount is specified distribution's mean
                    # exc.update(loc=np.log(exc['amount'])-(exc['scale']**2)/2)
                    # Option B. loc is chosen s.t. amount is specified distribution's median (consistent with ecoinvent)
                    exc.update(loc=np.log(exc['amount']))
                    changed.append([i, j])
        if "3.6" in ei_name:
            assert len(changed) == 319
        elif "3.7" in ei_name:
            if "3.7.1" not in ei_name:
                assert len(changed) == 1168
            else:
                assert len(changed) == 1013

        # 5. Make sure scale of lognormal is nonzero
        changed = []
        for i, act in enumerate(agg.data):
            excs = act.get('exchanges', [])
            for j, exc in enumerate(excs):
                if exc.get('uncertainty type', False) == sa.LognormalUncertainty.id and exc.get('scale') == 0:
                    exc.update({"uncertainty type": 0, "loc": np.nan, "scale": np.nan})
                    changed.append([i, j])
        assert len(changed) == 6

        # # # 6. Scale technosphere such that production exchanges are all 1
        # # # Commented out because incorrect, need to scale uncertainties in the exchanges as well then!
        # # acts_to_scale = []
        # # for act in agg.data:
        # #     excs = act.get('exchanges', [])
        # #     for exc in excs:
        # #         if exc.get('type') == 'production' and exc.get('amount')!=1:
        # #             acts_to_scale.append((act,exc.get('amount')))
        # #
        # # for act,production_amt in acts_to_scale:
        # #     excs = act.get('exchanges', [])
        # #     for exc in excs:
        # #         if exc.get('type') == 'production':
        # #             exc.update(amount=1)
        # #         else:
        # #             current_amt = exc.get('amount')
        # #             exc.update(amount=current_amt/production_amt)

        # 7. Remove repeating activities
        # Remove repetitive activities
        agg.data.remove(
            {
                'categories': ('Materials/fuels',),
                 'name': 'ammonium nitrate phosphate production',
                 'unit': 'kilogram',
                 'comment': '(1,1,5,1,1,na); assimilation MAP',
                 'location': 'RER',
                 'type': 'process',
                 'code': 'de221991cc69f37976042f05f448c94c',
                 'database': ag13_ei_name
            }
        )
        agg.data.remove(
            {
                'categories': ('Materials/fuels',),
                 'name': 'diammonium phosphate production',
                 'unit': 'kilogram',
                 'comment': "Mineral fertilizers. Model of transport: 'MT MAP DAP', weight transported in tons = 5,9E-02. Pedigree-Matrix='(3,3,2,1,2,na)'.",
                 'location': 'RER',
                 'type': 'process',
                 'code': '64037e162f6d6d1048470c3a1135f4fb',
                 'database': ag13_ei_name
            }
        )
        agg.data.remove(
            {
                'categories': ('Materials/fuels',),
                 'name': 'monoammonium phosphate production',
                 'unit': 'kilogram',
                 'comment': '',
                 'location': 'RER',
                 'type': 'process',
                 'code': '6d61eb45c1d285073770aa839426d97c',
                 'database': ag13_ei_name
            }
        )

        # 8. Write database
        agg.statistics()
        if len(list(agg.unlinked)) == 0:
            agg.write_database()
        else:
            print("Some exchanges are still unlinked")
            print(list(agg.unlinked))


def import_consumption_db(habe_path, co_name, exclude_dbs=(),):
    if co_name in bd.databases:
        print(co_name + " database already present!!! No import is needed")
    else:
        co = ConsumptionDbImporter(habe_path, co_name, exclude_dbs)
        co.write_database()