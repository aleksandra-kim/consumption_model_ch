import brightway2 as bw
import pandas as pd
import numpy as np
from copy import copy
import os, json
from pathlib import Path
import re
import string
# import bw2io

# Local files
from utils_allocation import *
from utils_consumption_db import *

# ################
# ## Ecoinvent ###
# ################

def import_ecoinvent(ei_path, ei_name):
    if ei_name in bw.databases:
        print(ei_name + " database already present!!! No import is needed")
    else:
        ei = bw.SingleOutputEcospold2Importer(ei_path, ei_name)
        ei.apply_strategies()
        ei.match_database(db_name='biosphere3',fields=('name', 'category', 'unit', 'location'))
        ei.statistics()
        ei.write_database()

def create_ecoinvent_33_project(ei33_path, ei33_name="ecoinvent 3.3 cutoff" ):
    current_project = deepcopy(bw.projects.current)
    bw.projects.set_current(ei33_name) # Temporarily switch to ecoinvent 3.3 project
    bw.bw2setup()
    import_ecoinvent(ei33_path, ei33_name)
    bw.projects.set_current(current_project) # Switch back



# ###############
# ## Exiobase ###
# ###############

def import_exiobase_22(ex22_path, ex22_name='EXIOBASE 2.2'):
    from bw2io.importers.exiobase2 import Exiobase22Importer
    if ex22_name in bw.databases:
        print(ex22_name + " database already present!!! No import is needed")
    else:
        ex22 = Exiobase22Importer(ex22_path, ex22_name)
        ex22.apply_strategies()
        ex22.write_database()
        
def import_exiobase_3(ex3_path, ex3_name):
    from bw2io.importers.exiobase3 import Exiobase3Importer
    if ex3_name in bw.databases:
        print("{} database already present!!! No import is needed".format(ex3_name))
    else:
        ex = Exiobase3Importer(ex3_path, ex3_name) # give path to IOT_year_pxp folder
        ex.apply_strategies()
        ex.write_database()


# #################
# ## Agribalyse ###
# #################

def import_agribalyse12(ag12_path, ei_name, ag12_name='Agribalyse 1.2'):
    '''
    Original import at https://nbviewer.jupyter.org/urls/bitbucket.org/cmutel/brightway2/raw/default/notebooks/IO%20-%20Importing%20Agribalyse%20with%20Ecoinvent%202.2.ipynb
    '''
    ag12_ei_name = ag12_name + ' - ' + ei_name
    
    if ei_name not in bw.databases:
        print('Cannot find database: ' + ei_name)
        return
    
    if ag12_ei_name in bw.databases:
        print(ag12_ei_name + " database already present!!! No import is needed")
    else:
        ag12_ei = bw.SingleOutputEcospold1Importer(ag12_path, ag12_ei_name)
        ag12_ei.apply_strategies()
        
        ag12_ei.write_excel(True)
        from bw2io.strategies.simapro import normalize_simapro_biosphere_categories, normalize_simapro_biosphere_names
        ag12_ei.apply_strategy(normalize_simapro_biosphere_categories)
        ag12_ei.apply_strategy(normalize_simapro_biosphere_names)

        from bw2io.strategies import link_iterable_by_fields
        import functools
        ag12_ei.apply_strategy(functools.partial(link_iterable_by_fields, other=bw.Database("biosphere3"), kind="biosphere"))

        ag12_ei.write_excel(True)
        
        ag12_ei_new_biosphere_name = ag12_ei_name + " - new biosphere"
        bw.Database(ag12_ei_new_biosphere_name).register()
        ag12_ei.add_unlinked_flows_to_biosphere_database(ag12_ei_new_biosphere_name)

        def get_unlinked(data):
            for ds in ag12_ei.data:
                for exc in ds['exchanges']:
                    if exc['type'] == 'production' and exc['name'] == 'Alfalfa, conventional, for animal feeding, at farm gate':
                        return ds, exc

        ds, exc = get_unlinked(ag12_ei.data)

        from bw2io.strategies import link_technosphere_based_on_name_unit_location
        ag12_ei.apply_strategy(link_technosphere_based_on_name_unit_location)

        def get_unlinked(data):
            for ds in ag12_ei.data:
                for exc in ds['exchanges']:
                    if exc['type'] == 'production' and not exc.get('input'):
                        return ds, exc

        ds, exc = get_unlinked(ag12_ei.data)

        for ds in ag12_ei.data:
            for exc in ds['exchanges']:
                if exc['type'] == 'production' and not exc.get('input'):
                    print("Fixing:", exc['name'])
                    exc['type'] = 'technosphere'

        ag12_ei.apply_strategy(link_technosphere_based_on_name_unit_location)
        ei_name_short = ei_name #TODO
        ag12_ei.apply_strategy(functools.partial(link_technosphere_based_on_name_unit_location, external_db_name=ei_name_short))
        ag12_ei.add_unlinked_activities()

        for ds in ag12_ei.data:
            del ds['code']

        from bw2io.strategies import set_code_by_activity_hash
        ag12_ei.apply_strategy(set_code_by_activity_hash)

        ag12_ei.apply_strategy(functools.partial(
                link_iterable_by_fields,
                other=ag12_ei.data,
                fields=('name', 'location', 'unit'),
                relink=True
        ))

        ag12_ei.statistics()
        ag12_ei.write_database()


def import_agribalyse_13(ag13_path, ei_name, ag13_name='Agribalyse 1.3'):

    if ei_name not in bw.databases:
        print('Cannot find database: ' + ei_name)
        return

    ag13_ei_name = "{} - {}".format(ag13_name, ei_name)
    if ag13_ei_name in bw.databases:
        print(ag13_ei_name + " database already present!!! No import is needed")
    else:
        ag13_ei = bw.SimaProCSVImporter(ag13_path, ag13_ei_name)
        ag13_ei.apply_strategies()
        # Apply all migrations with previous versions of ecoinvent
        ag13_ei.migrate('simapro-ecoinvent-3.3')
        # Update US locations
        from bw2io.strategies.locations import update_ecoinvent_locations
        ag13_ei = update_ecoinvent_locations(ag13_ei)
        # Biosphere flows
        ag13_ei_new_biosphere_name = ag13_ei_name + " - new biosphere"
        bw.Database(ag13_ei_new_biosphere_name).register()
        ag13_ei.add_unlinked_flows_to_biosphere_database(ag13_ei_new_biosphere_name)
        # Add unlinked waste flows as new activities
        ag13_ei.add_unlinked_activities()
        ag13_ei.match_database(ei_name, fields=('reference product','location', 'unit', 'name'))

        # 2. Define some of the migrations manually.
        #########
        #- Most of them are minor changes in names of activities and reference products
        #- Some activities contain `multiplier` field if the unit conversion is needed or reference products are not identical
        agribalyse13_ecoinvent_names = json.load(open("data/migrations/agribalyse-1.3.json"))
        bw.Migration("agribalyse13-ecoinvent-names").write(
            agribalyse13_ecoinvent_names,
            description="Change names of some activities"
        )
        ag13_ei.migrate('agribalyse13-ecoinvent-names')
        ag13_ei.match_database(ei_name, fields=('reference product','location', 'unit', 'name'))

        ### 4. Allocate by production volume
        #########
        ei36 = bw.Database(ei_name)
        # Create mapping between unlinked exchanges and ecoinvent activities that constitute each unlinked exchange.
        # In this case we do not need to do it manually since all unlinked exchanges need to be allocated geographically.
        # Example: (market for lime, GLO) is split by production volume into (market for lime, RoW) & (market for lime, RER).
        # Mapping is a list of dictionaries where each dictionary corresponds to an unlinked exchange.
        # The key is the name of the unlinked exchange and the values are ecoinvent activities codes.

        def create_location_mapping(ag13_ei, ei_name):

            ei = bw.Database(ei_name)

            unlinked_list = list(ag13_ei.unlinked)
            len_unlinked  = len(unlinked_list)

            mapping = [0]*len_unlinked

            for u in range(len_unlinked):
                new_el = {}
                name = unlinked_list[u]['name']
                loc  = unlinked_list[u]['location']
                acts_codes = [act['code'] for act in ei if name == act['name']]
                new_el[ (name, loc) ] = acts_codes
                mapping[u] = new_el

            return mapping

        mapping = create_location_mapping(ag13_ei, ei_name)
        agg = modify_exchanges(ag13_ei, mapping, ei_name)

        ### 5. Change uncertainty info
        #########
        import stats_arrays as sa
        changed = []
        for i,act in enumerate(agg.data):
            excs = act.get('exchanges', [])
            for j,exc in enumerate(excs):
                if exc.get('uncertainty type', False) == sa.LognormalUncertainty.id and \
                np.allclose(exc.get('amount'), exc.get('loc')):
                    # a. loc chosen such that amount is the mean of the specified distribution
        #             exc.update(loc=np.log(exc['amount'])-(exc['scale']**2)/2)
                    # b. loc chosen such that amount is the median of the specified distribution, same as majority in ecoinvent
                    exc.update(loc=np.log(exc['amount']))
                    changed.append([i,j])
        print(len(changed))
        if "3.6" in ei_name:
            assert len(changed)==319
        elif "3.7" in ei_name:
            if "3.7.1" not in ei_name:
                assert len(changed)==1168
            else:
                assert len(changed)==1013

        ### 6. Make sure scale of lognormal is nonzero
        #########
        changed = []
        for i,act in enumerate(agg.data):
            excs = act.get('exchanges', [])
            for j,exc in enumerate(excs):
                if exc.get('uncertainty type', False) == sa.LognormalUncertainty.id and \
                exc.get('scale')==0:
                    exc.update({"uncertainty type": 0, "loc": np.nan, "scale": np.nan})
                    changed.append([i,j])
        assert len(changed)==6

        # # ### 7. Scale technosphere such that production exchanges are all 1
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

        ### 8. Remove repeating activities
        #########
        # Remove repetitive activities
        agg.data.remove({'categories': ('Materials/fuels',),
                         'name': 'ammonium nitrate phosphate production',
                         'unit': 'kilogram',
                         'comment': '(1,1,5,1,1,na); assimilation MAP',
                         'location': 'RER',
                         'type': 'process',
                         'code': 'de221991cc69f37976042f05f448c94c',
                         'database': ag13_ei_name})
        agg.data.remove({'categories': ('Materials/fuels',),
                         'name': 'diammonium phosphate production',
                         'unit': 'kilogram',
                         'comment': "Mineral fertilizers. Model of transport: 'MT MAP DAP', weight transported in tons = 5,9E-02. Pedigree-Matrix='(3,3,2,1,2,na)'.",
                         'location': 'RER',
                         'type': 'process',
                         'code': '64037e162f6d6d1048470c3a1135f4fb',
                         'database': ag13_ei_name})
        agg.data.remove({'categories': ('Materials/fuels',),
                         'name': 'monoammonium phosphate production',
                         'unit': 'kilogram',
                         'comment': '',
                         'location': 'RER',
                         'type': 'process',
                         'code': '6d61eb45c1d285073770aa839426d97c',
                         'database': ag13_ei_name})

        # ### 9. Write database
        # #########
        agg.statistics()
        if len(list(agg.unlinked)) == 0:
            agg.write_database()
        else:
            print("Some exchanges are still unlinked")
            print(list(agg.unlinked))

def import_consumption_db(
    co_path, 
    habe_path, 
    exclude_dbs=[], 
    co_name=CONSUMPTION_DB_NAME, 
    habe_year='091011',
    ei_name="ecoinvent 3.7.1 cutoff",
    write_dir="write_files",
    replace_agribalyse_with_ei=True,
    ex_path=None,
    sut_path=None,
):
    '''
    - Raw data: Excel sheet can be obtained from https://doi.org/10.1021/acs.est.8b01452
    - Terms `exchange` and `input activity` are used interchangeably
    - Ecoinvent 3.3 is needed since reference product is taken from there. for that we create a new project with only ecoinvent 3.3
    - habe_year can also be 121314 or 151617
    '''

    if co_name in bw.databases:
        print(co_name + " database already present!!! No import is needed")
    else:
        ### 1. Extract data from Andi's excel file
        #########
        # Make sure that 'ecoinvent 3.3 cutoff' project has been created
        if 'ecoinvent 3.3 cutoff' not in bw.projects:
            print(
                'ecoinvent 3.3 cutoff project is needed for the import, please run `create_ecoinvent_33_project(path)`'
            )
            return
        # Create dataframe that will be our consumption database after we add activities and exchanges from the raw file
        df_bw = create_df_bw(CONSUMPTION_DB_NAME)
        # Read data from Andi's consumption database
        sheet_name = 'Overview & LCA-Modeling'
        df_raw = pd.read_excel(co_path, sheet_name=sheet_name, header=2)
        # Read data from HABE
        code_unit = get_units_habe(habe_path, habe_year)
        # Add ON columns
        df = complete_columns(df_raw)
        # Parse Andi's excel file
        act_indices = df.index[df['ConversionDem2FU'].notna()].tolist()  # indices of all activities
        exclude_dbs = [edb.lower() for edb in exclude_dbs]
        print("--> Creating consumption_db.xlsx")
#         for ind in act_indices:
#             # For each row
#             df_ind = df.iloc[ind]
#             df_ind = df_ind[df_ind.notna()]
#             # Add activity
#             df_bw, df_act = append_activity(df_bw, df_ind[:N_ACT_RELEVANT],
#                                             code_unit)  # only pass columns relevant to this function
#             # Add exchanges
#             df_bw = append_exchanges(
#                 df_bw,
#                 df_ind,
#                 df_act,
#                 exclude_dbs=exclude_dbs,
#                 replace_agribalyse_with_ei=replace_agribalyse_with_ei
#             )
#         df_bw.columns = list(string.ascii_uppercase[:len(df_bw.columns)])
#         # Update to relevant databases and save excel file
#         if "3.7.1" in ei_name:
#             use_ecoinvent_371 = True
#         else:
#             use_ecoinvent_371 = False

#         ag_name = 'Agribalyse 1.3 - {}'.format(ei_name)
#         if replace_agribalyse_with_ei:
#             df_agribalyse_ei = pd.read_excel("data/agribalyse_replaced_with_ecoinvent.xlsx")
#             df_bw = df_bw.append(df_agribalyse_ei, ignore_index=True)

        write_dir = Path(write_dir)
        write_dir.mkdir(parents=True, exist_ok=True)
        path_new_db = write_dir / 'consumption_db.xlsx'
#         df_bw = update_all_db(df_bw, use_ecoinvent_371=use_ecoinvent_371)
#         df_bw.to_excel(path_new_db, index=False, header=False)

        ### 2. Link to other databases
        #########
        # We assume that ecoinvent is always included, but exiobase and agribalyse may be excluded

        co = bw.ExcelImporter(path_new_db)
        co.apply_strategies()
        co.match_database(fields=('name', 'unit', 'location', 'categories'))
        co.match_database(ei_name, fields=('name', 'reference product', 'unit', 'location', 'categories'))

        ### 3. Define a migration for particular activities that can only be hardcoded
        #########
        if "3.5" in ei_name or "3.6" in ei_name or "3.7" in ei_name:
            print("Migration for 'steam production in chemical industry' and 'market for green bell pepper'")
            ecoinvent_35_36_37_change_names_data = json.load(open("data/migrations/ecoinvent-3.5-3.6-3.7.1.json"))
            bw.Migration("ecoinvent-35-36-37-change-names").write(
                ecoinvent_35_36_37_change_names_data,
                description="Change names of some activities"
            )
            co.migrate('ecoinvent-35-36-37-change-names')
            co.match_database(ei_name, fields=('name', 'reference product', 'unit', 'location', 'categories'))

        ### 4. Define a migration for rice production and specific locations
        #########
        # These locations have only non-basmati rice production
        if "3.6" in ei_name or "3.7" in ei_name:
            ecoinvent_36_371_rice_nonbasmati = json.load(open("data/migrations/ecoinvent-3.6-3.7.1.json"))
            bw.Migration("ecoinvent-36-371-rice-nonbasmati").write(
                ecoinvent_36_371_rice_nonbasmati,
                description="Change names of some activities"
            )
            co.migrate('ecoinvent-36-371-rice-nonbasmati')
            co.match_database(ei_name, fields=('name', 'reference product', 'unit', 'location', 'categories'))

        ### 5. Manually choose which ecoinvent exchanges should be taken for each unlinked exchange
        #########
        # The rest of the unlinked exchanges are not uniquely defined in ecoinvent 3.6 -> 1-to-multiple mapping.
        # For example 'rice production' is now divided into basmati and non-basmati rice.
        # Hence, we split them based on their shares in the production volumes.
        ei = bw.Database(ei_name)
        mapping = [
            {('market for rice', 'GLO'):
                 [act['code'] for act in ei if 'market for rice' in act['name']
                  and act['location'] == 'GLO'
                  and 'seed' not in act['name']]},

            {('rice production', 'RoW'):
                 [act['code'] for act in ei if 'rice production' in act['name']
                  and act['location'] == 'RoW'
                  and 'straw' not in act['reference product']]},

            {('rice production', 'IN'):
                 [act['code'] for act in ei if 'rice production' in act['name']
                  and act['location'] == 'IN'
                  and 'straw' not in act['reference product']]},

            {('market for wheat grain', 'GLO'):
                 [act['code'] for act in ei if 'market for wheat grain' in act['name']
                  and 'feed' not in act['name']]},

            {('market for maize grain', 'GLO'):
                 [act['code'] for act in ei if 'market for maize grain' in act['name']
                  and 'feed' not in act['name']]},

            {('market for mandarin', 'GLO'):
                 [act['code'] for act in ei if 'market for mandarin' in act['name']]},

            {('market for soybean', 'GLO'):
                 [act['code'] for act in ei if 'market for soybean' in act['name']
                  and all([_ not in act['name'] for _ in ['meal', 'beverage', 'seed', 'feed', 'oil']])]},
        ]

        ag_name = 'Agribalyse 1.3 - {}'.format(ei_name)
        if ag_name not in exclude_dbs and ag_name in bw.databases and not replace_agribalyse_with_ei:
            print("-->Linking to {}".format(ag_name))
            co.match_database(ag_name, fields=('name', 'unit', 'location'))

        ag12_name = 'Agribalyse 1.2 - {}'.format(ei_name)
        if ag12_name not in exclude_dbs and ag12_name in bw.databases:
            print("-->Linking to {}".format(ag12_name))
            co.match_database(ag12_name, fields=('name', 'unit', 'location'))
        co = modify_exchanges(co, mapping, ei_name)

        ### 4. Define migrations for exiobase
        #########
        ex_name = 'exiobase 2.2'
        if ex_name not in exclude_dbs:
            for db_name in bw.databases:
                if "exiobase" in db_name.lower():
                    ex_name = db_name
            print("--> Linking to {}".format(ex_name))
            if "3.8.1" in ex_name:
                print("Migration for {}".format(ex_name))
                # Only based on the `name` field
                exiobase_381_change_names_data = json.load(open("data/migrations/exiobase-3.8.1.json"))
                bw.Migration("exiobase-381-change-names").write(
                    exiobase_381_change_names_data,
                    description="Change names of some exiobase 3.8.1 activities"
                )
                co.migrate('exiobase-381-change-names')
                margins_path = Path(sut_path) / 'CH_2015.xls'
            elif '2.2' in ex_name:
                margins_path = Path(sut_path) / 'CH_2007.xls'
        # return co

            co = link_exiobase(co, ex_name, ex_path, margins_path)
            co.match_database(ex_name, fields=('name', 'unit', 'location',))

        # # Rename RoW locations
        # exiobase_381_change_locations_data = json.load(open("data/migrations/exiobase-row-locations.json"))
        # bw.Migration("exiobase-381-change-locations").write(
        #     exiobase_381_change_locations_data,
        #     description="Change RoW locations in exiobase"
        # )
        # co.migrate('exiobase-381-change-locations')
        co.statistics()
        if len(list(co.unlinked)) == 0:
            co.write_database()
        else:
            print("Some exchanges are still unlinked")
        # Give the required name to the consumption database
        if co_name != CONSUMPTION_DB_NAME:
            co_diff_name = bw.Database(CONSUMPTION_DB_NAME)
            co_diff_name.rename(co_name)

    # Sum up repetitive exchanges
    co = bw.Database(co_name)
    for act in co:
        excs = [exc.input for exc in act.exchanges()]
        rep_inputs = {exc for exc in excs if excs.count(exc) > 1}
        for rep_input in rep_inputs:
            amounts = [exc.amount for exc in act.exchanges() if exc.input == rep_input]
            [exc.delete() for exc in act.exchanges() if exc.input == rep_input]
            act.new_exchange(
                input=rep_input,
                amount=sum(amounts),
                type='technosphere'
            ).save()

            
            
def add_consumption_all_hh(
    co_name, 
    habe_path, 
    habe_year='091011', 
    option='disaggregated',
    write_dir="write_files",
):

    ### 1. Extract total demand from HABE
    #########
    get_path = lambda which_file: os.path.join( habe_path, [f for f in os.listdir(habe_path) if habe_year in f and which_file in f][0] )
    path_beschrei = get_path('Datenbeschreibung')
    path_ausgaben = get_path('Ausgaben')

    # change codes to be consistent with consumption database and Andi's codes
    co = bw.Database(co_name)
    ausgaben = pd.read_csv(path_ausgaben, sep='\t')
    codes_co_db = sorted([act['code'] for act in co])
    columns_a = ausgaben.columns.values
    columns_m = [columns_a[0]]
    for code_a in columns_a[1:]:
        code_m = code_a.replace('A', 'm') 
        if code_m in codes_co_db:
            columns_m.append(code_m)
        else:
            columns_m.append(code_a.lower())
    ausgaben.columns = columns_m

    # Compute total consumption
    total_consumption = ausgaben.sum()
    total_consumption = total_consumption.drop('HaushaltID')

    # Add other useful info, eg number of households and number of people
    meta = pd.read_excel(path_beschrei, sheet_name='Tabellen', skiprows=8, usecols=[0,1,3,4])
    meta.columns = ['category1', 'category2', 'n_rows', 'n_cols']
    meta.dropna(subset=['n_rows'], inplace=True)

    # Combine some columns together
    temp1 = meta[meta['category1'].notnull()][['category1', 'n_rows', 'n_cols']]
    temp1.columns = ['category2', 'n_rows', 'n_cols']
    temp2 = meta[meta['category2'].notnull()][['category2', 'n_rows', 'n_cols']]
    meta = pd.concat([temp1, temp2])
    meta.set_index('category2', inplace=True)

    # Add info
    total_consumption['n_households'] = meta.loc['HABE{}_Ausgaben'.format(habe_year)]['n_rows']
    total_consumption['n_people']     = meta.loc['HABE{}_Personen'.format(habe_year)]['n_rows']
                
    # Save total demand
    write_dir = Path(write_dir)
    path_demand = write_dir / "habe_totaldemands.xlsx"
    total_consumption.to_excel(path_demand)


    ### 2. Options
    #########

    ### OPTION 1 aggregated. Total demands extract directly from HABE raw files
    # Excel file `habe_totaldemands.xlsx` contains sums of all private households in Switzerland for all categories of the HBS. 
    # Units are the same as in the HBS (please refer to the SI-excel of Andi's ES&T-paper in order to translate the codenames). 
    # The attached vector is in "per month" quantities.

    ### OPTION 2 disaggregated. Andi's total demands from his Swiss consumption model
    # Excel file `heia2_totaldemands.xlsx` contains sums of all private households in Switzerland for all categories of the HBS. 
    # Please note that the units are basically the same as in the HBS (please refer to the SI-excel of Andi's ES&T-paper in 
    # order to translate the codenames). However, the attached vector is in "per year" instead of in "per month". Furthermore, 
    # there are a couple of demands that were computed by the model itself. The codenames for these computed/imputed categories 
    # start with "mx" and the units are as follows:
    # - kWh per year for electricity
    # - MJ per year for heating
    # - cubic meters per year for water supply and wastewater collection
    # - number of waste bags per year for refuse collection

    if option == 'aggregated':
        df   = pd.read_excel(path_demand)
        df.columns = ['code', 'amount']
        df.set_index('code', inplace=True)
        n_households = int(df.loc['n_households', 'amount'])
        # n_people     = int(df.loc['n_people', 'amount'])
        df = df.drop(['n_households', 'n_people'])
        df = df.reset_index()
        
    elif option == 'disaggregated':
        path = 'data/habe20092011_hh_prepared_imputed.csv'
        df = pd.read_csv(path, low_memory=False)
        n_households = df.shape[0]
        df = df.drop('haushaltid', axis=1).sum()
        df = df.reset_index()
        df.columns = ['code', 'amount']


    ### 3. Add total inputs from Andi's model as swiss consumption activity
    #########
    co_act_name = 'ch hh all consumption {}'.format(option)
    try: co.get(co_act_name).delete()
    except: pass
    consumption_all = co.new_activity(co_act_name, name=co_act_name, location='CH', unit='1 month of consumption')
    consumption_all.save()
    # Add production exchange for the activity `consumption`
    consumption_all.new_exchange(
        input = (consumption_all['database'], consumption_all['code']),
        amount = 1,
        type = 'production',
    ).save()
    consumption_all['agg_option'] = option
    consumption_all['n_households'] = n_households
    consumption_all.save()
    # Smth with codes
    codes = [act['code'] for act in co]
    unlinked_codes = []
    for i in range(len(df)):
        code = df.loc[i]['code']
        factor = 1
        # if "mx" in code:
        #     factor = 12 # TODO?? divide by number of months
        if code in codes:
            consumption_all.new_exchange(input  = (co.name, code), 
                                         amount = df.loc[i]['amount'] / factor,
                                         type   = 'technosphere').save()
        else:
            unlinked_codes.append(code)

    ### 4. Note that
    #########
    # the number of consumption exchanges is the same as the number of activities in the database,
    # but is a lot less than what Andi provided in his total demands. TODO not sure what this means anymore


def add_consumption_average_hh(consumption_all):
    ### 5. Add consumption activity for an average household
    #########
    co_name = consumption_all.get('database')
    option = consumption_all.get('agg_option')
    n_households = consumption_all.get('n_households')
    co = bw.Database(co_name)
    co_average_act_name = 'ch hh average consumption {}'.format(option)
    try: co.get(co_average_act_name).delete()
    except: pass
    consumption_average = consumption_all.copy(co_average_act_name, name=co_average_act_name)
    for exc in consumption_average.exchanges():
        if exc['type'] != 'production':
            exc['amount'] /= n_households
            exc.save()

def add_consumption_activities(
        co_name,
        habe_path,
        habe_year='091011',
        option='disaggregated',
):
    add_consumption_all_hh(co_name, habe_path, habe_year='091011', option='disaggregated', )
    add_consumption_all_hh(co_name, habe_path, habe_year='091011', option='aggregated', )
    co = bw.Database(co_name)
    demand_act_dis = co.search('consumption disaggregated')[0]
    demand_act_agg = co.search('consumption aggregated')[0]
    dict_dis = {a.input: a.amount for a in demand_act_dis.exchanges() if a['type'] == 'technosphere'}
    dict_agg = {b.input: b.amount for b in demand_act_agg.exchanges() if b['type'] == 'technosphere'}

    add_inputs = list(set(dict_dis.keys()) - set(dict_agg.keys()))
    fix_amounts = {}
    for input_, amount_dis in dict_dis.items():
        amount_agg = dict_agg.get(input_, np.nan)
        if not np.allclose(amount_dis, amount_agg):
            fix_amounts[input_] = amount_dis
    n_households_old = demand_act_dis['n_households']
    demand_act_dis.delete()
    demand_act_agg.delete()

    add_consumption_all_hh(co_name, habe_path, habe_year=habe_year, option=option, )
    demand = co.search('consumption aggregated')
    assert len(demand) == 1
    demand = demand[0]
    n_households_new = demand['n_households']

    for exc in demand.exchanges():
        amount = fix_amounts.get(exc.input, False)
        if amount:
            exc['amount'] = amount
            exc.save()
    for input_ in add_inputs:
        amount = fix_amounts.get(input_) / n_households_old * n_households_new
        demand.new_exchange(
            input=input_,
            amount=amount,
            type='technosphere',
        ).save()

    add_consumption_average_hh(demand)



def add_consumption_categories(co_name, co_path):
    co = bw.Database(co_name)

    sheet_name = 'Overview & LCA-Modeling'
    df_raw = pd.read_excel(co_path, sheet_name = sheet_name, header=2)

    categories_col_de = 'Original name in Swiss household budget survey'
    categories_col_en = 'Translated name'
    categories_raw = df_raw[[categories_col_de, categories_col_en]]

    categories = {}
    for v in categories_raw.values:
        v_list_de = v[0].split(':')
        v_list_en = v[1].split(':')
        if len(v_list_de)>1 and len(v_list_de[0].split('.')) == 1:
            categories[v_list_de[0]] = v_list_en[0]
    max_code_len = max({len(k) for k in categories.keys()})

    category_names_dict = {
        2: 'coarse',
        3: 'middle',
        4: 'fine',
    }
    for act in co:    
        code = re.sub(r'[a-z]+', '', act['code'], re.I)[:max_code_len]

        for i in range(2,max_code_len+1):
            try:
                category_name = 'category_' + category_names_dict[i]
                act[category_name] = categories[code[:i]]
                act.save()
            except:
                pass
        if act['name'] == "Desktop computers" or act['name'] == "Portable computers" or act['name'] == "Printers (incl. multifunctional printers)":
            act["category_coarse"] = "Durable goods"
            act.save()

def add_consumption_sectors(co_name):
    '''
    Add consumption sectors as separate activities in the consumption database
    '''
    co = bw.Database(co_name)
    demand_act = co.search("ch hh average consumption")[0]

    cat_option = 'category_coarse'

    cat_unique = []
    for act in co:
        cat_unique.append(act.get(cat_option) or 0)
    cat_unique = list(set(cat_unique))

    category_activities = {}
    category_activities_len = {}
    for cat_of_interest in cat_unique: 
        list_ = []
        for act in co:
            if act.get(cat_option) == cat_of_interest:
                list_.append(act)
        if len(list_) > 0:
            category_activities[cat_of_interest] = list_
            category_activities_len[cat_of_interest] = len(list_)

    dict_ = {}
    for cat_of_interest, activities in category_activities.items():
            
        excs_input_ag  = []
        excs_input_ex  = []
        excs_input_ec  = []

        for act in activities:
            for exc in act.exchanges():
                if 'Agribalyse' in exc.input['database']:
                    excs_input_ag.append(exc.input)
                elif 'EXIOBASE' in exc.input['database']:
                    excs_input_ex.append(exc.input)
                elif 'ecoinvent' in exc.input['database']:
                    excs_input_ec.append(exc.input)
                    
        dict_[cat_of_interest] = dict(
            n_activities = len(activities),
            n_agribalyse_exchanges = len(excs_input_ag), 
            n_exiobase_exchanges   = len(excs_input_ex), 
            n_ecoinvent_exchanges  = len(excs_input_ec), 
        )
            
    for cat_of_interest in category_activities:
        # Create new bw activity with a specific name
        try: co.get(cat_of_interest).delete()
        except: pass
        new_act = co.new_activity(cat_of_interest,
                                  name="{} sector".format(cat_of_interest),
                                  location='CH',
                                  unit='1 month of consumption',
                                  comment='Average consumption of one household',
                                 )
        new_act.save()

        # Add production exchange
        new_act.new_exchange(
            input = (new_act['database'], new_act['code']),
            amount = 1,
            type = 'production'
        ).save()

        for exc in demand_act.exchanges():
            if exc.input.get('category_coarse')==cat_of_interest:
                new_act.new_exchange(
                    input  = (exc.input['database'], exc.input['code']),
                    amount = exc.amount,
                    type   = 'technosphere'
                ).save()

