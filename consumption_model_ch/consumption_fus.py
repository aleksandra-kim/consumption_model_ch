import pandas as pd
import numpy as np
import bw2data as bd
import bw2calc as bc
from pathlib import Path
import re

from .utils import get_habe_filepath, read_pickle, write_pickle

dirpath = Path(__file__).parent.resolve() / "data"


def add_consumption_categories(co_name):
    co = bd.Database(co_name)

    sheet_name = 'Overview & LCA-Modeling'
    co_path = dirpath / "es8b01452_si_002.xlsx"
    df_raw = pd.read_excel(co_path, sheet_name=sheet_name, header=2)

    categories_col_de = 'Original name in Swiss household budget survey'
    categories_col_en = 'Translated name'
    categories_raw = df_raw[[categories_col_de, categories_col_en]]

    categories = {}
    for v in categories_raw.values:
        v_list_de = v[0].split(':')
        v_list_en = v[1].split(':')
        if len(v_list_de) > 1 and len(v_list_de[0].split('.')) == 1:
            categories[v_list_de[0]] = v_list_en[0]
    max_code_len = max({len(k) for k in categories.keys()})

    category_names_dict = {
        2: 'coarse',
        3: 'middle',
        4: 'fine',
    }
    for act in co:
        code = re.sub(r'[a-z]+', '', act['code'], re.I)[:max_code_len]

        for i in range(2, max_code_len + 1):
            try:
                category_name = 'category_' + category_names_dict[i]
                act[category_name] = categories[code[:i]]
                act.save()
            except:
                pass
        if act['name'] == "Desktop computers" or act['name'] == "Portable computers" or act[
            'name'] == "Printers (incl. multifunctional printers)":
            act["category_coarse"] = "Durable goods"
            act.save()


def add_consumption_sectors(co_name):
    """Add consumption sectors as separate activities in the consumption database."""
    co = bd.Database(co_name)
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

        excs_input_ag = []
        excs_input_ex = []
        excs_input_ec = []

        for act in activities:
            for exc in act.exchanges():
                if 'Agribalyse' in exc.input['database']:
                    excs_input_ag.append(exc.input)
                elif 'EXIOBASE' in exc.input['database']:
                    excs_input_ex.append(exc.input)
                elif 'ecoinvent' in exc.input['database']:
                    excs_input_ec.append(exc.input)

        dict_[cat_of_interest] = dict(
            n_activities=len(activities),
            n_agribalyse_exchanges=len(excs_input_ag),
            n_exiobase_exchanges=len(excs_input_ex),
            n_ecoinvent_exchanges=len(excs_input_ec),
        )

    for cat_of_interest in category_activities:
        # Create new bw activity with a specific name
        try:
            co.get(cat_of_interest).delete()
        except:
            pass
        new_act = co.new_activity(cat_of_interest,
                                  name="{} sector".format(cat_of_interest),
                                  location='CH',
                                  unit='1 month of consumption',
                                  comment='Average consumption of one household',
                                  )
        new_act.save()

        # Add production exchange
        new_act.new_exchange(
            input=(new_act['database'], new_act['code']),
            amount=1,
            type='production'
        ).save()

        for exc in demand_act.exchanges():
            if exc.input.get('category_coarse') == cat_of_interest:
                new_act.new_exchange(
                    input=(exc.input['database'], exc.input['code']),
                    amount=exc.amount,
                    type='technosphere'
                ).save()


def add_consumption_all_hh(
        co_name,
        dir_habe=None,
        option='disaggregated',
        write_dir="write_files",
):
    # 1. Get some metadata from the consumption database
    co = bd.Database(co_name)
    year_habe = co.metadata['year_habe']
    dir_habe = dir_habe or co.metadata['dir_habe']

    # 2. Extract total demand from HABE
    path_beschrei = get_habe_filepath(dir_habe, year_habe, 'Datenbeschreibung')
    path_ausgaben = get_habe_filepath(dir_habe, year_habe, 'Ausgaben')
    path_mengen = get_habe_filepath(dir_habe, year_habe, 'Mengen')

    # change codes to be consistent with consumption database and Andi's codes
    co = bd.Database(co_name)
    ausgaben = pd.read_csv(path_ausgaben, sep='\t')
    mengen = pd.read_csv(path_mengen, sep='\t')
    ausgaben.columns = [col.lower() for col in ausgaben.columns]
    mengen.columns = [col.lower() for col in mengen.columns]
    codes_co_db = sorted([act['code'] for act in co])
    columns_a = ausgaben.columns.values
    columns_m = [columns_a[0]]
    for code_a in columns_a[1:]:
        code_m = code_a.replace('a', 'm')
        if code_m in codes_co_db:
            columns_m.append(code_m)
        else:
            columns_m.append(code_a)
    ausgaben.columns = columns_m

    # Compute total consumption
    total_consumption = ausgaben.sum()
    total_consumption = total_consumption.drop('haushaltid')
    mengen = mengen.sum()
    mengen = mengen.drop('haushaltid')
    for i in range(len(mengen)):
        try:
            total_consumption[mengen.index[i]] = mengen.values[i]
        except KeyError:
            print(mengen.index[i])

    # Add other useful info, eg number of households and number of people
    meta = pd.read_excel(path_beschrei, sheet_name='Tabellen', skiprows=8, usecols=[0, 1, 3, 4])
    meta.columns = ['category1', 'category2', 'n_rows', 'n_cols']
    meta.dropna(subset=['n_rows'], inplace=True)

    # Combine some columns together
    temp1 = meta[meta['category1'].notnull()][['category1', 'n_rows', 'n_cols']]
    temp1.columns = ['category2', 'n_rows', 'n_cols']
    temp2 = meta[meta['category2'].notnull()][['category2', 'n_rows', 'n_cols']]
    meta = pd.concat([temp1, temp2])
    meta.set_index('category2', inplace=True)

    # Add info
    total_consumption['n_households'] = meta.loc['HABE{}_Ausgaben'.format(year_habe)]['n_rows']
    total_consumption['n_people'] = meta.loc['HABE{}_Personen'.format(year_habe)]['n_rows']

    # Save total demand
    write_dir = Path(write_dir)
    path_demand = write_dir / "habe_totaldemands.xlsx"
    total_consumption.to_excel(path_demand)

    # 3. Options

    # OPTION 1 aggregated. Total demands extract directly from HABE raw files
    # Excel file `habe_totaldemands.xlsx` contains sums of all private households in Switzerland for all categories of
    # the HBS. Units are the same as in the HBS (please refer to the SI-excel of Andi's ES&T-paper in order to translate
    # the codenames). The attached vector is in "per month" quantities.

    # OPTION 2 disaggregated. Andi's total demands from his Swiss consumption model
    # Excel file `heia2_totaldemands.xlsx` contains sums of all private households in Switzerland for all categories of
    # the HBS. Please note that the units are basically the same as in the HBS (please refer to the SI-excel of Andi's
    # ES&T-paper in order to translate the codenames). However, the attached vector is in "per year" instead of in
    # "per month". Furthermore, there are a couple of demands that were computed by the model itself. The codenames for
    # these computed/imputed categories start with "mx" and the units are as follows:
    # - kWh per year for electricity
    # - MJ per year for heating
    # - cubic meters per year for water supply and wastewater collection
    # - number of waste bags per year for refuse collection

    if option == 'aggregated':
        df = pd.read_excel(path_demand)
        df.columns = ['code', 'amount']
        df.set_index('code', inplace=True)
        n_households = int(df.loc['n_households', 'amount'])
        # n_people     = int(df.loc['n_people', 'amount'])
        df = df.drop(['n_households', 'n_people'])
        df = df.reset_index()

    elif option == 'disaggregated':
        path = dirpath / "functional_units" / 'habe20092011_hh_prepared_imputed.csv'
        df = pd.read_csv(path, low_memory=False)
        n_households = df.shape[0]
        df = df.drop('haushaltid', axis=1).sum()
        df = df.reset_index()
        df.columns = ['code', 'amount']

    else:
        n_households = None

    # 4. Add total inputs from Andi's model as swiss consumption activity
    co_act_name = 'ch hh all consumption {}'.format(option)
    try:
        co.get(co_act_name).delete()
    except:
        pass
    consumption_all = co.new_activity(co_act_name, name=co_act_name, location='CH', unit='1 month of consumption')
    consumption_all.save()
    # Add production exchange for the activity `consumption`
    consumption_all.new_exchange(
        input=(consumption_all['database'], consumption_all['code']),
        amount=1,
        type='production',
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
            consumption_all.new_exchange(input=(co.name, code),
                                         amount=df.loc[i]['amount'] / factor,
                                         type='technosphere').save()
        else:
            unlinked_codes.append(code)

    # Note that
    # - the number of consumption exchanges is the same as the number of activities in the database,
    # - but is a lot less than what Andi provided in his total demands. TODO not sure what this means anymore


def add_consumption_average_hh(consumption_all):
    """Add consumption activity for an average household."""
    co_name = consumption_all.get('database')
    option = consumption_all.get('agg_option')
    n_households = consumption_all.get('n_households')
    co = bd.Database(co_name)
    co_average_act_name = 'ch hh average consumption {}'.format(option)
    try:
        co.get(co_average_act_name).delete()
    except:
        pass
    consumption_average = consumption_all.copy(co_average_act_name, name=co_average_act_name)
    for exc in consumption_average.exchanges():
        if exc['type'] != 'production':
            exc['amount'] /= n_households
            exc.save()


def add_consumption_activities(
        co_name,
        dir_habe=None,
        option='disaggregated',
):
    # Delete all existing consumption activities
    co = bd.Database(co_name)
    [act.delete() for act in co.search("consumption disaggregated")]
    [act.delete() for act in co.search("consumption aggregated")]

    # Add new consumption activities
    dir_habe = dir_habe or co.metadata['dir_habe']
    add_consumption_all_hh(co_name, dir_habe, option='disaggregated', )
    add_consumption_all_hh(co_name, dir_habe, option='aggregated', )

    demand_act_dis = co.search('consumption disaggregated')[0]
    demand_act_agg = co.search('consumption aggregated')[0]
    dict_dis = {a.input: a.amount for a in demand_act_dis.exchanges() if a['type'] == 'technosphere'}
    dict_agg = {b.input: b.amount for b in demand_act_agg.exchanges() if b['type'] == 'technosphere'}

    demand_act_dis_dict = {k['name']: v for k, v in dict_dis.items()}
    demand_act_agg_dict = {k['name']: v for k, v in dict_agg.items()}

    add_inputs = list(set(dict_dis.keys()) - set(dict_agg.keys()))
    fix_amounts = {}
    for input_, amount_dis in dict_dis.items():
        amount_agg = dict_agg.get(input_, np.nan)
        if not np.allclose(amount_dis, amount_agg):
            fix_amounts[input_] = amount_dis
    n_households_old = demand_act_dis['n_households']
    demand_act_dis.delete()
    demand_act_agg.delete()

    add_consumption_all_hh(co_name, dir_habe, option=option, )
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

    write_dir = Path("write_files")
    path_demand_comparison = write_dir / "comparison_total_demands.xlsx"
    demand_new_dict = {c.input['name']: c.amount for c in demand.exchanges() if c['type'] == 'technosphere'}
    dis_agg_ratio = {k: demand_act_agg_dict.get(k, 0) / v for k, v in demand_act_dis_dict.items()}
    dis_new_ratio = {k: demand_new_dict.get(k, 0) / v for k, v in demand_act_dis_dict.items()}

    year_habe = co.metadata['year_habe']
    df = pd.DataFrame.from_dict(
        {
            'Froemelt': demand_act_dis_dict,
            'HABE 091011': demand_act_agg_dict,
            'HABE 091011 / Froemelt': dis_agg_ratio,
            'HABE {}'.format(year_habe): demand_new_dict,
            'HABE {} / Froemelt'.format(year_habe): dis_new_ratio,
        }
    )
    df.to_excel(path_demand_comparison)


def add_archetypes_consumption(co_name, archetypes_path=None):
    print("Creating archetypes functional units")
    if archetypes_path is None:
        archetypes_path = dirpath / "functional_units" / "hh_archetypes_weighted_working_tables.csv"
    co = bd.Database(co_name)
    df = pd.read_csv(archetypes_path)
    all_consumption_codes = [act['code'] for act in co]
    codes_to_ignore = [code for code in df.iloc[0].index if code not in all_consumption_codes]
    # print(codes_to_ignore)
    for i, df_row in df.iterrows():
        label = df_row['cluster_label_name']
        # Create new activity
        act_name = "archetype_{}_consumption".format(label)
        try:
            co.get(act_name).delete()
        except:
            pass
        archetype_act = co.new_activity(
            act_name,
            name=act_name,
            location='CH',
            unit='1 month of consumption',
            cluster_label_def=df_row['cluster_label_def']
        )
        archetype_act.save()
        # Add exchanges to this activity
        for code in df_row.index:
            if ("cluster" not in code) and (code not in codes_to_ignore):
                archetype_act.new_exchange(
                    input=(co.name, code),
                    amount=df_row[code],
                    type='technosphere'
                ).save()
    return


def get_archetypes_scores_per_month(co_name, method, fp_archetypes_scores):
    """Get total LCIA scores for all archetypes for one month of consumption."""
    co = bd.Database(co_name)
    archetypes = sorted([act for act in co if "archetype" in act['name'].lower()])
    if fp_archetypes_scores.exists():
        archetypes_scores = read_pickle(fp_archetypes_scores)
    else:
        archetypes_scores = {}
        for demand_act in archetypes:
            lca = bc.LCA({demand_act: 1}, method)
            lca.lci()
            lca.lcia()
            archetypes_scores[demand_act['name']] = lca.score
        write_pickle(archetypes_scores, fp_archetypes_scores)
    return archetypes_scores


def get_archetypes_scores_per_sector(co_name, method, write_dir):
    """Get total LCIA scores for all archetypes for one year of consumption split over sectors."""

    co = bd.Database(co_name)
    # archetypes = sorted([act for act in co if "archetype" in act['name'].lower()])
    a1 = [act for act in co if "archetype_z" in act['name'].lower()]
    a2 = [act for act in co if "archetype_ob" in act['name'].lower()]
    archetypes = a1 + a2
    sectors = sorted([act for act in co if "sector" in act['name'].lower()])

    scores = {}
    for archetype in archetypes:
        print("--> {}".format(archetype['name']))
        fp_archetype_scores = write_dir / "monthly_{}.pickle".format(archetype['name'])
        if fp_archetype_scores.exists():
            scores_per_sector = read_pickle(fp_archetype_scores)
        else:
            scores_per_sector = {}
            for sector in sectors:
                demand_sector = get_demand_per_sector(archetype, sector)
                lca = bc.LCA(demand_sector, method)
                lca.lci()
                lca.lcia()
                # print("{:8.3f}  {}".format(lca.score, sector['name']))
                scores_per_sector[sector['name']] = lca.score
            write_pickle(scores_per_sector, fp_archetype_scores)
        scores[archetype['name']] = scores_per_sector
        # print("\n")
    return scores


def get_demand_per_sector(act, sector):
    sector_name = sector['name'].replace(" sector", "")
    demands = {}
    for exc in act.exchanges():
        if exc.input['category_coarse'] == sector_name:
            demands[exc.input] = exc.amount
    assert len(demands) == len(list(sector.exchanges())) - 1
    return demands


# def add_archetypes_consumption(co_name, archetypes_path, ):
#     co = bd.Database(co_name)
#     df = pd.read_csv(archetypes_path)
#     all_consumption_codes = [act['code'] for act in co]
#     codes_to_ignore = [code for code in df.iloc[0].index if code not in all_consumption_codes]
#     codes_to_modify = {
#         code: "a{}".format(code[2:]) for code in codes_to_ignore
#         if code[:2] == 'mx' and "a{}".format(code[2:]) in all_consumption_codes
#     }
#     print(codes_to_modify)
#     for i, df_row in df.iterrows():
#         label = df_row['cluster_label_name']
#         print("Creating consumption activity of archetype {}".format(label))
#         # Create new activity
#         act_name = "archetype_{}_consumption".format(label)
#         try:
#             co.get(act_name).delete()
#         except:
#             pass
#         archetype_act = co.new_activity(
#             act_name,
#             name=act_name,
#             location='CH',
#             unit='1 month of consumption',
#             cluster_label_def=df_row['cluster_label_def']
#         )
#         archetype_act.save()
#         # Add exchanges to this activity
#
#         for code in df_row.index:
#             if ("cg" not in code) and ("cluster" not in code) and (code not in codes_to_ignore):
#                 archetype_act.new_exchange(
#                     input=(co.name, code),
#                     amount=df_row[code],
#                     type='technosphere'
#                 ).save()
#         for code in codes_to_modify.keys():
#             archetype_act.new_exchange(
#                 input=(co.name, codes_to_modify[code]),
#                 amount=df_row[code],
#                 type='technosphere'
#             ).save()
#
#     return
