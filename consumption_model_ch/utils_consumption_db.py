# -*- coding: utf-8 -*-
import bw2data as bd

import pandas as pd
import numpy as np
from copy import copy, deepcopy
import os, json
import re
import country_converter as coco
import warnings
from pathlib import Path

###########################
# ## 1. Define constants ###
# ##########################

# Database name
CONSUMPTION_DB_NAME = 'CH consumption 1.0'


###################################################################
# ## 2.Convert data to brightway database format -> all functions ##
# ##################################################################

# def append_ecoinvent_exchange(df, df_ind_j, ConversionDem2FU):
#     '''
#     Extract information about one input activity, eg name, unit, location, etc and append it to the dataframe df.
#     '''
#     # Extract the activity number
#     k = int(''.join(c for c in df_ind_j.index[0] if c.isdigit()))
#     # Extract information about activity and save it
#     input_act_str = df_ind_j['DB Act ' + str(k)]
#     input_act_db_code = df_ind_j['Activity ' + str(k)]

#     # Find this input activity in brightway databases
#     db_name = input_act_db_code.split("'")[1]

#     # Only include ecoinvent exchanges
#     dbs_no_list = ['agribalyse', 'exiobase', 'heia']
#     for db_no in dbs_no_list:
#         if db_no in db_name.lower():
#             return df

#     code = input_act_db_code.split("'")[3]
#     input_act_db_code_tuple = (db_name, code)

#     # Compute amount
#     input_act_amount = df_ind_j['On ' + str(k)] \
#                      * df_ind_j['Amount Act ' + str(k)] \
#                      * df_ind_j['CFL Act ' + str(k)] \
#                      * ConversionDem2FU

#     try:
#         # Find activity in the ecoinvent 3.3 cutoff using bw functionality
#         current_project = deepcopy(bd.projects.current)
#         bd.projects.set_current('ecoinvent 3.3 cutoff') # Temporarily switch to ecoinvent 3.3 project
#         act_bw = bd.get_activity(input_act_db_code_tuple)
#         bd.projects.set_current(current_project)
#         input_act_values_dict = create_input_act_dict(act_bw, input_act_amount)


#     except:
#         # If bd.get_activity does not work for whichever reason, fill info manually
#         input_act_values_dict = bw_get_activity_info_manually(input_act_str, db_name, input_act_amount)

#     # Add exchange to the dataframe with database in brightway format
#     df = append_exchanges_in_correct_columns(df, input_act_values_dict)

#     return df





# def update_exiobase_amounts(ex_name, exiobase_path):
#     # The following code is needed to update amounts of exiobase exchanges, see Froemelt thesis, Appendix D.4, p.242:
#     # "the shares of EXIOBASE sectors were determined based on the Swiss final demand vector provided by EXIOBASE"
#     # This update is needed in case new regions appear in future exiobase versions,
#     # and to be consistent with the most updated exiobase Swiss final demand.
#
#     # 1. Find activities in the consumption database that have exchanges from all regionalized exiobase sectors
#     co = bd.Database(CONSUMPTION_DB_NAME)
#     acts_to_modify = {}
#     exiobase_set = set()
#     for act in co:
#         exiobase_excs = np.array([exc.input['name'] for exc in act.exchanges() if exc.input['database'] == ex_name])
#         excs_sectors_shares_to_modify = []
#         excs_conversion_only_to_modify = []
#         for exc in list(set(exiobase_excs)):
#             if sum(exiobase_excs==exc)>3:
#                 exiobase_set.add(exc)
#                 excs_sectors_shares_to_modify.append(exc)
#             else:
#                 exiobase_bw_excs = [exchange for exchange in act.exchanges() if exchange.input['database'] == ex_name and
#                                     exc == exchange.input['name'] and exchange['type']=='technosphere']
#                 excs_conversion_only_to_modify += exiobase_bw_excs
#         if len(exiobase_excs) > 0:
#             acts_to_modify[act] = {
#                 "sectors_shares": excs_sectors_shares_to_modify,
#                 "conversion_only": excs_conversion_only_to_modify,
#             }
#     # Save exiobase sectors that are used in the consumption database, where we use shares from the CH final demand,
#     # so that we can compare these shares between exiobase 2 and 3
#     filepath_exiobase_set = 'write_files/exiobase_in_consumption_db.pickle'
#     exiobase_list = sorted(list(exiobase_set))
#     with open(filepath_exiobase_set, 'wb') as f:
#         pickle.dump(exiobase_list, f)
#
#     # 2. Find shares of the sectors based on the Exiobase household consumption
#     if '2.2' in ex_name:
#         filename = "mrFinalDemand_version2.2.2.txt"
#         columns = ['Unnamed: 0', 'Unnamed: 1', 'CH']
#     elif '3.8.1' in ex_name:
#         filename = "Y.txt"
#         columns = ['region', 'Unnamed: 1', 'CH']
#     filepath = Path(exiobase_path) / filename
#     df = pd.read_table(filepath)
#     df = df[columns]
#     df.columns = ['location', 'name', 'hh_consumption']
#     df = df.dropna()
#     df.index = np.arange(len(df))
#
#     # 3. Modify names of the sectors to be consistent with brightway
#     names_dict = {}
#     for name in set(df['name']):
#         start = name.find('(')
#         end = name.find(')')
#         if start!=end and name[start+1:end].isnumeric():
#             names_dict[name] = name[:start-1]
#     for name, name_no_number in names_dict.items():
#         mask = df['name']==name
#         df['name'][mask] = name_no_number
#
#     # 4. Replace old exiobase exchanges with new ones
#     chf_to_euro_2007 = 0.594290
#     chf_to_euro_2015 = 0.937234
#     ex = bd.Database(ex_name)
#     l = len(acts_to_modify)
#     iact = 0
#     for act_to_modify, excs_all in acts_to_modify.items():
#         print("{:3d}/{:3d} {}".format(iact, l, act_to_modify['name']))
#         original_ConversionDem2FU = act_to_modify['original_ConversionDem2FU']
#         ConversionDem2FU = original_ConversionDem2FU/chf_to_euro_2007*chf_to_euro_2015
#         excs_sectors_shares = excs_all["sectors_shares"]
#         # 4.1 modify sectors shares
#         for exc in excs_sectors_shares:
#             # Delete old exchanges
#             [bw_exc.delete() for bw_exc in act_to_modify.exchanges() if bw_exc.input['name'] == exc
#              and bw_exc['type']=='technosphere']
#             # Add new exchanges
#             sector = df[df['name'] == exc].copy()
#             amounts = np.array([float(val) for val in sector['hh_consumption'].values])
#             if sum(amounts) != 0:
#                 amounts /= sum(amounts)
#                 sector['amount'] = amounts
#                 for _,row in sector.iterrows():
#                     input_ = [act for act in ex if act['name']==row['name'] and act['location']==row['location']]
#                     amount = row['amount']*ConversionDem2FU
#                     if len(input_)==1:
#                         act_to_modify.new_exchange(
#                             input=input_[0],
#                             amount=amount,
#                             type="technosphere"
#                         ).save()
#                     else:
#                         print(input_)
#         excs_conversion_only = excs_all["conversion_only"]
#         amounts = {exchange.input: 0 for exchange in excs_conversion_only}
#         for exc in excs_conversion_only:
#             amounts[exc.input] += deepcopy(exc.amount)
#             # Delete old exchange
#             exc.delete()
#         for input_,amount in amounts.items():
#             # Add exchange with updated amount
#             act_to_modify.new_exchange(
#                 input=input_,
#                 amount=amount/chf_to_euro_2007*chf_to_euro_2015,
#                 type="technosphere"
#             ).save()
#         iact += 1
#

def get_margins_df(filepath, margins_label):
    dataframe = pd.read_excel(filepath, sheet_name=margins_label, skiprows=11)
    columns = ['Unnamed: 2', 'Final consumption expenditure by households']
    dataframe = dataframe[columns]
    dataframe.columns = ['name', margins_label]
    dataframe = dataframe.dropna()
    dataframe = dataframe.set_index('name')
    return dataframe


def concat_margins_df(margins_filepath):
    trd_label = 'trade_margins_init'
    tsp_label = 'transport_margins_init'
    tax_label = 'product_taxes_init'
    bpt_label = 'bptot_ini'  # basic price total
    ppt_label = 'purchaser_price'  # purchaser price total
    labels = [trd_label, tsp_label, tax_label, bpt_label, ppt_label]
    df_trd = get_margins_df(margins_filepath, trd_label)
    df_tsp = get_margins_df(margins_filepath, tsp_label)
    df_tax = get_margins_df(margins_filepath, tax_label)
    df_bpt = get_margins_df(margins_filepath, bpt_label)
    df_margins = pd.concat([df_trd, df_tsp, df_tax, df_bpt], axis=1)
    return df_margins, labels


def get_margins_shares(margins_filepath, migrations_exiobase_filepath):
    exiobase_381_change_names_data = json.load(open(migrations_exiobase_filepath))
    exiobase_381_change_names_dict = {}
    for el in exiobase_381_change_names_data['data']:
        old_name = el[0][0]
        new_name = el[1]['name']
        exiobase_381_change_names_dict[old_name] = new_name

    df_margins, labels = concat_margins_df(margins_filepath)
    trd_label, tsp_label, tax_label, bpt_label, ppt_label = labels

    new_index = []
    for old_name in df_margins.index:
        new_name = exiobase_381_change_names_dict.get(old_name, old_name)
        new_name = re.sub(r'\(.[0-9]+\)', '', new_name).rstrip()
        new_index.append(new_name)
    df_margins = df_margins.set_index([new_index])
    df_margins[ppt_label] = df_margins[trd_label] + df_margins[tsp_label] \
                            + df_margins[tax_label] + df_margins[bpt_label]

    df_margins = df_margins[df_margins[ppt_label] != 0]
    df_margins['trade_share'] = df_margins[trd_label] / df_margins[ppt_label]
    df_margins['transport_share'] = df_margins[tsp_label] / df_margins[ppt_label]
    df_margins['tax_share'] = df_margins[tax_label] / df_margins[ppt_label]
    # remove negative shares
    df_margins['trade_share'] = df_margins['trade_share'].apply(lambda x: x if x > 0 else 0)
    df_margins['transport_share'] = df_margins['transport_share'].apply(lambda x: x if x > 0 else 0)
    df_margins['tax_share'] = df_margins['tax_share'].apply(lambda x: x if x > 0 else 0)

    return df_margins


def link_exiobase(co, ex_name, ex_path, margins_filepath, migrations_exiobase_filepath):
    exiobase_dict = {}
    all_exiobase_acts = set()
    exiobase_trade_margin_sectors = set()
    exiobase_transport_margin_sectors = set()
    for i, act in enumerate(co.data):
        flag_margins = False
        exiobase_excs = np.array([exc['name'] for exc in act.get('exchanges', []) if exc['database'] == ex_name])
        exiobase_sectors = []
        all_exiobase_acts = all_exiobase_acts.union(set(exiobase_excs))
        for exc in list(set(exiobase_excs)):
            if sum(exiobase_excs == exc) > 3:
                exiobase_sectors.append(exc)
            else:
                if 'transport' in exc.lower():
                    exiobase_transport_margin_sectors.add(exc)
                else:
                    exiobase_trade_margin_sectors.add(exc)
                flag_margins = True
        if len(exiobase_sectors) + len(exiobase_sectors) > 0:
            exiobase_dict[i] = {
                "exchanges": exiobase_sectors,
                "margins": flag_margins,
                "activity_name": act['name'],
            }

    # 2. Find shares of the sectors based on the Exiobase household consumption
    if '2.2' in ex_name:
        filename = "mrFinalDemand_version2.2.2.txt"
        columns = ['Unnamed: 0', 'Unnamed: 1', 'CH']
        chf_to_euro = 0.594290
    elif '3.8.1' in ex_name:
        filename = "Y.txt"
        columns = ['region', 'Unnamed: 1', 'CH']
        chf_to_euro = 0.937234

    filepath = Path(ex_path) / filename

    df = pd.read_table(filepath)
    df = df[columns]
    df.columns = ['location', 'name', 'hh_consumption']
    df = df.dropna()
    df.index = np.arange(len(df))

    for i, row in df.iterrows():
        new_name = re.sub(r'\(.[0-9]+\)', '', row['name']).rstrip()
        row['name'] = new_name

    df['hh_consumption'] = [float(val) for val in df['hh_consumption'].values]

    exiobase_shares = {}
    for act in all_exiobase_acts:
        df_subset = df[df['name'] == act]
        sum_ = sum(df_subset['hh_consumption'].values)
        location_share = {}
        locations = df_subset['location'].values
        # in case sum is equal to 0, replace it with 1 to keep original 0's for shares
        if sum_ != 0:
            shares = df_subset['hh_consumption'].values / sum_
        else:
            shares = df_subset['hh_consumption'].values
        exiobase_shares[act] = [('SUM_ALL', sum_)] + [x for x in zip(locations, shares)]

    exiobase_transport_margin_sectors_dict = {}
    sum_ = 0
    for act in exiobase_transport_margin_sectors:
        assert exiobase_shares[act][0][0] == 'SUM_ALL'
        value = exiobase_shares[act][0][1]
        exiobase_transport_margin_sectors_dict[act] = value
        sum_ += value
    exiobase_transport_margin_sectors_dict = {k: v / sum_ for k, v in exiobase_transport_margin_sectors_dict.items()}

    exiobase_trade_margin_sectors_dict = {}
    sum_ = 0
    for act in exiobase_trade_margin_sectors:
        assert exiobase_shares[act][0][0] == 'SUM_ALL'
        value = exiobase_shares[act][0][1]
        exiobase_trade_margin_sectors_dict[act] = value
        sum_ += value
    exiobase_trade_margin_sectors_dict = {k: v / sum_ for k, v in exiobase_trade_margin_sectors_dict.items()}

    df_margins = get_margins_shares(margins_filepath, migrations_exiobase_filepath)
    dict_margins = df_margins.T.to_dict()

    mln_to_unit = 1e-6
    chf_to_mln_euro = chf_to_euro * mln_to_unit
    # purchaser_to_basic = 1
    for i, data in exiobase_dict.items():
        # 1. Find production exchange
        production_exchange = [exc for exc in co.data[i]['exchanges'] if exc['type'] == 'production']
        assert len(production_exchange) == 1

        # 2. Compute CFL Act if activity is linked to several exiobase sectors (share of each sector in current act)
        cfl_acts = {}
        for exchange_name in data['exchanges']:
            cfl_act = [share for location, share in exiobase_shares[exchange_name] if location == "SUM_ALL"]
            assert len(cfl_act) == 1
            cfl_acts[exchange_name] = cfl_act[0]
        sum_ = sum(cfl_acts.values())
        for exchange_name, cfl_act in cfl_acts.items():
            if sum_ != 0:
                cfl_acts[exchange_name] = cfl_act / sum_
            else:
                cfl_acts[exchange_name] = 1

        technosphere_exchanges = []
        margins_exchanges = []
        for exchange_name in data['exchanges']:
            try:
                dict_ = dict_margins[exchange_name]
                trd_share = dict_.get('trade_share', 0)
                tns_share = dict_.get('transport_share', 0)
                tax_share = dict_.get('tax_share', 0)
            except KeyError:
                trd_share, tns_share, tax_share = 0, 0, 0
            purchaser_to_basic = 1 - trd_share - tns_share - tax_share
            #         print(exchange_name, purchaser_to_basic)
            # 3. amount_act in Andi's excel is equivalent to share in the Swiss final demand
            technosphere_exchanges += [
                {
                    'name': exchange_name,
                    'location': location,
                    'amount': share * chf_to_mln_euro * purchaser_to_basic * cfl_acts[exchange_name],
                    'unit': 'million €',
                    'database': ex_name,
                    'type': 'technosphere',
                }
                for location, share in exiobase_shares[exchange_name]
                if location != "SUM_ALL"
            ]
            # 4. Add margin sector amount_act and cfl_act values
            if data['margins']:
                amount_act_transport = tns_share / purchaser_to_basic
                for act_transport in exiobase_transport_margin_sectors:
                    cfl_act_transport = cfl_acts[exchange_name] * exiobase_transport_margin_sectors_dict[act_transport]
                    margins_exchanges.append(
                        {
                            'name': act_transport,
                            'location': 'CH',
                            'amount': chf_to_mln_euro * purchaser_to_basic * cfl_act_transport * amount_act_transport,
                            'unit': 'million €',
                            'database': ex_name,
                            'type': 'technosphere',
                        }
                    )
                amount_act_trade = trd_share / purchaser_to_basic
                for act_trade in exiobase_trade_margin_sectors:
                    cfl_act_trade = cfl_acts[exchange_name] * exiobase_trade_margin_sectors_dict[act_trade]
                    margins_exchanges.append(
                        {
                            'name': act_trade,
                            'location': 'CH',
                            'amount': chf_to_mln_euro * purchaser_to_basic * cfl_act_trade * amount_act_trade,
                            'unit': 'million €',
                            'database': ex_name,
                            'type': 'technosphere',
                        }
                    )
        # Correction for shares from all sectors
        co.data[i]['exchanges'] = production_exchange + technosphere_exchanges + margins_exchanges

    # Sum up repetitive exchanges

    return co