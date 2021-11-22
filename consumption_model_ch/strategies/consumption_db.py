# -*- coding: utf-8 -*-
import bw2data as bd
import pandas as pd
import numpy as np
from copy import deepcopy
import re
from pathlib import Path
from bw2io.utils import rescale_exchange  # rescales uncertainties as well

from ..data import get_exiobase_migration_data, get_exiobase_margins_data


def get_allocated_excs(db, mapping, db_name):
    """
    Function that generates list of exchanges with
    Each "inner" list contains dictionaries of exchanges in the correct format.
    By correct format we mean the one consistent with the format of exchanges
    in the (not written yet) consumption database (see eg co.data[0]['exchanges']).
    We exclude amounts for now and instead add field 'production volume share'

    Parameters
    ----------
    db : bd.Database
        Database that contains unlinked exchanges
    mapping : list of dictionaries
        Each dictionary corresponds to an unlinked exchange with key = name of the unlinked exchange and
                                                                 values = codes of allocation activities
    db_name : name of the database to link to

    Returns
    -------
    unlinked_list_used : list
        List of unlinked exchanges that are actually present in the mapping.
    allocated_excs : list of lists
        Each inner list contains exchanges dictionaries

    """

    unlinked_list = list(db.unlinked)
    len_unlinked = len(unlinked_list)

    unlinked_names_loc = [0] * len_unlinked
    for i in range(len_unlinked):
        unlinked_names_loc[i] = (unlinked_list[i]['name'], unlinked_list[i]['location'])

    unlinked_list_used = []
    allocated_excs = []

    for m in range(len(mapping)):
        try:
            # If current element from mapping is in unlinked exchanges, save it in `unlinked_list_used`
            index = unlinked_names_loc.index(list(mapping[m].keys())[0])
            unlinked_list_used.append(unlinked_list[index])

            # Change exchanges of the current activity if some of them are unlinked. This involves adding new allocation
            # exchanges to `allocated_excs` and adding field `production volume share`
            new_exchanges = []
            vols = 0
            codes = list(mapping[m].values())[0]
            for code in codes:
                act = bd.get_activity((db_name, code))
                production_exc = next(item for item in act.exchanges() if item['type'] == 'production')
                vol = production_exc['production volume']
                vols += vol
                exc = deepcopy(unlinked_list[index])
                # Update some values to be consistent with db_name
                exc2 = {'name': act['name'],
                        'reference product': act['reference product'],
                        'location': act['location'],
                        'production volume share': vol,
                        'unit': act['unit'],
                        'database': db_name,
                        'type': 'technosphere'}
                exc.update(exc2)

                new_exchanges.append(exc)

            for exc in new_exchanges:
                exc['production volume share'] /= vols

            allocated_excs.append(new_exchanges)

        except ValueError:
            pass

    return unlinked_list_used, allocated_excs


def compare_exchanges(exc1, exc2, db_name):
    """Function that compares two exchanges based on certain fields. Return True if exchanges are the same."""

    # Do not consider biosphere exchanges
    if exc1['type'] == 'biosphere' or exc2['type'] == 'biosphere':
        return False

    # Do not need to consider exchanges that are not in the database we're linking to
    try:
        if exc1['input'][0] != db_name:
            return False
    except:
        pass

    # Compare exchanges based on their dictionary fields
    fields_to_compare = ['name', 'location', 'unit', 'type']
    same = all([exc1[f] == exc2[f] for f in fields_to_compare])

    return same


def modify_exchanges(db, mapping, db_name):
    """Change exchanges of activities if unlinked, adjust their amount based on `production volume share` field.

    TODO: change the code to removing unlinked exchanges instead of adding them - line 121-...
    TODO: uncertainty info is not scaled!!!

    """

    db1 = deepcopy(db)
    unlinked_list_used, allocated_excs = get_allocated_excs(db, mapping, db_name)
    for act in db1.data:
        try:
            exchanges = deepcopy(act['exchanges'])
            new_exchanges = []
            for exc in exchanges:
                ind = next(
                    (i for i, item in enumerate(unlinked_list_used) if compare_exchanges(exc, item, db_name)), None
                )
                if ind is not None:
                    # if we find current exchange in the unlinked exchanges list, replace it with other ones
                    # while using allocation by production volume
                    allocated_excs_new_amt = deepcopy(allocated_excs[ind])
                    # for exc_new_amt in allocated_excs_new_amt:
                    #     exc_new_amt['amount'] = exc_new_amt['production volume share'] * exc['amount']
                    for exc_new_amt in allocated_excs_new_amt:
                        if 'production volume share' in exc_new_amt:
                            exc_new_amt['amount'] = exc['amount']
                            new_exchanges.append(rescale_exchange(exc_new_amt, exc_new_amt['production volume share']))
                else:
                    # if we don't find current exchange in the unlinked exchanges list, append current to the list
                    new_exchanges.append(exc)

            act['exchanges'] = new_exchanges
        except:
            pass

    # db1.match_database(db_name, fields=('name', 'reference product', 'unit', 'location'))

    return db1


def get_margins_df(year, margins_label):
    dataframe = get_exiobase_margins_data(year, margins_label)
    columns = ['Unnamed: 2', 'Final consumption expenditure by households']
    dataframe = dataframe[columns]
    dataframe.columns = ['name', margins_label]
    dataframe = dataframe.dropna()
    dataframe = dataframe.set_index('name')
    return dataframe


def concat_margins_df(year):
    trd_label = 'trade_margins_init'
    tsp_label = 'transport_margins_init'
    tax_label = 'product_taxes_init'
    bpt_label = 'bptot_ini'  # basic price total
    ppt_label = 'purchaser_price'  # purchaser price total
    labels = [trd_label, tsp_label, tax_label, bpt_label, ppt_label]
    df_trd = get_margins_df(year, trd_label)
    df_tsp = get_margins_df(year, tsp_label)
    df_tax = get_margins_df(year, tax_label)
    df_bpt = get_margins_df(year, bpt_label)
    df_margins = pd.concat([df_trd, df_tsp, df_tax, df_bpt], axis=1)
    return df_margins, labels


def get_margins_shares(year):
    exiobase_381_change_names_data = get_exiobase_migration_data()
    exiobase_381_change_names_dict = {}
    for el in exiobase_381_change_names_data['data']:
        old_name = el[0][0]
        new_name = el[1]['name']
        exiobase_381_change_names_dict[old_name] = new_name

    df_margins, labels = concat_margins_df(year)
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


def link_exiobase(co, ex_name, ex_path):
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
        year = 2007
    elif '3.8.1' in ex_name:
        filename = "Y.txt"
        columns = ['region', 'Unnamed: 1', 'CH']
        chf_to_euro = 0.937234
        year = 2015

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
        # location_share = {}
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

    df_margins = get_margins_shares(year)
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
