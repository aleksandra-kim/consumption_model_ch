import pandas as pd
import numpy as np
from copy import copy
import os, json
import re
import brightway2 as bw


###########################
### 1. Define constants ###
###########################

# Database name
CONSUMPTION_DB_NAME = 'CH consumption 1.0'
# Number of relevant columns in the raw file (df_raw) to extract info about activity
N_ACT_RELEVANT = 11
# Index of the column where activities start
FIRST_ACT_IND = 7
# Number of columns that contain info about one activity
N_COLUMNS_INPUT_ACTIVITY = 5

# Column names for exchanges needed by brightway
EXC_COLUMNS_DICT = {
        'name': 'A', 
        'reference product': 'B', 
        'location': 'C', 
        'amount': 'D', 
        'unit': 'E', 
        'database': 'F', 
        'type': 'G', 
        'categories': 'H',
        'comment': 'I',
    }

# Conversion from type in databases to type that should be in excel file to import a new database
ACTIVITY_TYPE_DICT = {
    'process': 'technosphere',
    'emission': 'biosphere',
}

# Units conversion for consistency
UNIT_DICT = {
    'kg': 'kilogram',
    'lt': 'litre'
}


####################################
### 2.Import ecoinvent 3.3 cutoff ##
####################################
# This is needed because Andi's activities were given as ecoinvent 3.3 cutoff codes, 
# so information on the exact location, reference product, etc is contained in that code.
# Without using ecoinvent 3.3, we are left with numerous unlinked exchanges

def create_ecoinvent_33_project(ei33_path):
    
    ei33_name = "ecoinvent 3.3 cutoff"  
    bw.projects.set_current(ei33_name)
    
    bw.bw2setup()
    
    if ei33_name in bw.databases:
        print(ei33_name + " database already present!!! No import is needed")
    else:
        ei33 = bw.SingleOutputEcospold2Importer(ei33_path, ei33_name)
        ei33.apply_strategies()
        ei33.match_database(db_name='biosphere3',fields=('name', 'category', 'unit', 'location'))
        ei33.statistics()
        ei33.write_database()



###################################################################
### 3.Convert data to brightway database format -> all functions ##
###################################################################

def complete_columns(df):
    '''
    Add missing On columns
    ''' 
    
    column_names = list(df.columns)
    indices = [i for i,el in enumerate(column_names)  if 'Activity' in el]
    column_names_complete = copy(column_names)

    n_el_added = 0
    for ind in indices:
        if 'On' not in column_names[ind-1]:
            act_name = column_names[ind]
            act_number = act_name[act_name.find(' ')+1:]
            column_names_complete.insert(ind+n_el_added, 'On ' + act_number)
            n_el_added += 1
        
    df.columns = column_names_complete[:len(column_names)]
    
    return df


def create_df_bw(db_name, n_cutoff_cols = len(EXC_COLUMNS_DICT)+3):
    '''
    Create dataframe for a new database in the Brightway format and add the necessary meta information
    '''
    df = pd.DataFrame([['cutoff', n_cutoff_cols], ['database', db_name]], columns=list('AB'))
    df = df.append(pd.Series(), ignore_index=True)
    return df


def compute_act_unit(df, code_unit):
    '''
    Depending on whether `Quantity code` is present for a specific activity, 
    set unit to the unit of the first input activity or CHF.
    Comments on units from Andi for all codes that start with `mx`:
        - kWh per year for electricity
        - MJ per year for heating
        - cubic meters per year for water supply and wastewater collection
        - number of waste bags per year for refuse collection
        --> we gonna hardcode them ;)
        --> # TODO important Andi's model (total demands excel file) gives per year, but we divide by 12 later on
    '''
    
    unit = df['DB Act 1'].split('(')[1].split(',')[0]
    
    if unit == 'million \u20ac':
        unit = 'CHF'
        
    if 'Quantity code' in df.keys():
        name = df['Translated name'].lower()
        code = df['Quantity code']
        if 'electricity' in name:
            unit = 'kilowatt hour'
        elif 'heating' in name:
            unit = 'megajoule'
        elif 'water supply' in name:
            unit = 'cubic meter'
        elif 'wastewater collection' in name:
            unit = 'cubic meter'
        elif 'refuse collection' in name:
            unit = "number of waste bags"
        elif code in code_unit.keys():
            unit = UNIT_DICT[code_unit[code]]
            
    return unit


def append_activity(df, df_ind, code_unit):
    '''
    Append activity from row df_ind to the dataframe df in the brightway format
    '''
    # Append empty row
    df = df.append(pd.Series(), ignore_index=True)
    
    # Extract activity information
    act_name = df_ind['Translated name']
    if 'Quantity code' in df_ind.index:
        act_code = df_ind['Quantity code']
    else:
        act_code = df_ind['Variable code']
    act_unit = compute_act_unit(df_ind, code_unit)
    
    len_df = len(df)
    
    act_data = [ ['Activity', act_name],
                 ['reference product',  act_name],
                 ['code', act_code],
                 ['location', 'CH'],
                 ['amount', 1],
                 ['unit', act_unit] ]
    
    df_act = pd.DataFrame( act_data, 
                           columns=list('AB'),
                           index = np.arange(len_df,len_df+len(act_data)) )
                          
    df = df.append(df_act, sort=False)
    
    return df, df_act


def append_exchanges_in_correct_columns(df, dict_with_values):
    '''
    Make sure that exchanges values are appended to df in the correct columns.
    '''  
    col_names = list(dict_with_values.keys()) # order of columns is determined by this list
    col_excel_literal = [EXC_COLUMNS_DICT[m] for m in col_names]
    
    if dict_with_values != EXC_COLUMNS_DICT:
        col_data  = [dict_with_values[m] for m in col_names]
    else:
        col_data = col_names
    
    df = df.append(pd.DataFrame([col_data], columns=col_excel_literal, index=[len(df)]), sort=False)
    
    return df


def append_exchanges_column_names(df):
    '''
    Add column names for exchanges
    '''
    df = df.append(pd.DataFrame(['Exchanges'], columns=['A'], index=[len(df)]), sort=False)
    df = append_exchanges_in_correct_columns(df, EXC_COLUMNS_DICT)
    return df


def append_first_exchange(df, df_act_dict):
    '''
    Append first exchange which is activity itself, the amount is always 1, 
    the database is always the one that is being currently created, type is `production`.
    '''
    
    first_exc_data_dict = { 'name': df_act_dict['Activity'],
                            'reference product': df_act_dict['reference product'],
                            'location': df_act_dict['location'],
                            'amount': 1,
                            'unit': df_act_dict['unit'],
                            'database': CONSUMPTION_DB_NAME,
                            'type': 'production',
                          }
    
    df = append_exchanges_in_correct_columns(df, first_exc_data_dict)
    
    return df


def is_pattern_correct(df_ind_j):
    '''
    Check that input activity info has correct pattern. 
    In case the pattern is not correct, move on to the next 5 columns and check their pattern.
    This is needed because for some input activities some relevant values are missing, eg only 'On' value is present.
    '''
    list_ = list(df_ind_j.index)
    pattern = ['On', 'Activity', 'DB Act', 'CFL Act', 'Amount Act']
    check = [pattern[n] in list_[n] for n in range(N_COLUMNS_INPUT_ACTIVITY)]
    if np.all(check): 
        return 1
    else: 
        return 0


def append_one_exchange(df, df_ind_j, ConversionDem2FU):
    '''
    Extract information about one input activity, eg name, unit, location, etc and append it to the dataframe df.
    '''    
    # Extract the activity number
    k = int(''.join(c for c in df_ind_j.index[0] if c.isdigit()))
    # Extract information about activity and save it
    input_act_str = df_ind_j['DB Act ' + str(k)]
    input_act_db_code = df_ind_j['Activity ' + str(k)]
    
    # Find this input activity in brightway databases
    db_name = input_act_db_code.split("'")[1]
    code = input_act_db_code.split("'")[3]
    input_act_db_code_tuple = (db_name, code)
    
    # TODO remove HEIA for now
    if 'heia' in db_name:
        return df
    
    # Compute amount
    input_act_amount = df_ind_j['On ' + str(k)] \
                     * df_ind_j['Amount Act ' + str(k)] \
                     * df_ind_j['CFL Act ' + str(k)] \
                     * ConversionDem2FU
    
    try:
        # Find activity in the ecoinvent 3.3 cutoff using bw functionality
#         current_project = deepcopy(bw.projects.current)
#         bw.projects.set_current('ecoinvent 3.3 cutoff') # Temporarily switch to ecoinvent 3.3 project
        act_bw = bw.get_activity(input_act_db_code_tuple)
#         bw.projects.set_current(current_project)
        input_act_values_dict = create_input_act_dict(act_bw, input_act_amount)
        
    except:
        # If bw.get_activity does not work for whichever reason, fill info manually
        input_act_values_dict = bw_get_activity_info_manually(input_act_str, db_name, input_act_amount)
        
    # Add exchange to the dataframe with database in brightway format
    df = append_exchanges_in_correct_columns(df, input_act_values_dict)
    
    return df


def append_exchanges(df, df_ind, df_act, append_one_exchange=append_one_exchange):
    '''
    Add all exchanges (input activities) from the row df_ind to consumption database dataframe.
    '''
    # Add exchanges column names
    df = append_exchanges_column_names(df)
    
    # Add first exchange that is the same as the activity itself, type of this exchange is production
    df_act_dict = df_act.set_index('A').to_dict()['B']
    df = append_first_exchange(df, df_act_dict)
    
    # Add all exchanges
    n_exchanges = (len(df_ind)-FIRST_ACT_IND) // N_COLUMNS_INPUT_ACTIVITY
#     if n_exchanges != (len(df_ind) - FIRST_ACT_IND) / N_COLUMNS_INPUT_ACTIVITY:
#         print('smth is not right with exchanges of Activity -> ' + str(df_ind['Translated name']))
    
    if df_act_dict['unit'] == 'CHF':
        # For activities that have only exiobase exchanges, conversion factor is multiplied by 1e6
        # We assume that activities that have exiobase exchanges, have ONLY exiobase exchanges
        ConversionDem2FU = df_ind['ConversionDem2FU']
    else:
        ConversionDem2FU = df_ind['ConversionDem2FU']
        
    skip = 0
    for j in range(1, n_exchanges+1):
        
        start = FIRST_ACT_IND + N_COLUMNS_INPUT_ACTIVITY*(j-1) + skip
        end = start + N_COLUMNS_INPUT_ACTIVITY
        df_ind_j = df_ind[start:end]
        
        #Check that df_ind_j contains <On 1, Activity 1, DB Act 1, CFL Act 1, Amount Act 1> pattern
        flag = 1
        while flag:
            flag_pattern = is_pattern_correct(df_ind_j) 
            if flag_pattern == 1: # we don't need to skip if patter is correct
                flag = 0
            else:
                skip += 1
                start = FIRST_ACT_IND + N_COLUMNS_INPUT_ACTIVITY*(j-1) + skip
                end = start + N_COLUMNS_INPUT_ACTIVITY
                df_ind_j = df_ind[start:end]
        
        df = append_one_exchange(df, df_ind_j, ConversionDem2FU)
        
    return df


def create_input_act_dict(act_bw, input_act_amount):
    '''
    Create a dictionary with all info about input activities.
    '''
    
    input_act_values_dict = {
        'name': act_bw['name'], 
        'location': act_bw['location'], 
        'amount': input_act_amount, 
        'unit': act_bw['unit'], 
        'database': act_bw['database'], 
        # We do not expect type biosphere, but assign it via ACTIVITY_TYPE_DICT anyway 
        # to be sure that we don't encounter them.
        'type': ACTIVITY_TYPE_DICT[act_bw['type']],
    }
    try:
        input_act_values_dict['reference product'] = act_bw['reference product']
    except:
        pass
            
    return input_act_values_dict


def bw_get_activity_info_manually(input_act_str, db_name, input_act_amount):
    # Extract the activity name
    apostrophes = [(m.start(0), m.end(0)) for m in re.finditer("'", input_act_str)]
    if len(apostrophes) == 1:
        ap_start = 0
        ap_end = apostrophes[0][0]
    else:
        ap_start = apostrophes[0][1]
        ap_end = apostrophes[1][0]
    input_act_name = input_act_str[ ap_start:ap_end ]
    input_act_unit_loc = input_act_str[ input_act_str.find("(") : input_act_str.find(")")+1 ]
    input_act_unit_loc_split = [ re.sub('[^-A-Za-z0-9-€-]', ' ' , el).rstrip().lstrip() \
                                 for el in input_act_unit_loc.split(',')]
    input_act_unit = input_act_unit_loc_split[0]
    input_act_location = input_act_unit_loc_split[1]

    # Add comment when activity cannot be found
    input_act_values_dict = {}
    input_act_values_dict['name'] = input_act_name
    input_act_values_dict['unit'] = input_act_unit
    input_act_values_dict['location'] = input_act_location
    input_act_values_dict['amount'] = input_act_amount
    input_act_values_dict['database'] = db_name
    input_act_values_dict['type'] = ACTIVITY_TYPE_DICT['process'] # TODO remove hardcoding
    input_act_values_dict['comment'] = 'TODO could not find this activity'

    return input_act_values_dict


def get_units_habe(path, year):
    '''
    Extract information about units of some activities from HABE metadata.
    '''
    # Get path of the HABE data description (Datenbeschreibung)
    get_path = lambda which_file: os.path.join( path, 
                                               [f for f in os.listdir(path) if year in f and which_file in f][0] )
    path_beschrei = get_path('Datenbeschreibung')
    
    # Get meta information about units
    mengen_meta = pd.read_excel(path_beschrei, sheet_name='Mengen', skiprows=14, usecols=[1,2,4,7])
    mengen_meta.columns = ['category', 'name', 'code', 'unit']
    mengen_meta.dropna(subset=['code'], inplace=True)
    
    # Combine name and category columns together
    temp1 = mengen_meta[mengen_meta['category'].notnull()][['category', 'code', 'unit']]
    temp1.columns = ['name', 'code', 'unit']
    temp2 = mengen_meta[mengen_meta['name'].notnull()][['name', 'code', 'unit']]
    mengen_meta = pd.concat([temp1, temp2])
    
    mengen_meta.sort_index(inplace=True)
    
    # Get units for codes
    code_unit = {}
    for i, el in mengen_meta.iterrows():
        code = el['code'].lower()
        code_unit[code] = el['unit']
        
    return code_unit



##################################################################################################
### 4. Replace names of old databases with the new ones in the consumption database excel file ###
##################################################################################################

# Constants
DB_COLUMN = 'F'
CONSUMPTION_DB_NAME = 'CH consumption 1.0'

def replace_one_db(df, db_old_name, db_new_name):
    '''
    Replace database name with a new one (eg in case a newer version is available)
    '''
    df_updated = copy(df)
    
    where = np.where(df_updated[DB_COLUMN]==db_old_name)[0]
    if where.shape[0] != 0:
        df_updated[DB_COLUMN][where] = db_new_name
        
    return df_updated


def update_all_db(df):
    '''
    Update all databases in the consumption database
    '''
    db_old_list = ['Agribalyse 1.2', 
                   'ecoinvent 3.3 cutoff']
    db_new_list = ['Agribalyse 1.3 - ecoinvent 3.6 cutoff',
                  'ecoinvent 3.6 cutoff']
    
    assert len(db_old_list) == len(db_new_list)
    
    for i in range(len(db_old_list)):
        df = replace_one_db(df, db_old_list[i], db_new_list[i])
        
    return df