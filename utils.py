import brightway2 as bw
from copy import deepcopy

from bw2io.utils import rescale_exchange # rescales uncertainties as well

# Local files
# from rescale_uncertainty_ocelot import * # TODO change stuff here
#TODO mapping now contains only name and location, which doesn't always identify unique exchanges
# should include unit, reference product, etc


def get_allocated_excs(db, mapping, db_name):
    """
    Function that generates list of exchanges with 
    Each "inner" list contains dictionaries of exchanges in the correct format. 
    By correct format we mean the one consistent with the format of exchanges 
    in the (not written yet) consumption database (see eg co.data[0]['exchanges']).
    We exclude amounts for now and instead add field 'production volume share'

    Attributes
    ----------
    unlinked_list : list
        List that contains unlinked exchanges
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

    unlinked_names_loc = [0]*len_unlinked
    for i in range(len_unlinked):
        unlinked_names_loc[i] =  (unlinked_list[i]['name'], unlinked_list[i]['location'])
               
    unlinked_list_used = []
    allocated_excs = []
    
    for m in range(len(mapping)):
        try:
            # If current element from mapping is in unlinked exchanges, save it in `unlinked_list_used`
            index = unlinked_names_loc.index(list(mapping[m].keys())[0])
            unlinked_list_used.append(unlinked_list[index])
            
            # Change exchanges of the current activity if some of them are unlinked. 
            # This involves adding new allocation exchanges to `allocated_excs` and adding field `production volume share`  
            new_exchanges = []
            vols = 0
            codes = list(mapping[m].values())[0]
            for code in codes:
                act = bw.get_activity((db_name, code))
                production_exc = next(item for item in act.exchanges() if item['type']=='production')
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
    '''
    Function that compares two exchanges based on certain fields. Return True if exchanges are the same.
    '''

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
    fields_to_compare = ['name','location','unit','type']
    same = all([exc1[f]==exc2[f] for f in fields_to_compare])
    
    return same



def modify_exchanges(db, mapping, db_name):
    """
    Change exchanges of activities if they are unlinked, adjust their amount based on `production volume share` field.
    TODO: change the code to removing unlinked exchanges instead of adding them - line 121-...
    TODO: uncertainty info is not scaled!!!
    """

    db1 = deepcopy(db)
    unlinked_list_used, allocated_excs = get_allocated_excs(db, mapping, db_name)
    for act in db1.data: 
        try:
            exchanges = deepcopy(act['exchanges'])
            new_exchanges = []
            ind_amount = []

            for exc in exchanges:

                ind = next((i for i, item in enumerate(unlinked_list_used) if compare_exchanges(exc, item, db_name)), None)

                if ind != None:
                    # if we find current exchange in the unlinked exchanges list, replace it with other ones
                    # while using allocation by production volume
                    allocated_excs_new_amt = deepcopy(allocated_excs[ind])
                    # for exc_new_amt in allocated_excs_new_amt:÷
                        # exc_new_amt['amount'] = exc_new_amt['production volume share'] * exc['amount']
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

    db1.match_database(db_name, fields=('name','reference product', 'unit','location'))
    
    return db1

