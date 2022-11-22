import bw2data as bd
import bw2io as bi
from copy import deepcopy

from .importers import ConsumptionDbImporter


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

        
def import_exiobase_3(ex3_path, ex3_name):
    if ex3_name not in bd.databases:
        ex = bi.Exiobase3MonetaryImporter(ex3_path, ex3_name)  # give path to IOT_year_pxp folder
        ex.apply_strategies()
        ex.write_database()


def import_consumption_db(directory_habe, year_habe, co_name, ei33_path, exclude_databases=(), exiobase_path=None):
    if co_name not in bd.databases:
        create_ecoinvent_33_project(ei33_path)
        co = ConsumptionDbImporter(
            directory_habe,
            year_habe,
            exclude_databases=exclude_databases,
            replace_agribalyse_with_ecoinvent=True,
            exiobase_path=exiobase_path,
        )
        ei_name = co.determine_ecoinvent_db_name()
        co.match_database()
        co.match_database(db_name='biosphere3', fields=('name', 'category', 'unit', 'location'))
        co.match_database(db_name=ei_name, fields=('name', 'unit', 'reference product', 'location'))

        use_exiobase = True
        for db in exclude_databases:
            if 'exiobase' in db.lower():
                use_exiobase = False
                break
        if use_exiobase:
            ex_name = co.determine_exiobase_db_name()
            co.match_database(db_name=ex_name, fields=('name', 'unit', 'location'))

        co.apply_strategies()
        co.statistics()
        co.write_database()

