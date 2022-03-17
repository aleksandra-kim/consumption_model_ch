from time import time
# import functools
from copy import deepcopy

import bw2io as bi
import bw2data as bd
from bw2io.importers.base_lci import LCIImporter
# from bw2io.strategies import migrate_datasets, migrate_exchanges

from ..extractors import ConsumptionDbExtractor
from ..migrations import create_consumption_db_migrations
from ..strategies import modify_exchanges, link_exiobase

# Default name of the consumption database
CONSUMPTION_DB_NAME = 'swiss consumption 1.0'


class ConsumptionDbImporter(LCIImporter):
    format = "ConsumptionDB Excel"

    def __init__(
        self,
        directory,
        name=None,
        exclude_databases=(),
        replace_agribalyse_with_ecoinvent=True,
        exiobase_path=None,
    ):
        start = time()
        self.directory = directory
        self.db_name = name or CONSUMPTION_DB_NAME
        self.df, self.filepath_consumption_excel = ConsumptionDbExtractor.extract(
            directory,
            name=self.db_name,
            exclude_databases=exclude_databases,
            replace_agribalyse_with_ecoinvent=replace_agribalyse_with_ecoinvent,
        )
        print(
            u"Created consumption_db.xlsx file in BW format in {:.2f} seconds".format(
                time() - start
            )
        )
        co = bi.ExcelImporter(self.filepath_consumption_excel)
        co.data = self.update_ecoinvent_name(co.data)
        if "exiobase 2.2" not in exclude_databases:
            co.data = self.update_exiobase_name(co.data)
        co = self.map_by_production_volume(co)
        create_consumption_db_migrations()
        co.migrate("ecoinvent-35-36-37-38-change-names")
        co.migrate("ecoinvent-36-371-38-rice-nonbasmati")
        co.migrate("ecoinvent-38-marine-fish")
        exclude_databases = [db.lower() for db in exclude_databases]
        if "exiobase 2.2" not in exclude_databases:
            co.migrate("exiobase-381-change-names")
            co.migrate("exiobase-row-locations")
            exiobase_name = self.determine_exiobase_db_name()
            co = link_exiobase(co, exiobase_name, exiobase_path)
        self.strategies = [
            # functools.partial(migrate_datasets, migration="ecoinvent-35-36-37-38-change-names"),
            # functools.partial(migrate_exchanges, migration="ecoinvent-35-36-37-38-change-names"),
            # functools.partial(migrate_exchanges, migration="ecoinvent-36-371-38-rice-nonbasmati"),
            # functools.partial(migrate_exchanges, migration="ecoinvent-38-marine-fish"),
        ]
        self.data = co.data

    @classmethod
    def update_ecoinvent_name(cls, data):
        # Figure out the new ecoinvent name
        ei_name = cls.determine_ecoinvent_db_name()
        # Replace `ecoinvent 3.3 cutoff` that was used in the original consumption model with the current ecoinvent
        data_updated = deepcopy(data)
        for act in data_updated:
            for exc in act.get('exchanges', []):
                if exc['database'] == 'ecoinvent 3.3 cutoff':
                    exc.update(dict(database=ei_name))
        return data_updated

    @classmethod
    def update_exiobase_name(cls, data):
        # Figure out the new ecoinvent name
        ex_name = cls.determine_exiobase_db_name()
        # Replace `exiobase 2.2` that was used in the original consumption model with the current exiobase
        data_updated = deepcopy(data)
        for act in data_updated:
            for exc in act.get('exchanges', []):
                if exc['database'] == 'EXIOBASE 2.2':
                    exc.update(dict(database=ex_name))
        return data_updated

    @staticmethod
    def determine_ecoinvent_db_name():
        """Function that naively determines how ecoinvent is called in a particular BW project."""
        ei_name = [db for db in bd.databases if 'ecoinvent' in db.lower()]
        assert len(ei_name) == 1
        return ei_name[0]

    @staticmethod
    def determine_exiobase_db_name():
        """Function that naively determines how ecoinvent is called in a particular BW project."""
        ex_name = [db for db in bd.databases if 'exiobase' in db.lower() and 'biosphere' not in db.lower()]
        assert len(ex_name) == 1
        return ex_name[0]

    @classmethod
    def map_by_production_volume(cls, db):
        ei_name = cls.determine_ecoinvent_db_name()
        ei = bd.Database(ei_name)
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
        db = modify_exchanges(db, mapping, ei_name)
        return db

    def write_database(self, data=None, name=None, *args, **kwargs):
        db = super(ConsumptionDbImporter, self).write_database(data, name, *args, **kwargs)
        db.metadata.update(
            {
                "fp_consumption_db": str(self.filepath_consumption_excel),
                "dir_habe": str(self.directory),
            }
        )
        db._metadata.flush()
        return db

    # Migration("exiobase-381-change-names").write(
    #     get_exiobase_migration_data(),
    #     "Change names of some exiobase 3.8.1 activities",
    # )


# 4. Link to other databases
#     co = bi.ExcelImporter(path_new_db)
#     co.apply_strategies()
#     co.match_database(fields=('name', 'unit', 'location', 'categories'))
#     co.match_database(ei_name, fields=('name', 'reference product', 'unit', 'location', 'categories'))
#
    # # 4.1 Ecoinvent
    # # Define migration for particular activities that can only be hardcoded
    # if "3.5" in ei_name or "3.6" in ei_name or "3.7" in ei_name or "3.8" in ei_name:
    #     print("Migration for 'steam production in chemical industry' and 'market for green bell pepper'")
    #     ecoinvent_35_36_37_38_change_names_data = json.load(open(DATADIR / "migrations" / "ecoinvent-3.5-3.6-3.7.1-3.8.json"))
    #     bi.Migration("ecoinvent-35-36-37-38-change-names").write(
    #         ecoinvent_35_36_37_38_change_names_data,
    #         description="Change names of some activities"
    #     )
    #     co.migrate('ecoinvent-35-36-37-38-change-names')
    #     co.match_database(ei_name, fields=('name', 'reference product', 'unit', 'location', 'categories'))

    # # Define a migration for rice production and specific locations
    # if "3.6" in ei_name or "3.7" in ei_name or "3.8" in ei_name:
    #     ecoinvent_36_371_38_rice_nonbasmati = json.load(open(DATADIR / "migrations" / "ecoinvent-3.6-3.7.1-3.8.json"))
    #     bi.Migration("ecoinvent-36-371-38-rice-nonbasmati").write(
    #         ecoinvent_36_371_38_rice_nonbasmati,
    #         description="Change names of some activities"
    #     )
    #     co.migrate('ecoinvent-36-371-38-rice-nonbasmati')
    #     co.match_database(ei_name, fields=('name', 'reference product', 'unit', 'location', 'categories'))

    # if "3.8" in ei_name:
    #     ecoinvent_38_marine_fish = json.load(open(DATADIR / "migrations" / "ecoinvent-3.8.json"))
    #     bi.Migration("ecoinvent-38-marine-fish").write(
    #         ecoinvent_38_marine_fish,
    #         description="Change reference product"
    #     )
    #     co.migrate('ecoinvent-38-marine-fish')
    #     co.match_database(ei_name, fields=('name', 'reference product', 'unit', 'location', 'categories'))
#
#     # Manually choose which ecoinvent exchanges should be taken for each unlinked exchange
#     # - The rest of the unlinked exchanges are not uniquely defined in ecoinvent 3.6 -> 1-to-multiple mapping.
#     # - For example 'rice production' is now divided into basmati and non-basmati rice.
#     # - Hence, we split them based on their shares in the production volumes.
#     ei = bd.Database(ei_name)
#     mapping = [
#         {('market for rice', 'GLO'):
#              [act['code'] for act in ei if 'market for rice' in act['name']
#               and act['location'] == 'GLO'
#               and 'seed' not in act['name']]},
#
#         {('rice production', 'RoW'):
#              [act['code'] for act in ei if 'rice production' in act['name']
#               and act['location'] == 'RoW'
#               and 'straw' not in act['reference product']]},
#
#         {('rice production', 'IN'):
#              [act['code'] for act in ei if 'rice production' in act['name']
#               and act['location'] == 'IN'
#               and 'straw' not in act['reference product']]},
#
#         {('market for wheat grain', 'GLO'):
#              [act['code'] for act in ei if 'market for wheat grain' in act['name']
#               and 'feed' not in act['name']]},
#
#         {('market for maize grain', 'GLO'):
#              [act['code'] for act in ei if 'market for maize grain' in act['name']
#               and 'feed' not in act['name']]},
#
#         {('market for mandarin', 'GLO'):
#              [act['code'] for act in ei if 'market for mandarin' in act['name']]},
#
#         {('market for soybean', 'GLO'):
#              [act['code'] for act in ei if 'market for soybean' in act['name']
#               and all([_ not in act['name'] for _ in ['meal', 'beverage', 'seed', 'feed', 'oil']])]},
#     ]
#     co = modify_exchanges(co, mapping, ei_name)
#
#     # 4.2 Agribalyse
#     ag_name = 'Agribalyse 1.3 - {}'.format(ei_name)
#     if ag_name not in exclude_databases and ag_name in bd.databases and not replace_agribalyse_with_ecoinvent:
#         print("-->Linking to {}".format(ag_name))
#         co.match_database(ag_name, fields=('name', 'unit', 'location'))
#
#     # 4.3 Exiobase
#     ex_name = 'exiobase 2.2'
#     if ex_name not in exclude_databases:
#         for db_name in bd.databases:
#             if "exiobase" in db_name.lower():
#                 ex_name = db_name
#         print("--> Linking to {}".format(ex_name))
    #     migrations_exiobase_filepath = DATADIR / "migrations" / "exiobase-3.8.1.json"
    #     if "3.8.1" in ex_name:
    #         print("Migration for {}".format(ex_name))
    #         # Only based on the `name` field
    #         exiobase_381_change_names_data = json.load(open(migrations_exiobase_filepath))
    #         bi.Migration("exiobase-381-change-names").write(
    #             exiobase_381_change_names_data,
    #             description="Change names of some exiobase 3.8.1 activities"
    #         )
#             co.migrate('exiobase-381-change-names')
#             margins_path = Path(sut_path) / 'CH_2015.xls'
#         elif '2.2' in ex_name:
#             margins_path = Path(sut_path) / 'CH_2007.xls'
#
#         co = link_exiobase(co, ex_name, exiobase_path, margins_path, migrations_exiobase_filepath)
#         co.match_database(ex_name, fields=('name', 'unit', 'location',))
#
#     co.statistics()
#     if len(list(co.unlinked)) == 0:
#         print("Writing consumption database")
#         co.write_database()
#         # Give the required name to the consumption database
#         if consumption_db_name != CONSUMPTION_DB_NAME:
#             co_diff_name = bd.Database(CONSUMPTION_DB_NAME)
#             co_diff_name.rename(consumption_db_name)
#
#         # Sum up repetitive exchanges
#         co = bd.Database(consumption_db_name)
#         for act in co:
#             excs = [exc.input for exc in act.exchanges()]
#             rep_inputs = {exc for exc in excs if excs.count(exc) > 1}
#             for rep_input in rep_inputs:
#                 amounts = [exc.amount for exc in act.exchanges() if exc.input == rep_input]
#                 [exc.delete() for exc in act.exchanges() if exc.input == rep_input]
#                 act.new_exchange(
#                     input=rep_input,
#                     amount=sum(amounts),
#                     type='technosphere'
#                 ).save()
#     else:
#         print("Some exchanges are still unlinked")