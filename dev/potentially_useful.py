import bw2data as bd
import bw2calc as bc
import bw2io as bi
from pypardiso import spsolve
import numpy as np
from pathlib import Path
import json

from consumption_model_ch.strategies.allocation import modify_exchanges

dirpath = Path(__file__).parent.resolve() / "data"


class exiobaseLCA:

    def __init__(
            self,
            project,
            demand,
            exiobase_scores_precomputed,
            exiobase_name='EXIOBASE 2.2',
            ecoinvent_name='ecoinvent 3.6 cutoff',
            agribalyse_name='Agribalyse 1.3 - ecoinvent 3.6 cutoff',
            ch_consumption_name='CH consumption 1.0',
    ):
        # BW / LCA setup
        self.project = project
        bd.projects.set_current(self.project)
        self.demand = demand
        self.exiobase_scores_precomputed = exiobase_scores_precomputed
        self.methods = list(list(exiobase_scores_precomputed.values())[0].keys())
        self.lca = bc.LCA(self.demand, self.methods[0])
        self.lca.lci()
        self.lca.lcia()
        self.lca.build_demand_array()
        # Database names
        self.exiobase_name = exiobase_name
        self.ecoinvent_name = ecoinvent_name
        self.agribalyse_name = agribalyse_name
        self.ch_consumption_name = ch_consumption_name
        # Find databases indices in matrices
        self.biosphere_without_exiobase, self.d_exiobase_adjusted = self.precompute()
        self.weights = self.compute_exiobase_weights()

    def precompute(self):
        ## Precompute whatever possible
        A = self.lca.technosphere_matrix
        # Note that A has a block triangular form (order of databases might be different)
        # [   A_ch        0         0      0
        #   L_ag_ch      A_ag       0      0
        #   L_ec_ch    L_ec_ag     A_ec    0
        #   L_ex_ch       0         0     A_ex ]

        B = self.lca.biosphere_matrix
        # B in block form
        # [  B_ch  B_ag  B_ec  B_ex ]

        d = self.lca.demand_array
        # Demand in block form
        # [  d_ch  d_ag  d_ec  d_ex ]

        # Find indices of activities for each database (where the databases start and end)
        keys_db = [k[0] for k in list(self.lca.activity_dict.keys())]

        # 1. Exiobase
        db_list = [ind for ind, val in enumerate(keys_db) if val == self.exiobase_name]
        min_ex, max_ex = min(db_list), max(db_list) + 1
        d_exiobas = d[min_ex:max_ex]
        A_exiobas = A[min_ex:max_ex, min_ex:max_ex]
        B_exiobas = B[:, min_ex:max_ex]

        # 2. Ecoinvent
        db_list = [ind for ind, val in enumerate(keys_db) if val == self.ecoinvent_name]
        min_ec, max_ec = min(db_list), max(db_list) + 1
        d_ecoinve = d[min_ec:max_ec]
        A_ecoinve = A[min_ec:max_ec, min_ec:max_ec]
        B_ecoinve = B[:, min_ec:max_ec]

        # 3. Agribalyse
        db_list = [ind for ind, val in enumerate(keys_db) if val == self.agribalyse_name]
        min_ag, max_ag = min(db_list), max(db_list) + 1
        d_agribal = d[min_ag:max_ag]
        A_agribal = A[min_ag:max_ag, min_ag:max_ag]
        B_agribal = B[:, min_ag:max_ag]

        # 4. CH consumption database
        db_list = [ind for ind, val in enumerate(keys_db) if val == self.ch_consumption_name]
        min_ch, max_ch = min(db_list), max(db_list) + 1
        d_consump = d[min_ch:max_ch]
        A_consump = A[min_ch:max_ch, min_ch:max_ch]
        B_consump = B[:, min_ch:max_ch]

        # 5. L matrices are links between different databases
        L_ag_ch = A[min_ag:max_ag, min_ch:max_ch]  # ch_consumption and agribalyse
        L_ec_ch = A[min_ec:max_ec, min_ch:max_ch]  # ch_consumption and ecoinvent
        L_ex_ch = A[min_ex:max_ex, min_ch:max_ch]  # ch_consumption and exiobase
        L_ec_ag = A[min_ec:max_ec, min_ag:max_ag]  # agribalyse and ecoinvent

        # 6. Solutions of system of linear equations for all databases
        x_consump = spsolve(A_consump, d_consump)
        x_agribal = spsolve(A_agribal, d_agribal - L_ag_ch * x_consump)
        x_ecoinve = spsolve(A_ecoinve, d_ecoinve - L_ec_ch * x_consump - L_ec_ag * x_agribal)

        # 7. LCIA score without exiobase
        biosphere_without_exiobase = B_consump * x_consump \
                                     + B_agribal * x_agribal \
                                     + B_ecoinve * x_ecoinve

        # 8. Adjusted exiobase demand
        d_exiobas_adjusted = d_exiobas - L_ex_ch * x_consump

        return biosphere_without_exiobase, d_exiobas_adjusted

    def compute_exiobase_weights(self):
        reverse_dict = self.lca.reverse_dict()[0]
        act_inds = np.where(self.d_exiobase_adjusted != 0)[0]
        weights = {}
        for act in act_inds:
            bw_act = bd.get_activity(reverse_dict[act]).as_dict()['key']
            weights[bw_act] = self.d_exiobase_adjusted[act]
        return weights

    def compute_exiobase_scores(self):
        exiobase_scores = {method: 0 for method in self.methods}
        for key, val in self.weights.items():
            for method in self.methods:
                exiobase_scores[method] += self.exiobase_scores_precomputed[key][method] * val
        return exiobase_scores

    def compute_ch_ag_ec_scores(self):
        ch_ag_ec_scores = {}
        for method in self.methods:
            self.lca.switch_method(method)
            self.lca.redo_lcia()
            C = sum(self.lca.characterization_matrix)
            score = C * self.biosphere_without_exiobase
            ch_ag_ec_scores[method] = score[0]
        return ch_ag_ec_scores

    def compute_total_scores(self):
        exiobase_scores = self.compute_exiobase_scores()
        ch_ag_ec_scores = self.compute_ch_ag_ec_scores()
        total_scores = {}
        for method in self.methods:
            total_scores[method] = exiobase_scores[method] + ch_ag_ec_scores[method]
        return total_scores


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
        agribalyse13_ecoinvent_names = json.load(open(dirpath / "migrations" / "agribalyse-1.3.json"))
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
