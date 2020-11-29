import brightway2 as bw
from pypardiso import spsolve
import numpy as np

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
        bw.projects.set_current(self.project)
        self.demand = demand
        self.exiobase_scores_precomputed = exiobase_scores_precomputed
        self.methods = list(list(exiobase_scores_precomputed.values())[0].keys())
        self.lca = bw.LCA(self.demand, self.methods[0])
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
            bw_act = bw.get_activity(reverse_dict[act]).as_dict()['key']
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

#     def contribution_analysis(self):
