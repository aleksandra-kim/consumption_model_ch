# Local files
from import_databases import *
from utils_exiobase import exiobaseLCA


if __name__ == "__main__":
    if "exiobase_reproduce_results" not in bw.projects:
        bw.projects.set_current("exiobase_industry_workaround") # make sure this project only has biosphere3 and EXIOBASE 2.2
        # projects to reproduce Andi's LCA results with older databases: ecoinvent 3.3, exiobase 2.2, agribalyse 1.2
        bw.projects.copy_project("exiobase_reproduce_results")
    bw.projects.set_current("exiobase_reproduce_results")

    # define paths
    with open('global_settings.json', 'rb') as f:
        settings = json.load(f)
    which_pc = settings['which_pc']
    if which_pc == 'local':
        # Older databases
        ei33_path = '/Users/akim/Documents/LCA_files/ecoinvent 3.3 cutoff/datasets'
        ex22_path = '/Users/akim/Documents/LCA_files/exiobase 2.2/mrIOT_IxI_fpa_coefficient_version2.2.2'
        ag12_path = '/Users/akim/Documents/LCA_files/agribalyse 1.2/AGRIBALYSEv1.2_Complete.xml'
        habe_path = '/Users/akim/Documents/LCA_files/HABE_2017/'
        co_path = 'data/es8b01452_si_002.xlsx'

    # Import databases
    ei33_name = "ecoinvent 3.3 cutoff"
    import_ecoinvent(ei33_path, ei33_name)
    import_agribalyse12(ag12_path, ei33_name)
    co_name = "CH consumption 1.0"
    # if co_name in bw.databases:
    #     del bw.databases[co_name]
    import_consumption_db(co_path, habe_path, exclude_dbs=['heia'], ei_name=ei33_name)
    # add_consumption_activities(co_name, habe_path)
    add_consumption_categories(co_name, co_path)
    add_consumption_sectors(co_name)

    co = bw.Database(co_name)
    demand_act = co.search('ch hh average consumption')
    assert len(demand_act)==1
    demand_act = demand_act[0]
    demand = {demand_act: 1}
    method = ('IPCC 2013', 'climate change', 'GTP 100a')
    lca = bw.LCA(demand, method)
    lca.lci()
    lca.lcia()
    print("Andi's result: {}".format(lca.score))

    # project = "rebound"
    # fp_exiobase_scores_industry_workaround = Path("write_files") / "exiobase_lca.pickle"
    # with open(fp_exiobase_scores_industry_workaround, "rb") as f:
    #     exiobase_scores_industry_workaround = pickle.load(f)
    # exio_lca = exiobaseLCA(
    #     project,
    #     demand,
    #     exiobase_scores_industry_workaround,
    # )
    # scores = exio_lca.compute_total_scores()
    # print("Our scores: {}".format(scores))
