# Written by Andreas Froemelt

import json
import brightway2 as bw2
import os
# from pyexio.exiobase_importer import ExiobaseImporter
from pyexio.exio_lca import ExioLCA
import numpy as np
import pyprind
# import multiprocessing as mp
import time

class MatchBiosphereFlows(object):
    """
    This class imports the migration json-file provided by Chris as a dict. PLEASE NOTE: the original migrations-file was
    adjusted: "Non-combustion"-emissions are now assigned to "fossil"-emissions (and not to "non-fossil" as suggested by
    Chris) to be consistent with the EXIOBASE-characterization factors. Also note that PFC- and HFC-emissions are assigned
    to fossil carbon dioxide since the PFC and HFC are already given in kgCO2-eq! THIS HAS TWO IMPLICATIONS: 1. PFC
    and HFC emissions are only considered for climate change issues; 2. PFC and HFC are not considered by EXIOBASE-
    characterization factors which should lead to higher results for subsequent computations with brightway2.
    """
    def __init__(self):

        assert os.path.exists(os.path.join(os.path.dirname(__file__), 'exiomigration_adjusted.json'))

        with open(os.path.join(os.path.dirname(__file__), 'exiomigration_adjusted.json')) as f:
            migration = json.load(f)

        self.migr_dict = {(m[0][0], m[1]['categories'][0]): m[1] for m in migration['data']}


class Exio2BW2Exporter(object):
    """
    This class constructs a new brightway2-activity based on the computed life cycle inventory of a EXIOBASE-activity.
    See also the MatchBiosphereFlows-description for the matching of biosphere-flows from EXIOBASE with biosphere-flows
    in ecoinvent.
    """
    def __init__(self, exio_lca, name, bw2_project, matchbiosphereflows=None):
        """
        Init-function of the Exio2BW2Exporter-class
        :param exio_lca: ExioLCA-instance which holds the computed life cycle inventory of an EXIOBASE-activity
        :param name: (string) Name for the brightway2-activity
        :param bw2_project: (string) Name of the brightway2-project to which the activity shall be exported
        :param matchbiosphereflows: MatchBiosphereFlows-instance
        """

        # First check if a MatchBiosphereFlows-instance was passed and if not, construct one on-the-fly
        if not matchbiosphereflows:
            matchbiosphereflows = MatchBiosphereFlows()

        # Extract the translation-dict from the MatchBiosphereFlows-instance to translate EXIOBASE-biosphere-flows to
        # ecoinvent-biosphere-flows
        self.migr_dict_ = matchbiosphereflows.migr_dict

        # Call the brightway2-database to which the EXIOBASE-activity shall be written
        self._db_ = self._bw2_setup(bw2_project)

        # Get the life cycle inventory data of the EXIOBASE-activity under consideration
        self.lci_ = self._get_lci(exio_lca)

        # Add the EXIOBASE-activity to the brightway2-project
        self._add_exio_to_bw2(name)


    def _bw2_setup(self, bw2_project):
        """
        This function extracts the brightway2-database to which the EXIOBASE-activity shall be written. If the brightway2-
        project does not exist, it will be setup.
        :param bw2_project: (string) Name for the brightway2-project to which EXIOBASE shall be exported
        :return: brightway2-database
        """

        # If the brightway2-project does not exist yet, we will set it up first
        if not bw2_project in bw2.projects:
            bw2.projects.set_current(bw2_project)
            bw2.bw2setup()
        else:
            bw2.projects.set_current(bw2_project)

        # Get or initialize the EXIOBASE-Database in the brightway2-project
        if not 'EXIOBASE 2.2' in bw2.databases:
            db = bw2.Database('EXIOBASE 2.2')
            db.write({})
        else:
            db = bw2.Database('EXIOBASE 2.2')

        return db


    def _get_lci(self, exio_lca):
        """
        This function computes the aggregated life cycle inventory of the activity passed to the class.
        :param exio_lca: ExioLCA-instance (see init-function)
        :return: Life cycle inventory data as a dict
        """
        return exio_lca.inventory.sum(axis=1).to_dict()


    def _add_exio_to_bw2(self, name):
        """
        This function constructs a new brightway2-activity. See also the MatchBiosphereFlows-description for
        the matching of biosphere-flows from EXIOBASE with biosphere-flows in ecoinvent.
        :param name: (string) Name for the brightway2-activity
        """

        # Initialize a new brightway2-activity:
        act = self._db_.new_activity(name)

        # Construct a name and location-attribute for the brightway2-activity:
        nm = name.split(':')
        if len(nm)>2:  # This is necessary since there are names which also contain a ":"
            nm1 = ':'.join(nm[:-1])
            nm2 = nm[-1]
        else:
            nm1 = nm[0]
            nm2 = nm[1]

        act['name'] = nm1
        act['location'] = nm2
        act['unit'] = 'million €'

        # Get the ecoinvent-biosphere-database
        bw2biosphere = bw2.Database("biosphere3")

        # Now we will iterate through all emissions according to the EXIOBASE-LCI-emissions and link it to the corresponding
        # biosphere-flows in ecoinvent
        for ky in self.lci_.keys():

            # The following if-loop is necessary to include land use correctly into brightway2. First: we need to
            # slightly change the key such that the land use category in the ecoinvent-biosphere can be found; Second:
            # we also need to convert km2 (given by EXIOBASE) to m2 (needed by ecoinvent)
            if ky[1] == 'nature':
                ky2 = (ky[0], 'natural resource')
                unitconverter = 1000000.
            else:
                ky2 = ky
                unitconverter = 1.

            # The following try-statement is necessary as not all biosphere-flows in EXIOBASE have a counterpart in
            # ecoinvent. HOWEVER: this does not mean that we neglect emissions in brightway2 --> all emissions which
            # contain a value in EXIOBASE have also a match in ecoinvent!
            try:
                bw2bioref = self.migr_dict_[ky2]
            except:
                continue

            # Extract the biosphere-flow from ecoinvent:
            ex = [(x, x.key) for x in bw2biosphere if bw2bioref['name'] == x['name'] and x['categories'] == tuple(bw2bioref['categories'])]

            # The following if-statement is just to make sure that we have full control over which biosphere-flow is
            # matched in the end. HOWEVER: it turned out that this if-statement is not needed.
            if len(ex) > 1:
                print("Biosphere flow is not unique:\n")
                for x in ex:
                    print(x[0])
                indx = input("give index of choice (starts with 0): ")
            else:
                indx = 0

            # Add a new exchange to the brightway2-activity
            newex = act.new_exchange(input=ex[indx][1], type='biosphere', amount=self.lci_[ky] * unitconverter)

            # The following if-statements add a new attribute to emissions for which the matching was doubtful --> see
            # the MatchBiosphereFlows-description for the matching of biosphere-flows from EXIOBASE with biosphere-flows
            # in ecoinvent. The insertion of this new attribute will allow for easily changing these flows later on if
            # necessary
            if ky[0]=="CO2 - non combustion":
                newex['orginal_exiobase_name'] = "CO2 - non combustion"
            elif ky[0]=="CH4 - non combustion":
                newex['orginal_exiobase_name'] = "CH4 - non combustion"
            elif ky[0]=="CO - non combustion":
                newex['orginal_exiobase_name'] = "CO - non combustion"
            elif ky[0]=="HFC":
                newex['orginal_exiobase_name'] = "HFC"
            elif ky[0]=="PFC":
                newex['orginal_exiobase_name'] = "PFC"

            newex.save()
            act.save()


def export_exio_to_bw2(bw2_project, overwrite=False, debugmode=False):
    """
    This function imports the whole EXIOBASE database as system processes in brightway2. For this it computes first the
    inventory for each EXIOBASE-activity, matches then the emissions to the brightway2-ecoinvent-biosphere and finally
    writes a new activity into a pre-defined brightway2-project. PLEASE NOTE: The EXIOBASE-import via ExiobaseImporter
    needs to be done beforehand.
    :param bw2_project: (String) Name of the brightway2-projects to which the EXIOBASE-system processes shall be exported
    :param overwrite: (Optional) True/False: indicate if already imported EXIOBASE activities shall be overwritten or
    not. If False, then computations can be interrupted and re-started later on.
    :param debugmode: (Optional) True/False: if true, only the first four processes will be computed
    """
    tic = time.time()

    # We start with initializing an ExioLCA-class and load the EXIOBASE-Input-Output-Table. PLEASE NOTE: you have to
    # imort EXIOBASE beforehand (via ExiobaseImporter)
    iot = ExioLCA(1.0)
    iot.load_lci_data()
    iot = iot.iot.copy()

    # Read the migrations-file for matching the EXIOBASE-Biosphereflows with the ecoinvent-biosphere-flows
    matchbiosphereflows = MatchBiosphereFlows()

    # Construct a demand-vector for a first LCI-computation that will "factorize" the technosphere
    dem = np.zeros(iot.shape[0])
    demvec = dem.copy()
    demvec[0] = 1.

    # "Pre"-LCI to factorize the technosphere matrix
    exio_lca = ExioLCA(demvec)
    exio_lca.lci()

    get_time("Time for preparation", tic)

    # The following if-loop checks if some EXIOBASE-activities were already imported and if yes, constructs a dict of
    # activities
    if bw2_project in bw2.projects:
        bw2.projects.set_current(bw2_project)
        if 'EXIOBASE 2.2' in bw2.databases:
            db = bw2.Database('EXIOBASE 2.2')
            acts = {(a['location'], a['name']): a for a in db}


    bar = pyprind.ProgBar(len(dem), monitor=True)

    # In the following, we iterate over all EXIOBASE-activities, compute the system processes and export them to brightway2
    for i, _ in enumerate(dem):

        # Get the name from the IOT-Table:
        name = iot.index[i]

        # The following try-statement checks if we have an activity-dict which indicates that part of the EXIOBASE has
        # already been imported:
        try:
            if name in acts.keys() and not overwrite:
                # In the case, we don't want to overwrite already imported activities, we will just skip the current activity
                bar.update()
                continue
            elif name in acts.keys():
                # Otherwise, we first delete the activity from the brightway2-project
                acts[name].delete()
        except:
            pass

        # Construct a new name for brightway2 (the name corresponds to the one which was suggested by Chris's import)
        name = "{}:{}".format(name[1], name[0])

        # Construct the demand-vector for the activity in question
        demvec = dem.copy()
        demvec[i] = 1.

        # Compute the life cycle inventory / system process
        exio_lca.redo_lci(demvec)

        # Export the activity to brightway2:
        Exio2BW2Exporter(exio_lca, name, bw2_project, matchbiosphereflows=matchbiosphereflows)

        bar.update()

        # Escape for debugging-purposes
        if debugmode and i > 3:
            break

    print(bar)


def get_time(printstring, tic):
    """
    This is a helper-function and prints the elapsed time taken from "tic" to present in a nice format
    :param printstring: String which will be displayed e.g.: 'Total Time' (note that ':' is not needed)
    :param tic: start time (needs to be taken as time.time())
    """
    elapsed = time.time() - tic  # elapsed time between tic and present
    m, s = divmod(elapsed, 60)  # get minutes and seconds by division and remainder
    h, m = divmod(m, 60)  # get hours and minutes

    # prepare nice numbers to print
    h = "0" + str(round(h)) if h < 10 else str(round(h))
    m = "0" + str(round(m)) if m < 10 else str(round(m))
    s = "0" + str(round(s)) if s < 10 else str(round(s))

    # print elapsed time and printstring:
    print("{}: {}:{}:{}".format(printstring, h, m, s))


#-----------------------------------------------------------------------------------------------------------------------
# THE FOLLOWING FUNCTIONS ARE NOT UP-TO-DATE --> THEY WERE INTENDED TO SPEED UP COMPUTATIONS BY PARALLEL COMPUTING
#-----------------------------------------------------------------------------------------------------------------------

#
# def export_exio_to_bw2_pool(path2exio, bw2_project, n_jobs=1):
#
#     impo = ExiobaseImporter(path2exio)
#     iot = impo.load_iot()
#     em = impo.load_emission()
#     cf_em = impo.load_emission_cfs()
#     res = impo.load_resources()
#     cf_res = impo.load_resources_cfs()
#
#     matchbiosphereflows = MatchBiosphereFlows()
#
#     dem = np.zeros(iot.shape[0])
#
#     met = ('Problem oriented approach: baseline (CML, 2001)', 'global warming (GWP100)', 'GWP100 (IPCC, 2007)', 'kg CO2 eq.')
#
#     pool = mp.Pool(processes=n_jobs)
#     [pool.apply(export_helper_for_pooling, args=(i, dem, bw2_project, iot, met, em, cf_em, res, cf_res, matchbiosphereflows)) for i, _ in enumerate(dem)]
#     pool.close()
#
#
# def export_helper_for_pooling(i, dem, bw2_project, iot, met, em, cf_em, res, cf_res, matchbiosphereflows):
#     name = iot.index[i]
#     if bw2_project in bw2.projects:
#         bw2.projects.set_current(bw2_project)
#
#         if 'EXIOBASE 2.2' in bw2.databases:
#             db = bw2.Database('EXIOBASE 2.2')
#
#             if name in [(a['location'], a['name']) for a in db]:
#                 return
#     name = "{}:{}".format(name[1], name[0])
#
#     demvec = dem.copy()
#     demvec[i] = 1
#
#     exio_lca = ExioLCA(iot, demvec, met, em, cf_em, res, cf_res)
#     exio_lca.lci()
#
#     Exio2BW2Exporter(exio_lca, name, bw2_project, matchbiosphereflows=matchbiosphereflows)



















