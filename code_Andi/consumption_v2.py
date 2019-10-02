# import psycopg2 as pg
# import psycopg2.extras as pge
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.cm as cm
import pandas as pd
# from matplotlib.backends.backend_pdf import PdfPages
import os
# from PyPDF2 import PdfFileMerger, PdfFileReader
# from sklearn import preprocessing
# from sklearn import linear_model
# from sklearn import model_selection
# from sklearn import ensemble
# from sklearn import neighbors
# from sklearn import decomposition
# from sklearn.cluster import KMeans, DBSCAN
# from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score, mean_squared_error
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, accuracy_score, classification_report
# # from sklearn.pipeline import make_pipeline
import xlrd
import copy
import time
# import sompy as smp
# from collections import Counter
import pickle
# import pyprind
import sys
# from scipy.cluster.hierarchy import cophenet
# from scipy.spatial.distance import pdist
# from scipy.cluster.hierarchy import inconsistent
# from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import csv
from scipy import stats
# from openpyxl import load_workbook
# import seaborn as sns
# sns.set(color_codes=True)
import brightway2 as bw2
# try:
#     from Clustering_Tools import *
# except:
#     sys.path.insert(0, r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\04_HEIA-Tools")
#     from Clustering_Tools import *
# from sompy.visualization.bmuhits import BmuHitsView
# from bokeh.io import output_file, save
# from bokeh.plotting import figure, show, ColumnDataSource
# from bokeh.models import LinearAxis, Range1d, HoverTool


class ConsumptionLCA(object):
    """
    This class performs LCA for all functional units of the process models which were defined to represent the consumption
    categories (see consumption-cockpit.xlsx). The idea is to speed up computations by providing "characterization"
    factors for each functional unit which can then be multiplied with the actual demand of consumption Archetypes or
    STATPOP-households.
    """
    def __init__(self, ei_lca, ex_lca, methods, excel, archetypes=False, **kwargs):
        """
        Init-function of the ConsumptionLCA-class
        :param ei_lca: Brightway2-LCA-class for the Brightway-project containing ecoinvent-based databases
        :param ex_lca: Brightway2-LCA-class for the Brightway-project containing EXIOBASE activities
        :param methods: dict with methods for which multiplication factors shall be computed.
        :param excel: dict with information of the excel-file in which the LCA-Modelling can be found. Keys needed are:
        'path' and 'sheet', optionally also 'wwtp', 'heating' if the class is applied to STATPOP-households and if municipality-
        specific scores for wastewater treatments shall be computed.
        :param archetypes: (optional) True/False to indicate if the class is applied to get multiplication factors for
        consumption archetypes of for STATPOP-households.
        """

        self.methods_ = methods

        # In case this class is applied for STATPOP-households, we will extract specific scores per municipality for
        # wastewater treatment plants, petrol car fleets and diesel car fleets.
        if not archetypes:
            self.wwtp_bfsnr_dict_ = self._get_wwtp_bfsnr_dict(excel)

            # Not yet implemented since not really necessary since we merge consumption model directly with MATSim-based mobility model
            #self.petrolfleet_bfsnr_dict_, self.dieselfleet_bfsnr_dict_ = self._get_cantonal_fleets()

            # Read the functional units or the process models respectively from the Excel-file
            self._read_fu_from_excel(excel, ei_lca=ei_lca)
        else:
            # Read the functional units or the process models respectively from the Excel-file
            self._read_fu_from_excel(excel, ei_lca=None)

        # The following attribute will collect all the functional units based on EXIOBASE. This attribute is actually
        # only used for documentation and possible post-analyses.
        self.exiobase_fus_ = []

        # The do_lca-function is called to compute the "characterization" factors.
        self.lca_fu_scores_ = self._do_lca(ei_lca, ex_lca)


    def _get_wwtp_bfsnr_dict(self, excel):
        """
        This function constructs a dict to translate municipality-numbers to the WWTP-size based on the information given
        in an excel-sheet.
        :param excel: Dict with information where to find information on the WWTP-sizes. Keys needed: 'path' and 'wwtp'.
        The latter shall incidate the name of the sheet.
        :return: A translation-dict containing the municipality BFSNR as a key and the WWTP-size as a value
        """

        # Open the excel-file
        ws = xlrd.open_workbook(excel['path']).sheet_by_name(excel['wwtp'])

        # Determine the start for reading the information:
        ara_col = ws.col_values(0)
        startread = ara_col.index('py_start_read') + 2

        # Core: read the information and construct the dict:
        wwtp_dict = {}
        for bfsnr, ara in zip(ws.col_values(5)[startread:], ara_col[startread:]):
            bfsnr = int(bfsnr)
            ara = int(ara)

            # There are municipalities with more than one WWTP. Therefore, we only consider the largest size (this makes
            # sense in view of the fact that the "probability" is larger for a certain HH to be connected to the larger
            # plant). Please note that large WWTPs correspond to a small ARA class-value.
            if bfsnr in wwtp_dict.keys():
                if wwtp_dict[bfsnr] > ara:
                    wwtp_dict[bfsnr] = ara
                else:
                    continue
            else:
                wwtp_dict[bfsnr] = ara

        return wwtp_dict


    def _get_cantonal_fleets(self):
        """
        This function constructs a dict which translates municipal numbers to the respective cantonal fleet. It returns
        separate dicts for the diesel cars and the petrol cars.
        :return: two translation-dicts containing the municipality BFSNR as a key and the cantonal diesel car fleet as
        well as the cantonal petrol car fleet as values.
        """

        # In a first step we get the complete cantonal fleets for all municipalities:
        conn = get_pg_connection()
        cur = conn.cursor(cursor_factory=pge.RealDictCursor)
        cur.execute("""
                SELECT * FROM working_tables.boundaries_gde geo LEFT JOIN working_tables.car_fleets_kt cars
                ON geo.kantonsnum = cars.kantonsnum WHERE geo.bfs_nummer IN (SELECT DISTINCT bfs_nummer FROM
                working_tables.hh)
                """)
        allfleets = cur.fetchall()
        cur.close()
        conn.close()

        # We then split the fleets into petrol and diesel fleets:
        petrolfleets = self._get_cantonal_fuelfleets(allfleets, 'petrol')
        dieselfleets = self._get_cantonal_fuelfleets(allfleets, 'diesel')

        return petrolfleets, dieselfleets


    def _get_cantonal_fuelfleets(self, allfleets, fuel):
        """
        Helper function which recomputes the cantonal fleet specifically for petrol or diesel
        :param allfleets: list of dicts with all cantonal fleets for all municipalities (obtained by
        _get_cantonal_fleets())
        :param fuel: fuel type for which the fleets need to be recomputed (usually just 'diesel' or 'petrol')
        :return: dict which translates the municipality number to the specific cantonal car fleet
        """

        # First: extract all the relevant information for the fuel-type-carfleet:
        fuelfleet = {x['bfs_nummer']: {ky: x[ky] for ky in x.keys() if fuel in ky} for x in allfleets}

        # Then go through all muncipalities and recompute the car-fleet shares:
        for bfsnr in fuelfleet.keys():

            # Get the total sum of the shares
            totshare = 0
            for ky in fuelfleet[bfsnr].keys():
                totshare += fuelfleet[bfsnr][ky]

            # Compute the new share:
            for ky in fuelfleet[bfsnr].keys():
                fuelfleet[bfsnr][ky] /= totshare

        return fuelfleet


    def _read_fu_from_excel(self, excel, ei_lca=None):
        """
        This function reads the functional unit of the process models which can be found in the excel specified
        (usually Consumption_Cockpit.xlsx in sheet LCA-Modelling).
        :param excel: dict with information in which excelfile the LCA-Modelling can be found (see init-function)
        :param ei_lca: (optional) brightway2-LCA-class (see init-function) -> passing this parameter means also that
        the specific multiplication factors for WWTP, heating and car fleets for each municipality shall be computed (or in other
        words: passing this LCA-class means that the ConsumptionLCA-Cockpit shall prepare the factors for STATPOP-
        households)
        """

        # Open the worksheet with the LCA-Modelling
        ws = xlrd.open_workbook(excel['path']).sheet_by_name(excel['sheet'])

        # Read the header row and determine the column which indicates if a certain attribute shall be included in LCA
        # or not:
        header = ws.row_values(0)
        role = ws.col_values(header.index('Role'))

        # Cut the header row to the relevant information
        heads = header[header.index('On 1'):len(header)]

        # Prepare a dict which will contain the demand-to-functional-unit conversion factors of the process models as
        # well as a list of attributes which shows for which attributes LCA-Models exist.
        self.attributes_dict_ = {}
        self.attributes_list_ = []

        # Go through all consumption categories
        for i, conscat in enumerate(role):

            # if the role-value is 0, this means an LCA shall be performed for the consumption category under consideration
            if conscat == 0:

                # Test if an amount-category is available for the consumption category
                if ws.cell_value(i, 1) != '':
                    attr = ws.cell_value(i, 1).lower()
                else:
                    attr = ws.cell_value(i, 0).lower()

                # Fill in the attributes-list as well as the dict containing the conversion factors.
                self.attributes_dict_.update({attr: ws.cell_value(i, 3)})
                self.attributes_list_.append(attr)

                nm = attr + "_fu_"

                # Cut out the row for the modelled activity and trim it to the relevant information
                acts = ws.row_values(i)[header.index('On 1'):len(header)]

                # Core of the function: read the different unit processes, multiply it already with the CFL-factor which
                # comes from Scherer & Pfister to convert the functional unit of the unit process to the product under
                # consideration --> store the functional unit in a dict and then as an attribute of the function
                fu_dict = {}
                for j, head in enumerate(heads):
                    # jump to the 'On'-cell --> if it is =1 then the subsequent cells shall be included in the LCA
                    if 'On' in head:
                        if acts[j] == 1:
                            # Check if there is a demand:
                            if acts[j + 3] * acts[j + 4]:

                                # Construct the correct activity name:
                                actkey = acts[j + 1].replace('(', '').replace(')', '').replace("'", '').split(',')
                                actkey = (actkey[0], ','.join(actkey[1:])[1:])

                                # There is a possibility that the same unit process is used several times within the
                                # same process model:
                                if actkey not in fu_dict.keys():
                                    fu_dict.update({actkey: acts[j + 3] * acts[j + 4]})
                                else:
                                    new_val = fu_dict[actkey] + (acts[j + 3] * acts[j + 4])
                                    fu_dict.update({actkey: new_val})

                setattr(self, nm, fu_dict)

                # In case we want to derive municipality-specific scores for WWTP and car fleets (this is the case if
                # the class shall be applied for STATPOP-households, but not for consumption archetypes), then we enter
                # the following if-loop.
                if ei_lca:
                    if attr == 'mx571203':
                        self.wwtp_lca_dict_ = self._get_wwtp_lca_dict(heads, acts, ei_lca)
                    if attr == 'mx571302':
                        self.heating_lca_dict_ = self._get_heating_lca_dict(excel, ei_lca)
                    # Not implemented yet since not really necessary because we merge consumption model and MATSim-based
                    # mobility model directly
                    # if attr == 'm621501':
                    #     self.petrolcar_lca_dict_ = self._get_petrolcar_lca_dict()
                    # if attr == 'm621501':
                    #     self.dieselcar_lca_dict_ = self._get_dieselcar_lca_dict()


    def _get_wwtp_lca_dict(self, heads, acts, lca):
        """
        This function computes LCA scores for the different WWTP-sizes. This information can be used in combination with
        the translation dict obtained in _get_wwtp_bfsnr_dict to compute municipality-specific LCA-WWTP-scores. The
        function is called within  _read_fu_from_excel().
        :param heads: The list of headers in the excel file relevant for the LCA-modelling
        :param acts: The list of excel-cells which contains the WWTP-information
        :param lca: Brightway2-LCA-class (see init function)
        :return: Dict with WWTP-class size as keys and LCA-score as value.
        """

        wwtp_lca_dict = {}
        i = 0
        bw2.projects.set_current('heia33')

        # Go through all headers, jump to 'On'-header:
        for j, head in enumerate(heads):
            if 'On' in head:
                # We now want to do LCA for all the WWTPs. Therefore, we also need activities for which the 'On'-header
                # is 0 (but of course we want to exclude empty cells)
                if acts[j] == 1 or acts[j] == 0:
                    # i determines the class of the WWTP --> IMPORTANT: the order in the acts-list needs to be correct
                    i += 1

                    # Construct the correct key for the activity
                    actkey = acts[j + 1].replace('(', '').replace(')', '').replace("'", '').split(',')
                    actkey = (actkey[0], ','.join(actkey[1:])[1:])

                    # Go through all methods an do the LCA for the WWTPs
                    for methodkey in self.methods_.keys():
                        lca.switch_method(self.methods_[methodkey])
                        lca.redo_lcia({actkey: 1})
                        wwtp_lca_dict.update({"{}_{}".format(i, methodkey): lca.score})

        return wwtp_lca_dict


    def _get_heating_lca_dict(self, excel, lca):
        """
        This function constructs a dict to translate GWS-code which indicates the concrete energy carrier for heating
        to concrete heating activities. This allows for STATPOP-HHs to use the correct activities for the specific case of
        an individual household. The function is called within  _read_fu_from_excel().
        :param excel: Dict with information where to find information on the GWS-energy carriers. Keys needed: 'path' and 'heating'.
        The latter shall incidate the name of the sheet.
        :return: A translation-dict containing the LCA-FU-score for the different GWS-energy carriers
        """

        bw2.projects.set_current('heia33')

        # Open the excel-file
        ws = xlrd.open_workbook(excel['path']).sheet_by_name(excel['heating'])

        # Determine the start for reading the information:
        gws_col = ws.col_values(2)
        header = ws.row_values(0)

        # Cut the header row to the relevant information
        heads = header[header.index('On 1'):len(header)]

        # Core: read the information and construct the dict:
        heating_dict = {}

        # Go through all GWS-codes
        for i, gws in enumerate(gws_col[1:]):
            # The following if-statement shall break the for-loop in case we reach the last GWS-code-entry
            if gws == '':
                break

            # Read the activities-row
            acts = ws.row_values(i + 1)[header.index('On 1'):len(header)]

            fu_dict = {}
            for j, head in enumerate(heads):
                # jump to the 'On'-cell --> if it is =1 then the subsequent cells shall be included in the LCA
                if 'On' in head:
                    if acts[j] == 1:
                        # Check if there is a demand:
                        if acts[j + 3] * acts[j + 4]:

                            # Construct the correct activity name:
                            actkey = acts[j + 1].replace('(', '').replace(')', '').replace("'", '').split(',')
                            actkey = (actkey[0], ','.join(actkey[1:])[1:])

                            # There is a possibility that the same unit process is used several times within the
                            # same process model:
                            if actkey not in fu_dict.keys():
                                fu_dict.update({actkey: acts[j + 3] * acts[j + 4]})
                            else:
                                new_val = fu_dict[actkey] + (acts[j + 3] * acts[j + 4])
                                fu_dict.update({actkey: new_val})

            # We then directly do LCA for the FU under consideration and for all methods
            for methodkey in self.methods_.keys():
                lca.switch_method(self.methods_[methodkey])
                lca.redo_lcia(fu_dict)
                heating_dict.update({"{}_{}".format(int(gws), methodkey): lca.score * ws.cell_value(i + 1, 3)})

        return heating_dict


    def _get_petrolcar_lca_dict(self):
        raise NotImplementedError


    def _get_dieselcar_lca_dict(self):
        raise NotImplementedError


    def _do_lca(self, ei_lca, ex_lca):
        """
        This is the core-function of the class and performs the actual LCA of the functional units derived in
        _read_fu_from_excel().
        :param ei_lca: Brightway2-LCA-class for the Brightway-project containing ecoinvent-based databases
        :param ex_lca: Brightway2-LCA-class for the Brightway-project containing EXIOBASE activities
        :return: dict with all the LCA-scores for the functional units.
        """

        # Prepare a results-dict-container for the LCA-scores
        lca_fu_scores = {}

        # Go through all processes:
        funits = [attr for attr in dir(self) if attr.endswith('_fu_')]
        for fu in funits:

            # In the following we test if the consumption category was modelled by EXIOBASE or not --> depending on
            # this test we set the brightway2-project accordingly and take the correct brightway2-LCA-class
            ex_test_list = [x[0] for x in getattr(self, fu).keys()]
            if any('EXIOBASE' in ky for ky in ex_test_list):
                bw2.projects.set_current('exiobase_industry_workaround')
                lca = ex_lca
                # For documentation purposes we also list the EXIOBASE-functional units
                self.exiobase_fus_.append(fu)
            else:
                bw2.projects.set_current('heia33')
                lca = ei_lca

            # For a certain process: go through all methods which shall be assessed and do the lca-computations
            for methodkey in self.methods_.keys():
                lca.switch_method(self.methods_[methodkey])
                lca.redo_lcia(getattr(self, fu))  # with gettattr: get the attribute value of "fu" dynamically

                # in the following, a keyname is constructed and the lca-score is saved to a dict
                name = fu.split('_')
                name = '_'.join(name[0:-1])
                lca_fu_scores.update({'_'.join([name, methodkey]): lca.score})

        return lca_fu_scores


