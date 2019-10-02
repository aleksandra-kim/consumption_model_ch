from Andi_model import consumption as cons
import time
import pickle
import numpy as np
import multiprocessing as mp
from queue import Empty
import logging
import os


#----------------------------------------------------------------------------------------------------------------------
# Data Pre-Processing: NK-Modelling
#----------------------------------------------------------------------------------------------------------------------

#****************************
# "Nebenkosten pauschal": Model-Selection
#****************************
#
# # 1st STEP: Loading data
#
# tic = time.time()
# print("loading data...")
#
# nk_ky = 'NK'
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
# with open(os.path.join(datapath, "dict_of_data.pickle"), 'rb') as f:
#     dict_of_data = pickle.load(f)
#
# cons.get_time("Time for loading data", tic) # last run: 00:00:01
#
# # 2nd STEP: Training of data / tuning hyperparameters
# t_mod = time.time()
# print("training regression-models...")
#
# rf_df = pd.DataFrame(data=np.zeros((14, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'r2_oob', 'noTrees', 'pred_mean', 'pred_min', 'pred_max','orig_mean', 'orig_min', 'orig_max',
#                                '30trees', '100trees', '300trees', '500trees'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("RF-Regression (scaled): {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     rf_nk = cons.RF_NK(dict_of_data[ky][0], dict_of_data[ky][1], rf_df, ky)
#     with open(os.path.join(resultpath, "{}_{}_RF_scaled_Class_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(rf_nk, f)
#     cons.nk_model_basic_diagnostic_plot(rf_nk, resultpath, "{}_{}_RF_scaled_basic_diagnostic_plot".format(nk_ky, ky))
# savepath = os.path.join(resultpath, 'NK_RF-Results_scaled_Modelselection.xlsx')
# rf_df.to_excel(savepath, sheet_name='RF-Results_scaled', na_rep='-')
#
# rf2_df = pd.DataFrame(data=np.zeros((14, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'r2_oob', 'noTrees', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max',
#                                '30trees', '100trees', '300trees', '500trees'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("RF-Regression (not scaled): {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     rf2_nk = cons.RF_NK(dict_of_data[ky][0], dict_of_data[ky][1], rf2_df, ky, scaling=False)
#     with open(os.path.join(resultpath, "{}_{}_RF_notscaled_Class_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(rf2_nk, f)
#     cons.nk_model_basic_diagnostic_plot(rf2_nk, resultpath, "{}_{}_RF_notscaled_basic_diagnostic_plot".format(nk_ky, ky))
# savepath = os.path.join(resultpath, 'NK_RF-Results_notscaled_Modelselection.xlsx')
# rf2_df.to_excel(savepath, sheet_name='RF-Results_notscaled', na_rep='-')
#
# cons.get_time("time for training models", t_mod) # last run: 00:52:54 (only not scaled RF)
#
# cons.get_time("Total time", tic) # last run: 00:52:55 (only not scaled RF)

#****************************
# "Kehricht": Model-Selection
#****************************
#
# # 1st STEP: Loading data
#
# tic = time.time()
# print("loading data...")
# nk_ky = 'K'
#
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
# with open(os.path.join(datapath, "dict_of_data_kehricht.pickle"), 'rb') as f:
#     dict_of_data = pickle.load(f)
#
# cons.get_time("Time for loading data", tic) # last run: 00:00:00
#
# # 2nd STEP: Training of data / tuning hyperparameters
# t_mod = time.time()
# print("training regression-models...")
#
# rf_df = pd.DataFrame(data=np.zeros((14, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'r2_oob', 'noTrees', 'pred_mean', 'pred_min', 'pred_max','orig_mean', 'orig_min', 'orig_max',
#                                '30trees', '100trees', '300trees', '500trees'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("RF-Regression (scaled): {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     rf_nk = cons.RF_NK(dict_of_data[ky][0], dict_of_data[ky][1], rf_df, ky)
#     with open(os.path.join(resultpath, "{}_{}_RF_scaled_Class_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(rf_nk, f)
#     cons.nk_model_basic_diagnostic_plot(rf_nk, resultpath, "{}_{}_RF_scaled_basic_diagnostic_plot".format(nk_ky, ky))
#
# savepath = os.path.join(resultpath, '{}_RF-Results_scaled_Modelselection.xlsx'.format(nk_ky))
# rf_df.to_excel(savepath, sheet_name='RF-Results_scaled', na_rep='-')
#
# rf2_df = pd.DataFrame(data=np.zeros((14, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'r2_oob', 'noTrees', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max',
#                                '30trees', '100trees', '300trees', '500trees'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("RF-Regression (not scaled): {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     rf2_nk = cons.RF_NK(dict_of_data[ky][0], dict_of_data[ky][1], rf2_df, ky, scaling=False)
#     with open(os.path.join(resultpath, "{}_{}_RF_notscaled_Class_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(rf2_nk, f)
#     cons.nk_model_basic_diagnostic_plot(rf2_nk, resultpath, "{}_{}_RF_notscaled_basic_diagnostic_plot".format(nk_ky, ky))
#
# savepath = os.path.join(resultpath, '{}_RF-Results_notscaled_Modelselection.xlsx'.format(nk_ky))
# rf2_df.to_excel(savepath, sheet_name='RF-Results_notscaled', na_rep='-')
#
# cons.get_time("time for training models", t_mod) # last run: 01:18:05
#
# cons.get_time("Total time", tic) # last run: 01:18:05

# ****************************
# "Electricity": Model-Selection
# ****************************

# 1st STEP: Loading data

# tic = time.time()
# print("loading data...")
# nk_ky = 'El'
#
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
# with open(os.path.join(datapath, "dict_of_data_electricity.pickle"), 'rb') as f:
#     dict_of_data = pickle.load(f)
#
# cons.get_time("Time for loading data", tic) # last run: 00:00:00
#
# # 2nd STEP: Training of data / tuning hyperparameters
# t_mod = time.time()
# print("training regression-models...")
#
# rf_df = pd.DataFrame(data=np.zeros((14, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'r2_oob', 'noTrees', 'pred_mean', 'pred_min', 'pred_max','orig_mean', 'orig_min', 'orig_max',
#                                '30trees', '100trees', '300trees', '500trees'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("RF-Regression (scaled): {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     rf_nk = cons.RF_NK(dict_of_data[ky][0], dict_of_data[ky][1], rf_df, ky)
#     with open(os.path.join(resultpath, "{}_{}_RF_scaled_Class_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(rf_nk, f)
#     cons.nk_model_basic_diagnostic_plot(rf_nk, resultpath, "{}_{}_RF_scaled_basic_diagnostic_plot".format(nk_ky, ky))
#
# savepath = os.path.join(resultpath, '{}_RF-Results_scaled_Modelselection.xlsx'.format(nk_ky))
# rf_df.to_excel(savepath, sheet_name='RF-Results_scaled', na_rep='-')
#
# rf2_df = pd.DataFrame(data=np.zeros((14, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'r2_oob', 'noTrees', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max',
#                                '30trees', '100trees', '300trees', '500trees'])
# for i, ky in enumerate(dict_of_data.keys()):
#     print("RF-Regression (not scaled): {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     rf2_nk = cons.RF_NK(dict_of_data[ky][0], dict_of_data[ky][1], rf2_df, ky, scaling=False)
#     with open(os.path.join(resultpath, "{}_{}_RF_notscaled_Class_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(rf2_nk, f)
#     cons.nk_model_basic_diagnostic_plot(rf2_nk, resultpath, "{}_{}_RF_notscaled_basic_diagnostic_plot".format(nk_ky, ky))
#
# savepath = os.path.join(resultpath, '{}_RF-Results_notscaled_Modelselection.xlsx'.format(nk_ky))
# rf2_df.to_excel(savepath, sheet_name='RF-Results_notscaled', na_rep='-')
#
# cons.get_time("time for training models", t_mod) # last run: 01:59:39
#
# cons.get_time("Total time", tic) # last run: 01:59:40

#****************************
# "Heating": Model-Selection
#****************************

# # 1st STEP: Loading data
#
# tic = time.time()
# print("loading data...")
# nk_ky = 'EnBr'
#
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
# with open(os.path.join(datapath, "dict_of_data_heating.pickle"), 'rb') as f:
#     dict_of_data = pickle.load(f)
#
# cons.get_time("Time for loading data", tic) # last run: 00:00:00
#
# # 2nd STEP: Training of data / tuning hyperparameters
# t_mod = time.time()
# print("training regression-models...")
#
# rf_df = pd.DataFrame(data=np.zeros((14, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'r2_oob', 'noTrees', 'pred_mean', 'pred_min', 'pred_max','orig_mean', 'orig_min', 'orig_max',
#                                '30trees', '100trees', '300trees', '500trees'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("RF-Regression (scaled): {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     rf_nk = cons.RF_NK(dict_of_data[ky][0], dict_of_data[ky][1], rf_df, ky)
#     with open(os.path.join(resultpath, "{}_{}_RF_scaled_Class_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(rf_nk, f)
#     cons.nk_model_basic_diagnostic_plot(rf_nk, resultpath, "{}_{}_RF_scaled_basic_diagnostic_plot".format(nk_ky, ky))
#
# savepath = os.path.join(resultpath, '{}_RF-Results_scaled_Modelselection.xlsx'.format(nk_ky))
# rf_df.to_excel(savepath, sheet_name='RF-Results_scaled', na_rep='-')
#
# rf2_df = pd.DataFrame(data=np.zeros((14, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'r2_oob', 'noTrees', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max',
#                                '30trees', '100trees', '300trees', '500trees'])
# for i, ky in enumerate(dict_of_data.keys()):
#     print("RF-Regression (not scaled): {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     rf2_nk = cons.RF_NK(dict_of_data[ky][0], dict_of_data[ky][1], rf2_df, ky, scaling=False)
#     with open(os.path.join(resultpath, "{}_{}_RF_notscaled_Class_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(rf2_nk, f)
#     cons.nk_model_basic_diagnostic_plot(rf2_nk, resultpath, "{}_{}_RF_notscaled_basic_diagnostic_plot".format(nk_ky, ky))
#
# savepath = os.path.join(resultpath, '{}_RF-Results_notscaled_Modelselection.xlsx'.format(nk_ky))
# rf2_df.to_excel(savepath, sheet_name='RF-Results_notscaled', na_rep='-')
#
# cons.get_time("time for training models", t_mod) # last run: 00:48:32
#
# cons.get_time("Total time", tic) # last run: 00:48:33

#****************************
# "Water": Model-Selection
#****************************

# # 1st STEP: Loading data
#
# tic = time.time()
# print("loading data...")
# nk_ky = 'W'
#
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
# with open(os.path.join(datapath, "dict_of_data_water.pickle"), 'rb') as f:
#     dict_of_data = pickle.load(f)
#
# cons.get_time("Time for loading data", tic) # last run: 00:00:01
#
# # 2nd STEP: Training of data / tuning hyperparameters
# t_mod = time.time()
# print("training regression-models...")
#
# rf_df = pd.DataFrame(data=np.zeros((14, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'r2_oob', 'noTrees', 'pred_mean', 'pred_min', 'pred_max','orig_mean', 'orig_min', 'orig_max',
#                                '30trees', '100trees', '300trees', '500trees'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("RF-Regression (scaled): {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     rf_nk = cons.RF_NK(dict_of_data[ky][0], dict_of_data[ky][1], rf_df, ky)
#     with open(os.path.join(resultpath, "{}_{}_RF_scaled_Class_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(rf_nk, f)
#     cons.nk_model_basic_diagnostic_plot(rf_nk, resultpath, "{}_{}_RF_scaled_basic_diagnostic_plot".format(nk_ky, ky))
#
# savepath = os.path.join(resultpath, '{}_RF-Results_scaled_Modelselection.xlsx'.format(nk_ky))
# rf_df.to_excel(savepath, sheet_name='RF-Results_scaled', na_rep='-')
#
# rf2_df = pd.DataFrame(data=np.zeros((14, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'r2_oob', 'noTrees', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max',
#                                '30trees', '100trees', '300trees', '500trees'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("RF-Regression (not scaled): {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     rf2_nk = cons.RF_NK(dict_of_data[ky][0], dict_of_data[ky][1], rf2_df, ky, scaling=False)
#     with open(os.path.join(resultpath, "{}_{}_RF_notscaled_Class_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(rf2_nk, f)
#     cons.nk_model_basic_diagnostic_plot(rf2_nk, resultpath, "{}_{}_RF_notscaled_basic_diagnostic_plot".format(nk_ky, ky))
#
# savepath = os.path.join(resultpath, '{}_RF-Results_notscaled_Modelselection.xlsx'.format(nk_ky))
# rf2_df.to_excel(savepath, sheet_name='RF-Results_notscaled', na_rep='-')
#
# cons.get_time("time for training models", t_mod) # last run: 00:40:33
#
# cons.get_time("Total time", tic) # last run: 00:40:34

#****************************
# "Wastewater": Model-Selection
#****************************
#
# # 1st STEP: Loading data
#
# tic = time.time()
# print("loading data...")
# nk_ky = 'AW'
#
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
# with open(os.path.join(datapath, "dict_of_data_ww.pickle"), 'rb') as f:
#     dict_of_data = pickle.load(f)
#
# cons.get_time("Time for loading data", tic) # last run: 00:00:00
#
# # 2nd STEP: Training of data / tuning hyperparameters
# t_mod = time.time()
# print("training regression-models...")
#
# rf_df = pd.DataFrame(data=np.zeros((14, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'r2_oob', 'noTrees', 'pred_mean', 'pred_min', 'pred_max','orig_mean', 'orig_min', 'orig_max',
#                                '30trees', '100trees', '300trees', '500trees'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("RF-Regression (scaled): {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     rf_nk = cons.RF_NK(dict_of_data[ky][0], dict_of_data[ky][1], rf_df, ky)
#     with open(os.path.join(resultpath, "{}_{}_RF_scaled_Class_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(rf_nk, f)
#     cons.nk_model_basic_diagnostic_plot(rf_nk, resultpath, "{}_{}_RF_scaled_basic_diagnostic_plot".format(nk_ky, ky))
#
# savepath = os.path.join(resultpath, '{}_RF-Results_scaled_Modelselection.xlsx'.format(nk_ky))
# rf_df.to_excel(savepath, sheet_name='RF-Results_scaled', na_rep='-')
#
# rf2_df = pd.DataFrame(data=np.zeros((14, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'r2_oob', 'noTrees', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max',
#                                '30trees', '100trees', '300trees', '500trees'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("RF-Regression (not scaled): {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     rf2_nk = cons.RF_NK(dict_of_data[ky][0], dict_of_data[ky][1], rf2_df, ky, scaling=False)
#     with open(os.path.join(resultpath, "{}_{}_RF_notscaled_Class_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(rf2_nk, f)
#     cons.nk_model_basic_diagnostic_plot(rf2_nk, resultpath, "{}_{}_RF_notscaled_basic_diagnostic_plot".format(nk_ky, ky))
#
# savepath = os.path.join(resultpath, '{}_RF-Results_notscaled_Modelselection.xlsx'.format(nk_ky))
# rf2_df.to_excel(savepath, sheet_name='RF-Results_notscaled', na_rep='-')
#
# cons.get_time("time for training models", t_mod) # last run: 00:39:16
#
# cons.get_time("Total time", tic) # last run: 00:39:17

#----------------------------------------------------------------------------------------------------------------------
# Filtering features for Self-Organizing Map
#----------------------------------------------------------------------------------------------------------------------

# #****************************
# # Compute feature importance scores (here only Random Forest)
# #****************************
# tic=time.time()
# print("load data...")
#
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
# print("RF: not correlated dataset...")
# t_notcorr = time.time()
#
# with open(os.path.join(datapath, "filtered_data_for_som_notcorr.pickle"), 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
#
# rf_fi_notcorr = cons.RF_FI(filtered_data_for_som, title='not_corr', rfmse=True,
#                     predictors=filtered_data_for_som.attributes_,
#                     targets=[t for t in filtered_data_for_som.attributes_ if t.startswith('a') or t.startswith('cg') or t.startswith('m')])
# cons.get_time("Time for not correlated dataset", t_notcorr)  # last run: 22:41:07
#
# try:
#     del rf_fi_notcorr.data_._conn_
# except:
#     pass
#
# with open(os.path.join(resultpath, "rf_fi_notcorr.pickle"), 'wb') as f:
#     pickle.dump(rf_fi_notcorr, f)
#
# rf_fi_notcorr.save2excel(os.path.join(resultpath, "rf_fi_notcorr.xlsx"))
#
# attrslist = [a for a in dir(rf_fi_notcorr) if a.endswith('_') and not a.startswith('__')]
# for a in attrslist:
#     if not a in ('data_', 'rf_scores_', 'meta_', 'predictors_', 'rfmse_scores_', 'targets_', 'title_'):
#         delattr(rf_fi_notcorr, a)
#
# with open(os.path.join(resultpath, "rf_fi_notcorr_light.pickle"), 'wb') as f:
#     pickle.dump(rf_fi_notcorr, f)
#
# print("RF: full dataset...")
# t_full = time.time()
#
# with open(os.path.join(datapath, "filtered_data_for_som_full.pickle"), 'rb') as f:
#     filtered_data_for_som_full = pickle.load(f)
#
# rf_fi_full = cons.RF_FI(filtered_data_for_som_full, title='full', rfmse=True,
#                     predictors=filtered_data_for_som_full.attributes_,
#                     targets=[t for t in filtered_data_for_som_full.attributes_ if t.startswith('a') or t.startswith('cg') or t.startswith('m')])
# cons.get_time('Time for full dataset', t_full)  # last run: 124:23:57
#
# try:
#     del rf_fi_full.data_._conn_
# except:
#     pass
#
# with open(os.path.join(resultpath, "rf_fi_full.pickle"), 'wb') as f:
#     pickle.dump(rf_fi_full, f)
#
# rf_fi_full.save2excel(os.path.join(resultpath, "rf_fi_full.xlsx"))
#
# attrslist = [a for a in dir(rf_fi_full) if a.endswith('_') and not a.startswith('__')]
# for a in attrslist:
#     if not a in ('data_', 'rf_scores_', 'meta_', 'predictors_', 'rfmse_scores_', 'targets_', 'title_'):
#         delattr(rf_fi_full, a)
#
# with open(os.path.join(resultpath, "rf_fi_full_light.pickle"), 'wb') as f:
#     pickle.dump(rf_fi_full, f)
#
# cons.get_time('time for FI-computing', tic)  # last run: 147:23:06


#----------------------------------------------------------------------------------------------------------------------
# Tuning Self-Organizing Maps --> PART I: not-correlated/coupled dataset
#----------------------------------------------------------------------------------------------------------------------

# # PRINT from consumption_cockpit
# # no. of neurons: 493.3051793768235
# # ratio of side lengths: 2.2506451553657376
# # length 1: 14.804860638896777
# # length 2: 33.32048787279793
# # definite lengths: 15 : 33
# # no of neurons: 495
#
#
# # IMPORTANT: in the sompy-package, I found a bug how they compute the topographic error the following function is a copy/paste
# # from the original function in sompy, but it uses the scaled data rather than the unscaled (which is wrong!)
# def get_topographic_error(som):
#     bmus1 = som.find_bmu(som._data, njb=1, nth=1)
#     bmus2 = som.find_bmu(som._data, njb=1, nth=2)
#     bmus_gap = np.abs(
#         (som.bmu_ind_to_xy(np.array(bmus1[0]))[:, 0:2] - som.bmu_ind_to_xy(np.array(bmus2[0]))[:, 0:2]).sum(axis=1))
#     return np.mean(bmus_gap != 1)

# #****************************
# # Tuning the neighborhood radius
# #****************************
#
# tic = time.time()
# print("Initialization/Loading data...")
#
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
#
# with open(os.path.join(datapath, 'filtered_data_for_som_notcorr.pickle'), 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# # SOM-Parameters fixed:
# mapform = 'planar'
# mapsize = [15, 33]
# lattice = 'rect'
# normalization = 'var'
# initialization = 'pca'
# neighborhood = 'gaussian'
# training = 'batch'
# compnames = ["Q {}".format(filtered_data_for_som.varnames_[c]) if c.startswith('m') else filtered_data_for_som.varnames_[c] for c in filtered_data_for_som.attributes_]
#
# # Training parameters fixed:
# train_rough_len = 990  # recommendation of SOM-Toolbox (Kohonen took too long): total no. of epochs: 10 x map units --> (10 x 495)/5 divided by 5 to have it 4 times shorter than finetuning
# train_finetune_len = 3960  # recommendation of SOM-Toolbox (Kohonen took too long) (10 x 495)/ 5 x 4
#
# # ATTENTION: in a first try, we already investigated if we should set final radius to 1 or to 0. Obviously, the current
# # implementation of SOMPY doesn't care --> it doesn't matter if 0 or 1 --> with both we get the exact same results and
# # look at only the BMU in the final epochs.
# radii = {'def': (0, 0, 0),
#          'bestrec': (17, 4, 1),
#          'halfdia1': (17, 1, 1),
#          '34thdia4th': (25, 6, 1),
#          '34thdia1': (25, 1, 1),
#          '14thdia4th': (8, 2, 1),
#          '14thdia1': (8, 1, 1)
#          }
#
# rad_soms_df = pd.DataFrame(data=np.zeros((5, len(radii)), dtype=float),
#                         columns=[ky for ky in radii.keys()],
#                         index=['rough_radius_ini', 'rough_radius_final', 'fine_radius_final',
#                                'topographic error', 'quantization error'])
#
# cons.get_time("Time for Initialization", tic)
# t_som = time.time()
#
# for i, ky in enumerate(radii.keys()):
#     t_som_temp = time.time()
#     print("SOM-Variant {}: {} of {}".format(ky, i+1, len(radii)))
#     if ky == 'def':
#         som = smp.SOMFactory.build(filtered_data_for_som.data_, mask=None, mapsize=mapsize, mapshape=mapform, lattice=lattice,
#                                    normalization=normalization, initialization=initialization, neighborhood=neighborhood,
#                                    training=training, name=ky, component_names=compnames)
#         som.train(n_job=1, verbose=None)
#
#         topographic_error = get_topographic_error(som)
#         quantization_error = np.mean(som._bmu[1])
#
#         metalist = [np.nan, np.nan, np.nan, topographic_error, quantization_error]
#         rad_soms_df[ky] = metalist
#
#         with open(os.path.join(resultpath, "som_{}.pickle".format(ky)), 'wb') as f:
#             pickle.dump(som, f)
#     else:
#         som = smp.SOMFactory.build(filtered_data_for_som.data_, mask=None, mapsize=mapsize, mapshape=mapform, lattice=lattice,
#                                    normalization=normalization, initialization=initialization, neighborhood=neighborhood,
#                                    training=training, name=ky, component_names=compnames)
#         som.train(n_job=1, verbose='info', train_rough_len=train_rough_len, train_rough_radiusin=radii[ky][0],
#                   train_rough_radiusfin=radii[ky][1], train_finetune_len=train_finetune_len,
#                   train_finetune_radiusin=radii[ky][1], train_finetune_radiusfin=radii[ky][2])
#         topographic_error = get_topographic_error(som)
#         quantization_error = np.mean(som._bmu[1])
#
#         metalist = [radii[ky][0], radii[ky][1], radii[ky][2], topographic_error, quantization_error]
#         rad_soms_df[ky] = metalist
#
#         with open(os.path.join(resultpath, "som_{}.pickle".format(ky)), 'wb') as f:
#             pickle.dump(som, f)
#     cons.get_time("time for {}".format(ky), t_som_temp)
#
# cons.get_time("Time for all SOMs", t_som)  # last run: 03:05:10, per SOM about 30 minutes
# savepath = os.path.join(resultpath, 'SOM_Radius_Tuning.xlsx')
# rad_soms_df.to_excel(savepath, sheet_name='tuning radii', na_rep='-')

#****************************
# Tuning the no of epochs
#****************************

# tic = time.time()
# print("Initialization/Loading data...")
#
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
# with open(os.path.join(datapath, 'filtered_data_for_som_notcorr.pickle'), 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# # SOM-Parameters fixed:
# mapform = 'planar'
# mapsize = [15, 33]
# lattice = 'rect'
# normalization = 'var'
# initialization = 'pca'
# neighborhood = 'gaussian'
# training = 'batch'
# compnames = ["Q {}".format(filtered_data_for_som.varnames_[c]) if c.startswith('m') else filtered_data_for_som.varnames_[c] for c in filtered_data_for_som.attributes_]
#
# # Training parameters fixed:
# train_rough_radiusin = 17
# train_rough_radiusfin = 4
# train_finetune_radiusin = 4
# train_finetune_radiusfin = 1
#
# train_len = {'10_10': (450, 4500),
#          '10_4': (990, 3960),
#          '10_0.1': (4500, 450),
#          '10_0.25': (3960, 990),
#          '500_10': (22500, 225000)
#          }
#
# epo_soms_df = pd.DataFrame(data=np.zeros((4, len(train_len)), dtype=float),
#                         columns=[ky for ky in train_len.keys()],
#                         index=['rough_training', 'fine_tuning',
#                                'topographic error', 'quantization error'])
#
# cons.get_time("Time for Initialization", tic)
# t_som = time.time()
#
# for i, ky in enumerate(train_len.keys()):
#     t_som_temp = time.time()
#     print("SOM-Variant {}: {} of {}".format(ky, i+1, len(train_len)))
#     som = smp.SOMFactory.build(filtered_data_for_som.data_, mask=None, mapsize=mapsize, mapshape=mapform, lattice=lattice,
#                                normalization=normalization, initialization=initialization, neighborhood=neighborhood,
#                                training=training, name=ky, component_names=compnames)
#     som.train(n_job=1, verbose='info', train_rough_len=train_len[ky][0], train_rough_radiusin=train_rough_radiusin,
#               train_rough_radiusfin=train_rough_radiusfin, train_finetune_len=train_len[ky][1],
#               train_finetune_radiusin=train_finetune_radiusin, train_finetune_radiusfin=train_finetune_radiusfin)
#     topographic_error = get_topographic_error(som)
#     quantization_error = np.mean(som._bmu[1])
#
#     metalist = [train_len[ky][0], train_len[ky][1], topographic_error, quantization_error]
#     epo_soms_df[ky] = metalist
#
#     with open(os.path.join(resultpath, "som_{}.pickle".format(ky)), 'wb') as f:
#         pickle.dump(som, f)
#     cons.get_time("time for {}".format(ky), t_som_temp)
#
# cons.get_time("Time for all SOMs", t_som)  # last run: 28:33:19; 10x-SOMs ~ 30 min, while 500x-SOM: 26:27:02!
# savepath = os.path.join(resultpath, 'SOM_Epoch_Tuning.xlsx')
# epo_soms_df.to_excel(savepath, sheet_name='tuning epochs', na_rep='-')

#****************************
# Tuning/Testing the no of neurons
#****************************
#
# tic = time.time()
# print("Initialization/Loading data...")
#
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
#
# with open(os.path.join(datapath, 'filtered_data_for_som_notcorr.pickle'), 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# # SOM-Parameters fixed:
# mapform = 'planar'
# lattice = 'rect'
# normalization = 'var'
# initialization = 'pca'
# neighborhood = 'gaussian'
# training = 'batch'
# compnames = ["Q {}".format(filtered_data_for_som.varnames_[c]) if c.startswith('m') else filtered_data_for_som.varnames_[c] for c in filtered_data_for_som.attributes_]
#
# def compute_map_ratio(msize):
#     y = (33/15*msize)**0.5
#     x = 15/33*y
#     return [round(x), round(y)]
#
# # mapsize = {'doublerec': compute_map_ratio(1000),
# #            'noofHH': compute_map_ratio(9743),
# #            'doublenoofHH': compute_map_ratio(2*9743),
# #            'halfrec': compute_map_ratio(250)}
#
# mapsize = {'doublerec': compute_map_ratio(990),
#            '4timesrec': compute_map_ratio(1980),
#            'halfrec': compute_map_ratio(248)}
#
# # Training parameters fixed:
# def compute_train_len(mapsize):
#     msize = mapsize[0]*mapsize[1]
#     noepochs = msize*10
#     return round(noepochs/5), round(4*noepochs/5)
#
# def compute_radii(mapsize):
#     rough_ini = np.ceil(max(mapsize[0], mapsize[1])/2)
#     rough_fin = np.floor(rough_ini/4)
#     return rough_ini, rough_fin, rough_fin, 1
#
# neur_soms_df = pd.DataFrame(data=np.zeros((9, len(mapsize)), dtype=float),
#                         columns=[ky for ky in mapsize.keys()],
#                         index=['rough_training', 'fine_tuning', 'rough_radius_ini', 'rough_radius_final', 'fine_radius_final',
#                                'total neurons', 'mapratio', 'topographic error', 'quantization error'])
#
# cons.get_time("Time for Initialization", tic)
# t_som = time.time()
#
# for i, ky in enumerate(mapsize.keys()):
#     t_som_temp = time.time()
#     print("SOM-Variant {}: {} of {}".format(ky, i+1, len(mapsize)))
#     train_rough_len, train_finetune_len = compute_train_len(mapsize[ky])
#     train_rough_radiusin, train_rough_radiusfin, train_finetune_radiusin, train_finetune_radiusfin = compute_radii(mapsize[ky])
#     som = smp.SOMFactory.build(filtered_data_for_som.data_, mask=None, mapsize=mapsize[ky], mapshape=mapform, lattice=lattice,
#                                normalization=normalization, initialization=initialization, neighborhood=neighborhood,
#                                training=training, name=ky, component_names=compnames)
#     som.train(n_job=1, verbose='info', train_rough_len=train_rough_len, train_rough_radiusin=train_rough_radiusin,
#               train_rough_radiusfin=train_rough_radiusfin, train_finetune_len=train_finetune_len,
#               train_finetune_radiusin=train_finetune_radiusin, train_finetune_radiusfin=train_finetune_radiusfin)
#     topographic_error = get_topographic_error(som)
#     quantization_error = np.mean(som._bmu[1])
#
#     metalist = [train_rough_len, train_finetune_len, train_rough_radiusin, train_rough_radiusfin, train_finetune_radiusfin,
#                 mapsize[ky][0]*mapsize[ky][1], "{}:{}".format(mapsize[ky][0], mapsize[ky][1]),topographic_error, quantization_error]
#     neur_soms_df[ky] = metalist
#
#     with open(os.path.join(resultpath, "som_{}.pickle".format(ky)), 'wb') as f:
#         pickle.dump(som, f)
#     cons.get_time("time for {}".format(ky), t_som_temp)
#
# cons.get_time("Time for all SOMs", t_som)  # last run: 08:19:01;
# savepath = os.path.join(resultpath, 'SOM_Neurons_Tuning.xlsx')
# neur_soms_df.to_excel(savepath, sheet_name='tuning neurons', na_rep='-')

#****************************
# Testing different parameters
#****************************
#
# tic = time.time()
# print("Initialization/Loading data...")
#
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
#
# with open(os.path.join(datapath, 'filtered_data_for_som_notcorr.pickle'), 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# # SOM-Parameters fixed:
# mapform = 'planar'
# lattice = 'rect'
# training = 'batch'
# compnames = ["Q {}".format(filtered_data_for_som.varnames_[c]) if c.startswith('m') else filtered_data_for_som.varnames_[c] for c in filtered_data_for_som.attributes_]
#
# def compute_train_len(mapsize):
#     msize = mapsize[0]*mapsize[1]
#     noepochs = msize*10
#     return round(noepochs/5), round(4*noepochs/5)
#
# def compute_radii(mapsize):
#     rough_ini = np.ceil(max(mapsize[0], mapsize[1])/2)
#     rough_fin = np.floor(rough_ini/4)
#     return rough_ini, rough_fin, rough_fin, 1
#
# testing_dict = {
#     'bubble': {
#         'mapsize': [15, 33],
#         'initialization': 'pca',
#         'normalization': 'var',
#         'neighborhood': 'bubble'
#     },
#     'init1': {
#         'mapsize': [15, 33],
#         'initialization': 'random',
#         'normalization': 'var',
#         'neighborhood': 'gaussian'
#     },
#     'init2': {
#         'mapsize': [15, 33],
#         'initialization': 'random',
#         'normalization': 'var',
#         'neighborhood': 'gaussian'
#     },
#     'init3': {
#         'mapsize': [15, 33],
#         'initialization': 'random',
#         'normalization': 'var',
#         'neighborhood': 'gaussian'
#     },
#     'nonorm': {
#         'mapsize': [15, 33],
#         'initialization': 'pca',
#         'normalization': None,
#         'neighborhood': 'gaussian'
#     },
#     'nonorm2': {
#         'mapsize': [11, 47],
#         'initialization': 'pca',
#         'normalization': None,
#         'neighborhood': 'gaussian'},
#     'ms_extremeratio': {
#         'mapsize': [11, 47],
#         'initialization': 'pca',
#         'normalization': 'var',
#         'neighborhood': 'gaussian'
#     },
#     'ms_quadratic': {
#         'mapsize': [23, 23],
#         'initialization': 'pca',
#         'normalization': 'var',
#         'neighborhood': 'gaussian'
#     }
# }
#
#
# test_soms_df = pd.DataFrame(data=np.zeros((12, len(testing_dict)), dtype=float),
#                         columns=[ky for ky in testing_dict.keys()],
#                         index=['rough_training', 'fine_tuning', 'rough_radius_ini', 'rough_radius_final', 'fine_radius_final',
#                                'total neurons','mapsize', 'init', 'normalization','neighborhood',
#                                'topographic error', 'quantization error'])
#
# cons.get_time("Time for Initialization", tic)
# t_som = time.time()
#
# for i, ky in enumerate(testing_dict.keys()):
#     t_som_temp = time.time()
#     print("SOM-Variant {}: {} of {}".format(ky, i+1, len(testing_dict)))
#     train_rough_len, train_finetune_len = compute_train_len(testing_dict[ky]['mapsize'])
#     train_rough_radiusin, train_rough_radiusfin, train_finetune_radiusin, train_finetune_radiusfin = compute_radii(testing_dict[ky]['mapsize'])
#     som = smp.SOMFactory.build(filtered_data_for_som.data_, mask=None, mapsize=testing_dict[ky]['mapsize'], mapshape=mapform, lattice=lattice,
#                                normalization=testing_dict[ky]['normalization'], initialization=testing_dict[ky]['initialization'],
#                                neighborhood=testing_dict[ky]['neighborhood'],
#                                training=training, name=ky, component_names=compnames)
#     som.train(n_job=1, verbose='info', train_rough_len=train_rough_len, train_rough_radiusin=train_rough_radiusin,
#               train_rough_radiusfin=train_rough_radiusfin, train_finetune_len=train_finetune_len,
#               train_finetune_radiusin=train_finetune_radiusin, train_finetune_radiusfin=train_finetune_radiusfin)
#     topographic_error = get_topographic_error(som)
#     quantization_error = np.mean(som._bmu[1])
#
#     metalist = [train_rough_len, train_finetune_len, train_rough_radiusin, train_rough_radiusfin, train_finetune_radiusfin,
#                 testing_dict[ky]['mapsize'][0]*testing_dict[ky]['mapsize'][1], "{}:{}".format(testing_dict[ky]['mapsize'][0], testing_dict[ky]['mapsize'][1]),
#                 testing_dict[ky]['initialization'], testing_dict[ky]['normalization'],
#                 testing_dict[ky]['neighborhood'], topographic_error, quantization_error]
#     test_soms_df[ky] = metalist
#
#     with open(os.path.join(resultpath, "som_{}.pickle".format(ky)), 'wb') as f:
#         pickle.dump(som, f)
#     cons.get_time("time for {}".format(ky), t_som_temp)
#
# cons.get_time("Time for all SOMs", t_som)  # last run: 04:19:37
# savepath = os.path.join(resultpath, 'SOM_Testing.xlsx')
# test_soms_df.to_excel(savepath, sheet_name='testing', na_rep='-')


#****************************
# Fine-Tuning part I (Radius)
#****************************
#
# tic = time.time()
# print("Initialization/Loading data...")
#
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
#
# with open(os.path.join(datapath, 'filtered_data_for_som_notcorr.pickle'), 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# # SOM-Parameters fixed:
# mapform = 'planar'
# mapsize = [21, 47]  # decided based on computations for neurons-tuning (we did not take the best one, because of computational burden and because of Vesanto telling that neurons are less important than radius and we usually use 100 - 600 neurons)
# lattice = 'rect'
# normalization = 'var'
# initialization = 'pca'
# neighborhood = 'gaussian'
# training = 'batch'
# compnames = ["Q {}".format(filtered_data_for_som.varnames_[c]) if c.startswith('m') else filtered_data_for_som.varnames_[c] for c in filtered_data_for_som.attributes_]
#
# # Training parameters fixed:
# train_rough_len = 8*(21*47)  # decided based on computations for epoch-tuning
# train_finetune_len = 4*train_rough_len  # recommendation of SOM-Toolbox and decided based on computations for epoch-tuning
#
# # since the radius is obviously one of the most important parameters, we tune this again
# radii = {'bestrec': (24, 6, 1),
#          'halfdia1': (24, 1, 1),
#          '34thdia4th': (35, 9, 1),
#          '34thdia1': (35, 1, 1),
#          '14thdia4th': (12, 3, 1),
#          '14thdia1': (12, 1, 1)
#          }
#
# rad_soms_df = pd.DataFrame(data=np.zeros((5, len(radii)), dtype=float),
#                         columns=[ky for ky in radii.keys()],
#                         index=['rough_radius_ini', 'rough_radius_final', 'fine_radius_final',
#                                'topographic error', 'quantization error'])
#
# cons.get_time("Time for Initialization", tic)
# t_som = time.time()
#
# for i, ky in enumerate(radii.keys()):
#     t_som_temp = time.time()
#     print("SOM-Variant {}: {} of {}".format(ky, i+1, len(radii)))
#     som = smp.SOMFactory.build(filtered_data_for_som.data_, mask=None, mapsize=mapsize, mapshape=mapform, lattice=lattice,
#                                normalization=normalization, initialization=initialization, neighborhood=neighborhood,
#                                training=training, name=ky, component_names=compnames)
#     som.train(n_job=1, verbose='info', train_rough_len=train_rough_len, train_rough_radiusin=radii[ky][0],
#               train_rough_radiusfin=radii[ky][1], train_finetune_len=train_finetune_len,
#               train_finetune_radiusin=radii[ky][1], train_finetune_radiusfin=radii[ky][2])
#     topographic_error = get_topographic_error(som)
#     quantization_error = np.mean(som._bmu[1])
#
#     metalist = [radii[ky][0], radii[ky][1], radii[ky][2], topographic_error, quantization_error]
#     rad_soms_df[ky] = metalist
#
#     with open(os.path.join(resultpath, "som_{}.pickle".format(ky)), 'wb') as f:
#         pickle.dump(som, f)
#     cons.get_time("time for {}".format(ky), t_som_temp)
#
# cons.get_time("Time for all SOMs", t_som)  # last run: 40:06:03
# savepath = os.path.join(resultpath, 'SOM_Fine-Tuning.xlsx')
# rad_soms_df.to_excel(savepath, sheet_name='tuning radii', na_rep='-')

#****************************
# Fine-Tuning part II (epochs)
#****************************
#
# tic = time.time()
# print("Initialization/Loading data...")
#
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
# with open(os.path.join(datapath, 'filtered_data_for_som_notcorr.pickle'), 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# # SOM-Parameters fixed:
# mapform = 'planar'
# mapsize = [21, 47]  # decided based on computations for neurons-tuning (we did not take the best one, because of computational burden and because of Vesanto telling that neurons are less important than radius and we usually use 100 - 600 neurons)
# lattice = 'rect'
# normalization = 'var'
# initialization = 'pca'
# neighborhood = 'gaussian'
# training = 'batch'
# compnames = ["Q {}".format(filtered_data_for_som.varnames_[c]) if c.startswith('m') else filtered_data_for_som.varnames_[c] for c in filtered_data_for_som.attributes_]
#
# # Training parameters fixed:
# train_rough_radiusin = 35
# train_rough_radiusfin = 9
# train_finetune_radiusin = 9
# train_finetune_radiusfin = 1
#
# train_len = {'doubling': (15792, 63168),
#          'fine2xrough': (13160, 26320),
#          'fine7xrough': (4935, 34545),
#          'fine10xrough': (3589, 35891)
#          }
#
# epo_soms_df = pd.DataFrame(data=np.zeros((4, len(train_len)), dtype=float),
#                         columns=[ky for ky in train_len.keys()],
#                         index=['rough_training', 'fine_tuning',
#                                'topographic error', 'quantization error'])
#
# cons.get_time("Time for Initialization", tic)
# t_som = time.time()
#
# for i, ky in enumerate(train_len.keys()):
#     t_som_temp = time.time()
#     print("SOM-Variant {}: {} of {}".format(ky, i+1, len(train_len)))
#     som = smp.SOMFactory.build(filtered_data_for_som.data_, mask=None, mapsize=mapsize, mapshape=mapform, lattice=lattice,
#                                normalization=normalization, initialization=initialization, neighborhood=neighborhood,
#                                training=training, name=ky, component_names=compnames)
#     som.train(n_job=1, verbose='info', train_rough_len=train_len[ky][0], train_rough_radiusin=train_rough_radiusin,
#               train_rough_radiusfin=train_rough_radiusfin, train_finetune_len=train_len[ky][1],
#               train_finetune_radiusin=train_finetune_radiusin, train_finetune_radiusfin=train_finetune_radiusfin)
#     topographic_error = get_topographic_error(som)
#     quantization_error = np.mean(som._bmu[1])
#
#     metalist = [train_len[ky][0], train_len[ky][1], topographic_error, quantization_error]
#     epo_soms_df[ky] = metalist
#
#     with open(os.path.join(resultpath, "som_{}.pickle".format(ky)), 'wb') as f:
#         pickle.dump(som, f)
#     cons.get_time("time for {}".format(ky), t_som_temp)
#
# cons.get_time("Time for all SOMs", t_som)  # last run: 33:15:05; Last SOM (fine10xrough): 06:39:06
# savepath = os.path.join(resultpath, 'SOM_Epoch_Fine-Tuning.xlsx')
# epo_soms_df.to_excel(savepath, sheet_name='fine tuning epochs', na_rep='-')


#****************************
# Final calculation with best parameters and data with new pattern recognition filter
#****************************
#
# tic = time.time()
# print("Initialization/Loading data...")
#
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
# with open(os.path.join(datapath, 'filtered_data_for_som_notcorr.pickle'), 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# mapform = 'planar'
# lattice = 'rect'
# normalization = 'var'
# initialization = 'pca'
# neighborhood = 'gaussian'
# training = 'batch'
# compnames = ["Q {}".format(filtered_data_for_som.varnames_[c]) if c.startswith('m') else filtered_data_for_som.varnames_[c] for c in filtered_data_for_som.attributes_]
# mapsize = [21, 47]  # decided based on computations for neurons-tuning (we did not take the best one, because of computational burden and because of Vesanto telling that neurons are less important than radius and we usually use 100 - 600 neurons)
# train_rough_len = 8*(21*47)  # decided based on computations for epoch-tuning
# train_finetune_len = 4*train_rough_len  # recommendation of SOM-Toolbox and decided based on computations for epoch-tuning
# train_rough_radiusin = 35
# train_rough_radiusfin = 9
# train_finetune_radiusin = 9
# train_finetune_radiusfin = 1
#
# best_som_df = pd.DataFrame(data=np.zeros((16, 1), dtype=float),
#                         columns=['best_som'],
#                         index=['mapform','lattice','normalization','initialization','neighborhood','training','mapsize',
#                                'no of neurons','train_rough_len','train_finetune_len','train_rough_radiusin',
#                                'train_rough_radiusfin','train_finetune_radiusin','train_finetune_radiusfin',
#                                'topographic error', 'quantization error'])
#
# cons.get_time("Time for Initialization", tic)
# t_som = time.time()
# som = smp.SOMFactory.build(filtered_data_for_som.data_, mask=None, mapsize=mapsize, mapshape=mapform, lattice=lattice,
#                            normalization=normalization, initialization=initialization, neighborhood=neighborhood,
#                            training=training, name='best_som', component_names=compnames)
# som.train(n_job=1, verbose='info', train_rough_len=train_rough_len, train_rough_radiusin=train_rough_radiusin,
#           train_rough_radiusfin=train_rough_radiusfin, train_finetune_len=train_finetune_len,
#           train_finetune_radiusin=train_finetune_radiusin, train_finetune_radiusfin=train_finetune_radiusfin)
# topographic_error = get_topographic_error(som)
# quantization_error = np.mean(som._bmu[1])
#
# metalist = [mapform,lattice,normalization,initialization,neighborhood,training,"{}:{}".format(mapsize[0], mapsize[1]),
#             mapsize[0]*mapsize[1],train_rough_len,train_finetune_len,train_rough_radiusin,
#             train_rough_radiusfin,train_finetune_radiusin,train_finetune_radiusfin,
#             topographic_error, quantization_error]
#
# best_som_df['best_som'] = metalist
#
# with open(os.path.join(resultpath, "best_som.pickle"), 'wb') as f:
#     pickle.dump(som, f)
#
# cons.get_time("Time for best SOM", t_som)  # last run: 06:36:55
# savepath = os.path.join(resultpath, 'Best_SOM.xlsx')
# best_som_df.to_excel(savepath, sheet_name='best som', na_rep='-')


#----------------------------------------------------------------------------------------------------------------------
# Tuning Self-Organizing Maps --> PART II: full dataset
#----------------------------------------------------------------------------------------------------------------------

# PRINT from consumption_cockpit
# no. of neurons: 493.3051793768235
# ratio of side lengths: 2.4972571667482706
# length 1: 14.054849630798339
# length 2: 35.09857396808044
# definite lengths: 14 : 35
# no of neurons: 490


# IMPORTANT: in the sompy-package, I found a bug how they compute the topographic error the following function is a copy/paste
# from the original function in sompy, but it uses the scaled data rather than the unscaled (which is wrong!)
def get_topographic_error(som):
    bmus1 = som.find_bmu(som._data, njb=1, nth=1)
    bmus2 = som.find_bmu(som._data, njb=1, nth=2)
    bmus_gap = np.abs(
        (som.bmu_ind_to_xy(np.array(bmus1[0]))[:, 0:2] - som.bmu_ind_to_xy(np.array(bmus2[0]))[:, 0:2]).sum(axis=1))
    return np.mean(bmus_gap != 1)

#****************************
# Tuning the neighborhood radius
#****************************
#
# tic = time.time()
# print("Initialization/Loading data...")
#
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
#
# with open(os.path.join(datapath, 'filtered_data_for_som_full.pickle'), 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# # SOM-Parameters fixed:
# mapform = 'planar'
# mapsize = [14, 35]
# lattice = 'rect'
# normalization = 'var'
# initialization = 'pca'
# neighborhood = 'gaussian'
# training = 'batch'
# compnames = ["Q {}".format(filtered_data_for_som.varnames_[c]) if c.startswith('m') else filtered_data_for_som.varnames_[c] for c in filtered_data_for_som.attributes_]
#
# # Training parameters fixed:
# train_rough_len = 980  # recommendation of SOM-Toolbox (Kohonen took too long): total no. of epochs: 10 x map units --> (10 x 490)/5 divided by 5 to have it 4 times shorter than finetuning
# train_finetune_len = 3920  # recommendation of SOM-Toolbox (Kohonen took too long) (10 x 495)/ 5 x 4
#
# # ATTENTION: in a first try, we already investigated if we should set final radius to 1 or to 0. Obviously, the current
# # implementation of SOMPY doesn't care --> it doesn't matter if 0 or 1 --> with both we get the exact same results and
# # look at only the BMU in the final epochs.
# radii = {'def': (0, 0, 0),
#          'bestrec': (18, 5, 1),
#          'halfdia1': (18, 1, 1),
#          '34thdia4th': (26, 7, 1),
#          '34thdia1': (26, 1, 1),
#          '14thdia4th': (9, 2, 1),
#          '14thdia1': (9, 1, 1)
#          }
#
# rad_soms_df = pd.DataFrame(data=np.zeros((5, len(radii)), dtype=float),
#                         columns=[ky for ky in radii.keys()],
#                         index=['rough_radius_ini', 'rough_radius_final', 'fine_radius_final',
#                                'topographic error', 'quantization error'])
#
# cons.get_time("Time for Initialization", tic)
# t_som = time.time()
#
# for i, ky in enumerate(radii.keys()):
#     t_som_temp = time.time()
#     print("SOM-Variant {}: {} of {}".format(ky, i+1, len(radii)))
#     if ky == 'def':
#         som = smp.SOMFactory.build(filtered_data_for_som.data_, mask=None, mapsize=mapsize, mapshape=mapform, lattice=lattice,
#                                    normalization=normalization, initialization=initialization, neighborhood=neighborhood,
#                                    training=training, name=ky, component_names=compnames)
#         som.train(n_job=1, verbose=None)
#
#         topographic_error = get_topographic_error(som)
#         quantization_error = np.mean(som._bmu[1])
#
#         metalist = [np.nan, np.nan, np.nan, topographic_error, quantization_error]
#         rad_soms_df[ky] = metalist
#
#         with open(os.path.join(resultpath, "som_{}.pickle".format(ky)), 'wb') as f:
#             pickle.dump(som, f)
#     else:
#         som = smp.SOMFactory.build(filtered_data_for_som.data_, mask=None, mapsize=mapsize, mapshape=mapform, lattice=lattice,
#                                    normalization=normalization, initialization=initialization, neighborhood=neighborhood,
#                                    training=training, name=ky, component_names=compnames)
#         som.train(n_job=1, verbose='info', train_rough_len=train_rough_len, train_rough_radiusin=radii[ky][0],
#                   train_rough_radiusfin=radii[ky][1], train_finetune_len=train_finetune_len,
#                   train_finetune_radiusin=radii[ky][1], train_finetune_radiusfin=radii[ky][2])
#         topographic_error = get_topographic_error(som)
#         quantization_error = np.mean(som._bmu[1])
#
#         metalist = [radii[ky][0], radii[ky][1], radii[ky][2], topographic_error, quantization_error]
#         rad_soms_df[ky] = metalist
#
#         with open(os.path.join(resultpath, "som_{}.pickle".format(ky)), 'wb') as f:
#             pickle.dump(som, f)
#     cons.get_time("time for {}".format(ky), t_som_temp)
#
# cons.get_time("Time for all SOMs", t_som)  # last run: 03:20:42, per SOM about 30 minutes
# savepath = os.path.join(resultpath, 'SOM_Radius_Tuning.xlsx')
# rad_soms_df.to_excel(savepath, sheet_name='tuning radii', na_rep='-')

#****************************
# Tuning the no of epochs
#****************************
#
# tic = time.time()
# print("Initialization/Loading data...")
#
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
# with open(os.path.join(datapath, 'filtered_data_for_som_full.pickle'), 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# # SOM-Parameters fixed:
# mapform = 'planar'
# mapsize = [14, 35]
# lattice = 'rect'
# normalization = 'var'
# initialization = 'pca'
# neighborhood = 'gaussian'
# training = 'batch'
# compnames = ["Q {}".format(filtered_data_for_som.varnames_[c]) if c.startswith('m') else filtered_data_for_som.varnames_[c] for c in filtered_data_for_som.attributes_]
#
# # Training parameters fixed:
# train_rough_radiusin = 18
# train_rough_radiusfin = 5
# train_finetune_radiusin = 5
# train_finetune_radiusfin = 1
#
# train_len = {'10_10': (445, 4455),
#          '10_4': (980, 3920),
#          '10_0.1': (4455, 445),
#          '10_0.25': (3920, 980),
#          '500_10': (22273, 222727)
#          }
#
# epo_soms_df = pd.DataFrame(data=np.zeros((4, len(train_len)), dtype=float),
#                         columns=[ky for ky in train_len.keys()],
#                         index=['rough_training', 'fine_tuning',
#                                'topographic error', 'quantization error'])
#
# cons.get_time("Time for Initialization", tic)
# t_som = time.time()
#
# for i, ky in enumerate(train_len.keys()):
#     t_som_temp = time.time()
#     print("SOM-Variant {}: {} of {}".format(ky, i+1, len(train_len)))
#     som = smp.SOMFactory.build(filtered_data_for_som.data_, mask=None, mapsize=mapsize, mapshape=mapform, lattice=lattice,
#                                normalization=normalization, initialization=initialization, neighborhood=neighborhood,
#                                training=training, name=ky, component_names=compnames)
#     som.train(n_job=1, verbose='info', train_rough_len=train_len[ky][0], train_rough_radiusin=train_rough_radiusin,
#               train_rough_radiusfin=train_rough_radiusfin, train_finetune_len=train_len[ky][1],
#               train_finetune_radiusin=train_finetune_radiusin, train_finetune_radiusfin=train_finetune_radiusfin)
#     topographic_error = get_topographic_error(som)
#     quantization_error = np.mean(som._bmu[1])
#
#     metalist = [train_len[ky][0], train_len[ky][1], topographic_error, quantization_error]
#     epo_soms_df[ky] = metalist
#
#     with open(os.path.join(resultpath, "som_{}.pickle".format(ky)), 'wb') as f:
#         pickle.dump(som, f)
#     cons.get_time("time for {}".format(ky), t_som_temp)
#
# cons.get_time("Time for all SOMs", t_som)  # last run: 29:33:26; 10x-SOMs ~ 30 min, while 500x-SOM: 27:23:07!
# savepath = os.path.join(resultpath, 'SOM_Epoch_Tuning.xlsx')
# epo_soms_df.to_excel(savepath, sheet_name='tuning epochs', na_rep='-')

#****************************
# Tuning/Testing the no of neurons
#****************************
#
# tic = time.time()
# print("Initialization/Loading data...")
#
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
#
# with open(os.path.join(datapath, 'filtered_data_for_som_full.pickle'), 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# # SOM-Parameters fixed:
# mapform = 'planar'
# lattice = 'rect'
# normalization = 'var'
# initialization = 'pca'
# neighborhood = 'gaussian'
# training = 'batch'
# compnames = ["Q {}".format(filtered_data_for_som.varnames_[c]) if c.startswith('m') else filtered_data_for_som.varnames_[c] for c in filtered_data_for_som.attributes_]
#
# def compute_map_ratio(msize):
#     y = (35/14*msize)**0.5
#     x = 14/35*y
#     return [round(x), round(y)]
#
# # mapsize = {'doublerec': compute_map_ratio(1000),
# #            'noofHH': compute_map_ratio(9743),
# #            'doublenoofHH': compute_map_ratio(2*9743),
# #            'halfrec': compute_map_ratio(250)}
#
# mapsize = {'doublerec': compute_map_ratio(980),
#            '4timesrec': compute_map_ratio(1960),
#            'halfrec': compute_map_ratio(245)}
#
# # Training parameters fixed:
# def compute_train_len(mapsize):
#     msize = mapsize[0]*mapsize[1]
#     noepochs = msize*10
#     return round(noepochs/5), round(4*noepochs/5)
#
# def compute_radii(mapsize):
#     rough_ini = np.ceil(max(mapsize[0], mapsize[1])/2)
#     rough_fin = np.floor(rough_ini/4)
#     return rough_ini, rough_fin, rough_fin, 1
#
# neur_soms_df = pd.DataFrame(data=np.zeros((9, len(mapsize)), dtype=float),
#                         columns=[ky for ky in mapsize.keys()],
#                         index=['rough_training', 'fine_tuning', 'rough_radius_ini', 'rough_radius_final', 'fine_radius_final',
#                                'total neurons', 'mapratio', 'topographic error', 'quantization error'])
#
# cons.get_time("Time for Initialization", tic)
# t_som = time.time()
#
# for i, ky in enumerate(mapsize.keys()):
#     t_som_temp = time.time()
#     print("SOM-Variant {}: {} of {}".format(ky, i+1, len(mapsize)))
#     train_rough_len, train_finetune_len = compute_train_len(mapsize[ky])
#     train_rough_radiusin, train_rough_radiusfin, train_finetune_radiusin, train_finetune_radiusfin = compute_radii(mapsize[ky])
#     som = smp.SOMFactory.build(filtered_data_for_som.data_, mask=None, mapsize=mapsize[ky], mapshape=mapform, lattice=lattice,
#                                normalization=normalization, initialization=initialization, neighborhood=neighborhood,
#                                training=training, name=ky, component_names=compnames)
#     som.train(n_job=1, verbose='info', train_rough_len=train_rough_len, train_rough_radiusin=train_rough_radiusin,
#               train_rough_radiusfin=train_rough_radiusfin, train_finetune_len=train_finetune_len,
#               train_finetune_radiusin=train_finetune_radiusin, train_finetune_radiusfin=train_finetune_radiusfin)
#     topographic_error = get_topographic_error(som)
#     quantization_error = np.mean(som._bmu[1])
#
#     metalist = [train_rough_len, train_finetune_len, train_rough_radiusin, train_rough_radiusfin, train_finetune_radiusfin,
#                 mapsize[ky][0]*mapsize[ky][1], "{}:{}".format(mapsize[ky][0], mapsize[ky][1]),topographic_error, quantization_error]
#     neur_soms_df[ky] = metalist
#
#     with open(os.path.join(resultpath, "som_{}.pickle".format(ky)), 'wb') as f:
#         pickle.dump(som, f)
#     cons.get_time("time for {}".format(ky), t_som_temp)
#
# cons.get_time("Time for all SOMs", t_som)  # last run: 08:47:51;
# savepath = os.path.join(resultpath, 'SOM_Neurons_Tuning.xlsx')
# neur_soms_df.to_excel(savepath, sheet_name='tuning neurons', na_rep='-')

#****************************
# Testing different parameters
#****************************
#
# tic = time.time()
# print("Initialization/Loading data...")
#
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
#
# with open(os.path.join(datapath, 'filtered_data_for_som_full.pickle'), 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# # SOM-Parameters fixed:
# mapform = 'planar'
# lattice = 'rect'
# training = 'batch'
# compnames = ["Q {}".format(filtered_data_for_som.varnames_[c]) if c.startswith('m') else filtered_data_for_som.varnames_[c] for c in filtered_data_for_som.attributes_]
#
# def compute_train_len(mapsize):
#     msize = mapsize[0]*mapsize[1]
#     noepochs = msize*10
#     return round(noepochs/5), round(4*noepochs/5)
#
# def compute_radii(mapsize):
#     rough_ini = np.ceil(max(mapsize[0], mapsize[1])/2)
#     rough_fin = np.floor(rough_ini/4)
#     return rough_ini, rough_fin, rough_fin, 1
#
# testing_dict = {
#     'bubble': {
#         'mapsize': [14, 35],
#         'initialization': 'pca',
#         'normalization': 'var',
#         'neighborhood': 'bubble'
#     },
#     'init1': {
#         'mapsize': [14, 35],
#         'initialization': 'random',
#         'normalization': 'var',
#         'neighborhood': 'gaussian'
#     },
#     'init2': {
#         'mapsize': [14, 35],
#         'initialization': 'random',
#         'normalization': 'var',
#         'neighborhood': 'gaussian'
#     },
#     'init3': {
#         'mapsize': [14, 35],
#         'initialization': 'random',
#         'normalization': 'var',
#         'neighborhood': 'gaussian'
#     },
#     'nonorm': {
#         'mapsize': [14, 35],
#         'initialization': 'pca',
#         'normalization': None,
#         'neighborhood': 'gaussian'
#     },
#     'nonorm2': {
#         'mapsize': [8, 65],
#         'initialization': 'pca',
#         'normalization': None,
#         'neighborhood': 'gaussian'},
#     'ms_extremeratio': {
#         'mapsize': [8, 65],
#         'initialization': 'pca',
#         'normalization': 'var',
#         'neighborhood': 'gaussian'
#     },
#     'ms_quadratic': {
#         'mapsize': [22, 22],
#         'initialization': 'pca',
#         'normalization': 'var',
#         'neighborhood': 'gaussian'
#     }
# }
#
#
# test_soms_df = pd.DataFrame(data=np.zeros((12, len(testing_dict)), dtype=float),
#                         columns=[ky for ky in testing_dict.keys()],
#                         index=['rough_training', 'fine_tuning', 'rough_radius_ini', 'rough_radius_final', 'fine_radius_final',
#                                'total neurons','mapsize', 'init', 'normalization','neighborhood',
#                                'topographic error', 'quantization error'])
#
# cons.get_time("Time for Initialization", tic)
# t_som = time.time()
#
# for i, ky in enumerate(testing_dict.keys()):
#     t_som_temp = time.time()
#     print("SOM-Variant {}: {} of {}".format(ky, i+1, len(testing_dict)))
#     train_rough_len, train_finetune_len = compute_train_len(testing_dict[ky]['mapsize'])
#     train_rough_radiusin, train_rough_radiusfin, train_finetune_radiusin, train_finetune_radiusfin = compute_radii(testing_dict[ky]['mapsize'])
#     som = smp.SOMFactory.build(filtered_data_for_som.data_, mask=None, mapsize=testing_dict[ky]['mapsize'], mapshape=mapform, lattice=lattice,
#                                normalization=testing_dict[ky]['normalization'], initialization=testing_dict[ky]['initialization'],
#                                neighborhood=testing_dict[ky]['neighborhood'],
#                                training=training, name=ky, component_names=compnames)
#     som.train(n_job=1, verbose='info', train_rough_len=train_rough_len, train_rough_radiusin=train_rough_radiusin,
#               train_rough_radiusfin=train_rough_radiusfin, train_finetune_len=train_finetune_len,
#               train_finetune_radiusin=train_finetune_radiusin, train_finetune_radiusfin=train_finetune_radiusfin)
#     topographic_error = get_topographic_error(som)
#     quantization_error = np.mean(som._bmu[1])
#
#     metalist = [train_rough_len, train_finetune_len, train_rough_radiusin, train_rough_radiusfin, train_finetune_radiusfin,
#                 testing_dict[ky]['mapsize'][0]*testing_dict[ky]['mapsize'][1], "{}:{}".format(testing_dict[ky]['mapsize'][0], testing_dict[ky]['mapsize'][1]),
#                 testing_dict[ky]['initialization'], testing_dict[ky]['normalization'],
#                 testing_dict[ky]['neighborhood'], topographic_error, quantization_error]
#     test_soms_df[ky] = metalist
#
#     with open(os.path.join(resultpath, "som_{}.pickle".format(ky)), 'wb') as f:
#         pickle.dump(som, f)
#     cons.get_time("time for {}".format(ky), t_som_temp)
#
# cons.get_time("Time for all SOMs", t_som)  # last run: 04:24:49
# savepath = os.path.join(resultpath, 'SOM_Testing.xlsx')
# test_soms_df.to_excel(savepath, sheet_name='testing', na_rep='-')

#****************************
# Fine-Tuning part I (Radius)
#****************************
#
# tic = time.time()
# print("Initialization/Loading data...")
#
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
#
# with open(os.path.join(datapath, 'filtered_data_for_som_full.pickle'), 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# # SOM-Parameters fixed:
# mapform = 'planar'
# mapsize = [20, 49]  # decided based on computations for neurons-tuning (we did not take the best one, because of computational burden and because of Vesanto telling that neurons are less important than radius and we usually use 100 - 600 neurons)
# lattice = 'rect'
# normalization = 'var'
# initialization = 'pca'
# neighborhood = 'gaussian'
# training = 'batch'
# compnames = ["Q {}".format(filtered_data_for_som.varnames_[c]) if c.startswith('m') else filtered_data_for_som.varnames_[c] for c in filtered_data_for_som.attributes_]
#
# # Training parameters fixed:
# train_rough_len = round(10*(20*49)/11)  # decided based on computations for epoch-tuning
# train_finetune_len = 10*train_rough_len  # recommendation of SOM-Toolbox and decided based on computations for epoch-tuning
#
# # since the radius is obviously one of the most important parameters, we tune this again
# radii = {'bestrec': (25, 6, 1),
#          'halfdia1': (25, 1, 1),
#          '34thdia4th': (37, 9, 1),
#          '34thdia1': (37, 1, 1),
#          '14thdia4th': (13, 3, 1),
#          '14thdia1': (13, 1, 1)
#          }
#
# rad_soms_df = pd.DataFrame(data=np.zeros((5, len(radii)), dtype=float),
#                         columns=[ky for ky in radii.keys()],
#                         index=['rough_radius_ini', 'rough_radius_final', 'fine_radius_final',
#                                'topographic error', 'quantization error'])
#
# cons.get_time("Time for Initialization", tic)
# t_som = time.time()
#
# for i, ky in enumerate(radii.keys()):
#     t_som_temp = time.time()
#     print("SOM-Variant {}: {} of {}".format(ky, i+1, len(radii)))
#     som = smp.SOMFactory.build(filtered_data_for_som.data_, mask=None, mapsize=mapsize, mapshape=mapform, lattice=lattice,
#                                normalization=normalization, initialization=initialization, neighborhood=neighborhood,
#                                training=training, name=ky, component_names=compnames)
#     som.train(n_job=1, verbose='info', train_rough_len=train_rough_len, train_rough_radiusin=radii[ky][0],
#               train_rough_radiusfin=radii[ky][1], train_finetune_len=train_finetune_len,
#               train_finetune_radiusin=radii[ky][1], train_finetune_radiusfin=radii[ky][2])
#     topographic_error = get_topographic_error(som)
#     quantization_error = np.mean(som._bmu[1])
#
#     metalist = [radii[ky][0], radii[ky][1], radii[ky][2], topographic_error, quantization_error]
#     rad_soms_df[ky] = metalist
#
#     with open(os.path.join(resultpath, "som_{}.pickle".format(ky)), 'wb') as f:
#         pickle.dump(som, f)
#     cons.get_time("time for {}".format(ky), t_som_temp)
#
# cons.get_time("Time for all SOMs", t_som)  # last run: 10:58:49
# savepath = os.path.join(resultpath, 'SOM_Fine-Tuning.xlsx')
# rad_soms_df.to_excel(savepath, sheet_name='tuning radii', na_rep='-')

#****************************
# Fine-Tuning part II (Radius again with larger training lengths)
#****************************
#
# tic = time.time()
# print("Initialization/Loading data...")
# 
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
# 
# 
# with open(os.path.join(datapath, 'filtered_data_for_som_full.pickle'), 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
# 
# # SOM-Parameters fixed:
# mapform = 'planar'
# mapsize = [20, 49]  # decided based on computations for neurons-tuning (we did not take the best one, because of computational burden and because of Vesanto telling that neurons are less important than radius and we usually use 100 - 600 neurons)
# lattice = 'rect'
# normalization = 'var'
# initialization = 'pca'
# neighborhood = 'gaussian'
# training = 'batch'
# compnames = ["Q {}".format(filtered_data_for_som.varnames_[c]) if c.startswith('m') else filtered_data_for_som.varnames_[c] for c in filtered_data_for_som.attributes_]
# 
# # Training parameters fixed:
# train_rough_len = (20*49)*8  # decided based on computations for epoch-tuning
# train_finetune_len = 4*train_rough_len  # recommendation of SOM-Toolbox and decided based on computations for epoch-tuning
# 
# # since the radius is obviously one of the most important parameters, we tune this again
# radii = {'bestrec': (25, 6, 1),
#          'halfdia1': (25, 1, 1),
#          '34thdia4th': (37, 9, 1),
#          '34thdia1': (37, 1, 1),
#          '14thdia4th': (13, 3, 1),
#          '14thdia1': (13, 1, 1)
#          }
# 
# rad_soms_df = pd.DataFrame(data=np.zeros((5, len(radii)), dtype=float),
#                         columns=[ky for ky in radii.keys()],
#                         index=['rough_radius_ini', 'rough_radius_final', 'fine_radius_final',
#                                'topographic error', 'quantization error'])
# 
# cons.get_time("Time for Initialization", tic)
# t_som = time.time()
# 
# for i, ky in enumerate(radii.keys()):
#     t_som_temp = time.time()
#     print("SOM-Variant {}: {} of {}".format(ky, i+1, len(radii)))
#     som = smp.SOMFactory.build(filtered_data_for_som.data_, mask=None, mapsize=mapsize, mapshape=mapform, lattice=lattice,
#                                normalization=normalization, initialization=initialization, neighborhood=neighborhood,
#                                training=training, name=ky, component_names=compnames)
#     som.train(n_job=1, verbose='info', train_rough_len=train_rough_len, train_rough_radiusin=radii[ky][0],
#               train_rough_radiusfin=radii[ky][1], train_finetune_len=train_finetune_len,
#               train_finetune_radiusin=radii[ky][1], train_finetune_radiusfin=radii[ky][2])
#     topographic_error = get_topographic_error(som)
#     quantization_error = np.mean(som._bmu[1])
# 
#     metalist = [radii[ky][0], radii[ky][1], radii[ky][2], topographic_error, quantization_error]
#     rad_soms_df[ky] = metalist
# 
#     with open(os.path.join(resultpath, "som_{}.pickle".format(ky)), 'wb') as f:
#         pickle.dump(som, f)
#     cons.get_time("time for {}".format(ky), t_som_temp)
# 
# cons.get_time("Time for all SOMs", t_som)  # last run: 43:19:57
# savepath = os.path.join(resultpath, 'SOM_Fine-Tuning.xlsx')
# rad_soms_df.to_excel(savepath, sheet_name='tuning radii', na_rep='-')


# #----------------------------------------------------------------------------------------------------------------------
# # Clustering (only DBSCAN --> agglomerative clustering and K-Means on PC)
# #----------------------------------------------------------------------------------------------------------------------
# if __name__ == '__main__':
#     tic = time.time()
#     print('loading data...')
#
#     datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
#     resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
#     # Load the SOM
#     with open(os.path.join(datapath, 'best_som.pickle'), 'rb') as f:
#         som = pickle.load(f)
#
#     # We load arbitrarily the first "component map" to compute the connectivity (how the "pixels" are connected to
#     # each other).
#     som_map = som.codebook.matrix[:,0].reshape(som.codebook.mapsize[0],som.codebook.mapsize[1])
#     som_connectivity = grid_to_graph(*som_map.shape)
#
#     # Even though, we only compute DBSCAN on Linux-Server, we will construct the exactly same overview-xlsx and do thus
#     # not delete the following dicts:
#     hierarchical_params = {
#         'ward_eucl_scaled_nonconn': {'linkage': 'ward', 'connectivity': None, 'affinity': 'euclidean', 'scaling': True},
#         'ward_eucl_unscaled_nonconn': {'linkage': 'ward', 'connectivity': None, 'affinity': 'euclidean', 'scaling': False},
#         'ward_eucl_scaled_conn': {'linkage': 'ward', 'connectivity': som_connectivity, 'affinity': 'euclidean', 'scaling': True},
#         'ward_eucl_unscaled_conn': {'linkage': 'ward', 'connectivity': som_connectivity, 'affinity': 'euclidean', 'scaling': False},
#         'avg_l1_scaled_nonconn': {'linkage': 'average', 'connectivity': None, 'affinity': 'l1', 'scaling': True},
#         'avg_l1_unscaled_nonconn': {'linkage': 'average', 'connectivity': None, 'affinity': 'l1', 'scaling': False},
#         'avg_l1_scaled_conn': {'linkage': 'average', 'connectivity': som_connectivity, 'affinity': 'l1', 'scaling': True},
#         'avg_l1_unscaled_conn': {'linkage': 'average', 'connectivity': som_connectivity, 'affinity': 'l1', 'scaling': False},
#         'avg_cos_scaled_nonconn': {'linkage': 'average', 'connectivity': None, 'affinity': 'cosine', 'scaling': True},
#         'avg_cos_unscaled_nonconn': {'linkage': 'average', 'connectivity': None, 'affinity': 'cosine', 'scaling': False},
#         'avg_cos_scaled_conn': {'linkage': 'average', 'connectivity': som_connectivity, 'affinity': 'cosine', 'scaling': True},
#         'avg_cos_unscaled_conn': {'linkage': 'average', 'connectivity': som_connectivity, 'affinity': 'cosine', 'scaling': False}
#     }
#
#     kmeans_params = {
#         'kmeans_scaled': {'scaling': True},
#         'kmeans_unscaled': {'scaling': False},
#     }
#
#     dbscan_params = {
#                 'dbscan_eucl_scaled': {'metric': 'euclidean', 'scaling': True},
#                 'dbscan_eucl_unscaled': {'metric': 'euclidean', 'scaling': False},
#                 'dbscan_l1_scaled': {'metric': 'l1', 'scaling': True},
#                 'dbscan_l1_unscaled': {'metric': 'l1', 'scaling': False},
#                 'dbscan_cos_scaled': {'metric': 'cosine', 'scaling': True},
#                 'dbscan_cos_unscaled': {'metric': 'cosine', 'scaling': False},
#             }
#
#     clustering_overview_df = pd.DataFrame(data=np.zeros((7,3*len(hierarchical_params.keys())+2*len(kmeans_params.keys())+2*len(dbscan_params.keys())),dtype=float),
#                                           columns=["{}_Sil".format(ky) for ky in hierarchical_params.keys()] + ["{}_CH".format(ky) for ky in hierarchical_params.keys()] +
#                                                   ["{}_Dendr".format(ky) for ky in hierarchical_params.keys()] +
#                                                   ["{}_Sil".format(ky) for ky in kmeans_params.keys()] + ["{}_CH".format(ky) for ky in kmeans_params.keys()]
#                                                   + ["{}_Sil".format(ky) for ky in dbscan_params.keys()] + ["{}_CH".format(ky) for ky in dbscan_params.keys()],
#                                           index=['Max. Silhouette Avg', 'No of Clusters', 'Calinski-Harabasz', 'Largest Dendro Gap', 'Min Samples', 'eps', 'Share of outliers'])
#
#     cons.get_time("Time for data preparation", tic)
#
#     transl = {'Sil': 'silhouette_avg', 'CH': 'calinski_harabasz_score', 'Dendr': 'largest_dendro_gap'}
#
# #****************************
# # DBSCAN Clustering  --> Maybe apply DBSCAN also on server with a more exhaustive eps-list
# #****************************
#
#     print("Start DBSCAN Clustering...")
#     t_dbscan = time.time()
#
#     # The almost same reasons as for K-Means were taken to come up with the minsampleslist
#     minsampleslist = np.arange(2,205,1)
#
#     for i, ky in enumerate(dbscan_params.keys()):
#         print("Variant {}: {} of {}".format(ky, i + 1, len(kmeans_params)))
#
#         # In the following, we construct the eps-list based on the distances between the clustering data
#         # Please note: we start at the smallest distance, increase by the smallest distance and go up to half of the
#         # space in scaled data, but only to a fourth of the space with unscaled data.
#         if dbscan_params[ky]['scaling']:
#             clustering_data = som.codebook.matrix
#             scaler = 2
#         else:
#             clustering_data = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)
#             scaler = 4
#         dists = pairwise_distances(clustering_data, metric=dbscan_params[ky]['metric'], n_jobs=-1)
#         upper = np.triu(dists, k=1)
#         upper_flat = upper.reshape(1, upper.shape[0] * upper.shape[1])
#         upper_flat = upper_flat[upper_flat > 0]
#
#         # However, this procedure still came up with way too large eps-list for cosine, therefore, we inspected the data
#         # and decided for a 0.01 increase.
#         if not dbscan_params[ky]['metric'] == 'cosine':
#             epslist = np.arange(min(upper_flat), max(upper_flat) / scaler, min(upper_flat))
#         else:
#             epslist = np.arange(min(upper_flat), max(upper_flat) / scaler, 0.01)
#             continue  # Unfortunately, cosine did not work the way we wanted therefore, we skip this analysis
#
#         dbscan_clusterer = DBSCAN(eps=0.5, min_samples=5, metric=dbscan_params[ky]['metric'], n_jobs=-2)
#         dbscan_clustering = cons.Clustering_Analyzer(dbscan_clusterer, som, {'min_samples': minsampleslist, 'eps': epslist},
#                                                      scaling=dbscan_params[ky]['scaling'],
#                                                            saving={'title': 'DBSCAN {}'.format(ky),
#                                                                    'savepath': resultpath})
#         df = dbscan_clustering.make_score_dataframe()
#         df.to_excel(os.path.join(resultpath, "DBSCAN {}.xlsx".format(ky)))
#         for x in ('Sil', 'CH'):
#             df_x = df.copy()
#             df_x = df_x[df_x[transl[x]] == df_x[transl[x]].max()]
#             metalist = [df_x.loc[df_x.index[0], 'silhouette_avg'], df_x.loc[df_x.index[0], 'n_clusters'],
#                         df_x.loc[df_x.index[0], 'calinski_harabasz_score'], np.nan, df_x.loc[df_x.index[0], 'min_samples'], df_x.loc[df_x.index[0], 'eps'],
#                         df_x.loc[df_x.index[0], 'outlier_share']]
#             clustering_overview_df["{}_{}".format(ky, x)] = metalist
#
#     del dbscan_clustering
#     del dbscan_clusterer
#
#     cons.get_time("Time for DBSCAN clustering", t_dbscan)
#
#     savepath = os.path.join(resultpath, "Clustering_Overview.xlsx")
#     clustering_overview_df.to_excel(savepath, sheet_name='Clustering_Overview', na_rep='-')
#
#     cons.get_time("Total Time", tic) # last run only DBSCAN (without cosine-versions): ca. 05:30:25;


#----------------------------------------------------------------------------------------------------------------------
# Training the classifier
#----------------------------------------------------------------------------------------------------------------------
#
# #****************************
# # Training only with Mobility-data (since housing not stable)
# #****************************
#
# tic = time.time()
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption", 'Mobility')
#
# with open(os.path.join(datapath, 'rf_tuning_data.pickle'), 'rb') as f:
#          rf_data = pickle.load(f)
#
#
# # First, we only look at mobility (since data for housing is not so stable)
# X = rf_data['X_mob']
# y = rf_data['y']
#
# rf_tuning_params = {'default_gini': {'n_estimators': [1000, 2000, 3000], 'criterion': 'gini', 'max_features': 'sqrt'},
#                     'half_gini': {'n_estimators': [1000, 2000, 3000], 'criterion': 'gini', 'max_features': round(0.5*(X.shape[1]**0.5))},
#                     'twice_gini': {'n_estimators': [1000, 2000, 3000], 'criterion': 'gini', 'max_features': round(2*(X.shape[1]**0.5))},
#                     'max_gini': {'n_estimators': [1000, 2000, 3000], 'criterion': 'gini', 'max_features': None},
# }
#
#
# # Define the "base estimator" and then tune the classifiers
# rf_clf = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1, bootstrap=True,
#                                 oob_score=True, n_jobs=-1, warm_start=False, class_weight='balanced')
#
# rf_clf_tuner = cons.RFClassifierTuner(rf_clf, rf_tuning_params, X, y)
#
# # Store the results to excel as well as to a pickle-file
# rf_clf_tuner.save2excel(os.path.join(resultpath, 'rf_gini_tuner.xlsx'))
#
# rf_clf_dict = {'rf_clf_tuner': rf_clf_tuner, 'data': rf_data['data'], 'mobility_demand': rf_data['mobility_demand']}
# with open(os.path.join(resultpath, 'rf_gini_tuner.pickle'), 'wb') as f:
#     pickle.dump(rf_clf_dict, f)
#
# # We also store the individual RFs as well as the data
#
# for ky in rf_clf_tuner.scores_cv_.keys():
#     with open(os.path.join(resultpath, '{}.pickle'.format(ky)), 'wb') as f:
#         pickle.dump(getattr(rf_clf_tuner, ky), f)
# data = {'X_tune': rf_clf_tuner.X_tune_, 'y_tune': rf_clf_tuner.y_tune_, 'y_test': rf_clf_tuner.y_test_, 'X_test': rf_clf_tuner.X_test_}
# with open(os.path.join(resultpath, 'gini_data.pickle'), 'wb') as f:
#     pickle.dump(data, f)
#
#
# cons.get_time("Time for Training Gini-Classifier", tic)  # last run: 00:52:56
#
# t_entr = time.time()
#
# rf_tuning_params = {'default_entropy': {'n_estimators': [1000, 2000, 3000], 'criterion': 'entropy', 'max_features': 'sqrt'},
#                     'half_entropy': {'n_estimators': [1000, 2000, 3000], 'criterion': 'entropy', 'max_features': round(0.5*(X.shape[1]**0.5))},
#                     'twice_entropy': {'n_estimators': [1000, 2000, 3000], 'criterion': 'entropy', 'max_features': round(2*(X.shape[1]**0.5))},
#                     'max_entropy': {'n_estimators': [1000, 2000, 3000], 'criterion': 'entropy', 'max_features': None},
# }
#
#
# # Define the "base estimator" and then tune the classifiers
# rf_clf = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1, bootstrap=True,
#                                 oob_score=True, n_jobs=-1, warm_start=False, class_weight='balanced')
#
# rf_clf_tuner = cons.RFClassifierTuner(rf_clf, rf_tuning_params, X, y)
#
# # Store the results to excel as well as to a pickle-file
# rf_clf_tuner.save2excel(os.path.join(resultpath, 'rf_entropy_tuner.xlsx'))
#
# rf_clf_dict = {'rf_clf_tuner': rf_clf_tuner, 'data': rf_data['data'], 'mobility_demand': rf_data['mobility_demand']}
# with open(os.path.join(resultpath, 'rf_entropy_tuner.pickle'), 'wb') as f:
#     pickle.dump(rf_clf_dict, f)
#
# # We also store the individual RFs as well as the data
#
# for ky in rf_clf_tuner.scores_cv_.keys():
#     with open(os.path.join(resultpath, '{}.pickle'.format(ky)), 'wb') as f:
#         pickle.dump(getattr(rf_clf_tuner, ky), f)
# data = {'X_tune': rf_clf_tuner.X_tune_, 'y_tune': rf_clf_tuner.y_tune_, 'y_test': rf_clf_tuner.y_test_, 'X_test': rf_clf_tuner.X_test_}
# with open(os.path.join(resultpath, 'entropy_data.pickle'), 'wb') as f:
#     pickle.dump(data, f)
#
# cons.get_time("Time for Training Entropy-Classifier", t_entr)  # last run: 00:42:02
# cons.get_time("Total Time", tic)  # last run: 01:34:58

#****************************
# Training with Mobility and Housing data
#****************************
#
# tic = time.time()
# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption", 'Housing')
#
# with open(os.path.join(datapath, 'rf_tuning_data.pickle'), 'rb') as f:
#          rf_data = pickle.load(f)
#
#
# # Second, we only look at mobility (since data for housing is not so stable)
# X = rf_data['X_mob_hus']
# y = rf_data['y']
#
# rf_tuning_params = {'default_gini': {'n_estimators': [1000, 2000, 3000], 'criterion': 'gini', 'max_features': 'sqrt'},
#                     'half_gini': {'n_estimators': [1000, 2000, 3000], 'criterion': 'gini', 'max_features': round(0.5*(X.shape[1]**0.5))},
#                     'twice_gini': {'n_estimators': [1000, 2000, 3000], 'criterion': 'gini', 'max_features': round(2*(X.shape[1]**0.5))},
#                     'max_gini': {'n_estimators': [1000, 2000, 3000], 'criterion': 'gini', 'max_features': None},
# }
#
#
# # Define the "base estimator" and then tune the classifiers
# rf_clf = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1, bootstrap=True,
#                                 oob_score=True, n_jobs=-1, warm_start=False, class_weight='balanced')
#
# rf_clf_tuner = cons.RFClassifierTuner(rf_clf, rf_tuning_params, X, y)
#
# # Store the results to excel as well as to a pickle-file
# rf_clf_tuner.save2excel(os.path.join(resultpath, 'rf_gini_tuner.xlsx'))
#
# rf_clf_dict = {'rf_clf_tuner': rf_clf_tuner, 'data': rf_data['data'], 'mobility_demand': rf_data['mobility_demand']}
# with open(os.path.join(resultpath, 'rf_gini_tuner.pickle'), 'wb') as f:
#     pickle.dump(rf_clf_dict, f)
#
# # We also store the individual RFs as well as the data
#
# for ky in rf_clf_tuner.scores_cv_.keys():
#     with open(os.path.join(resultpath, '{}.pickle'.format(ky)), 'wb') as f:
#         pickle.dump(getattr(rf_clf_tuner, ky), f)
# data = {'X_tune': rf_clf_tuner.X_tune_, 'y_tune': rf_clf_tuner.y_tune_, 'y_test': rf_clf_tuner.y_test_, 'X_test': rf_clf_tuner.X_test_}
# with open(os.path.join(resultpath, 'gini_data.pickle'), 'wb') as f:
#     pickle.dump(data, f)
#
#
# cons.get_time("Time for Training Gini-Classifier", tic)  # last run: 00:52:56
#
# t_entr = time.time()
#
# rf_tuning_params = {'default_entropy': {'n_estimators': [1000, 2000, 3000], 'criterion': 'entropy', 'max_features': 'sqrt'},
#                     'half_entropy': {'n_estimators': [1000, 2000, 3000], 'criterion': 'entropy', 'max_features': round(0.5*(X.shape[1]**0.5))},
#                     'twice_entropy': {'n_estimators': [1000, 2000, 3000], 'criterion': 'entropy', 'max_features': round(2*(X.shape[1]**0.5))},
#                     'max_entropy': {'n_estimators': [1000, 2000, 3000], 'criterion': 'entropy', 'max_features': None},
# }
#
#
# # Define the "base estimator" and then tune the classifiers
# rf_clf = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1, bootstrap=True,
#                                 oob_score=True, n_jobs=-1, warm_start=False, class_weight='balanced')
#
# rf_clf_tuner = cons.RFClassifierTuner(rf_clf, rf_tuning_params, X, y)
#
# # Store the results to excel as well as to a pickle-file
# rf_clf_tuner.save2excel(os.path.join(resultpath, 'rf_entropy_tuner.xlsx'))
#
# rf_clf_dict = {'rf_clf_tuner': rf_clf_tuner, 'data': rf_data['data'], 'mobility_demand': rf_data['mobility_demand']}
# with open(os.path.join(resultpath, 'rf_entropy_tuner.pickle'), 'wb') as f:
#     pickle.dump(rf_clf_dict, f)
#
# # We also store the individual RFs as well as the data
#
# for ky in rf_clf_tuner.scores_cv_.keys():
#     with open(os.path.join(resultpath, '{}.pickle'.format(ky)), 'wb') as f:
#         pickle.dump(getattr(rf_clf_tuner, ky), f)
# data = {'X_tune': rf_clf_tuner.X_tune_, 'y_tune': rf_clf_tuner.y_tune_, 'y_test': rf_clf_tuner.y_test_, 'X_test': rf_clf_tuner.X_test_}
# with open(os.path.join(resultpath, 'entropy_data.pickle'), 'wb') as f:
#     pickle.dump(data, f)
#
# cons.get_time("Time for Training Entropy-Classifier", t_entr)  # last run: 00:54:57
# cons.get_time("Total Time", tic)  # last run: 01:41:58

#----------------------------------------------------------------------------------------------------------------------
# Classifier calibration with isotonic regression
#----------------------------------------------------------------------------------------------------------------------

# datapath = os.path.join(os.path.abspath("."), "Data", "Consumption")
# resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")
#
# with open(os.path.join(datapath, "twice_gini_1000.pickle"), 'rb') as f:
#     rf = pickle.load(f)
#
# with open(os.path.join(datapath, "gini_data.pickle"), 'rb') as f:
#     data = pickle.load(f)
#
# with open(os.path.join(datapath, "rf_tuning_data.pickle"),'rb') as f:
#     tuning_data = pickle.load(f)
#
#
# cccv = CalibratedClassifierCV(rf, cv='prefit', method='isotonic')  # was also computed with method='isotonic'
# cccv.fit(data['X_test'], data['y_test'])
#
# del data, rf, tuning_data
#
# with open(os.path.join(resultpath, "calibrated_classifier_isotonic.pickle"), 'wb') as f:
#     pickle.dump(cccv, f)



#----------------------------------------------------------------------------------------------------------------------
# Compute Consumption Demand
#----------------------------------------------------------------------------------------------------------------------
cpus = mp.cpu_count() - 9  # count CPUs

# Most of the following code was kindly provided by Rene Buffat
def consumptiondem_worker(q, clfmob, clfhus, archedict, mc_data, habe_attributes, datapath, savepath, wid): # if I got it right: q is the queue (list of municipality numbers) while wid stands for the CPU

    finished = False
    while not finished:
        try:

            vals = q.get() # take a value from the queue

            if vals == 'killitwithfire_worker':  # if-loop to terminate the processing
                logging.info(
                    "worker {} finished - received good kill".format(wid))
                finished = True
                break
            bfsnr = vals

            cons.do_consumption_demand(bfsnr, clfmob, clfhus, archedict, mc_data, habe_attributes, datapath, savepath)   # core of the function: calling the "actual" function

        except Empty:
            logging.info("worker {} finished - empty".format(wid))
            finished = True
            break
        except Exception as e:
            logging.exception("wid: {} / {}".format(wid, str(e)))

#in the case of multiprocessing, the main-statement is necessary
if __name__ == '__main__':
    t = time.time()
    print("Initialization/loading data...")
    datapath = os.path.join(os.path.abspath("."), "Data", "Consumption", "Export4Linux")
    resultpath = os.path.join(os.path.abspath("."), "Results", "Consumption")

    with open(os.path.join(datapath, "bfsnrs.pickle"), 'rb') as f:
        bfsnr_all = pickle.load(f)

    with open(os.path.join(datapath, "calibrated_classifier_mob.pickle"), 'rb') as f:
        clfmob = pickle.load(f)

    with open(os.path.join(datapath, "calibrated_classifier_hus.pickle"), 'rb') as f:
        clfhus = pickle.load(f)

    with open(os.path.join(datapath, "archedict.pickle"), 'rb') as f:
        archedict = pickle.load(f)

    with open(os.path.join(datapath, "mc_data.pickle"), 'rb') as f:
        mc_data = pickle.load(f)

    with open(os.path.join(datapath, "habe_attributes.pickle"), 'rb') as f:
        habe_attributes = pickle.load(f)

    cons.get_time("Time for intialization", t)  # last run: 00:00:07
    t_dem = time.time()
    print("start multi-processing...")

    q = mp.Queue()  # create the queue
    ps = [mp.Process(
        target=consumptiondem_worker,
        args=(q, clfmob, clfhus, archedict, mc_data, habe_attributes, datapath, resultpath, i)) for i in range(cpus)]

    #bfsnr_all = [4566, 4001, 230, 4946] #for debugging
    for bfsnr in bfsnr_all:
        q.put(bfsnr)

    #for termination?
    for _ in range(cpus):
        q.put("killitwithfire_worker")

    #start the multi-processing:
    for p in ps:
        p.start()

    for p in ps:
        p.join()

    cons.get_time("Time for computing demand", t_dem)  # last run: 01:13:45
    cons.get_time("Total Time", t)  # last run: 01:13:52



