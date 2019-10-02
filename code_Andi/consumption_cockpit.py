from Andi_model import consumption as cons
import time
import pickle
import sys
sys.path.insert(0, r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\04_HEIA-Tools")
from Clustering_Tools import *
import mobility as mob

#----------------------------------------------------------------------------------------------------------------------
# Data Screening: Create Boxplots, Violinplots and Histograms for all HABE-data
#----------------------------------------------------------------------------------------------------------------------

# # Figures for "Ausgaben": these lines were run for levels 'all', 3, 4, 5, 7

# t = time.time()
# wb = xlrd.open_workbook(r"D:\froemelt\Documents\Andi\02_Doktorat\00_Data\HABE\HABE091011_Datenbeschreibung_131125UOe.xlsx")
# levels = ['all', 3, 4, 5, 7]
# elapsed = time.time() - t
# print("elapsed time for initialization: {} s".format(round(elapsed, 1)))
#
# for level in levels:
#     t = time.time()
#     cons.create_consumptionfigs(wb, 'Ausgaben', 6, 'CHF/month', level=level)
#     elapsed = time.time() - t
#     print("elapsed time for level {}: {} s".format(level, round(elapsed, 1)))

# # Figures for "Konsumgueter":
#
# t = time.time()
# wb = xlrd.open_workbook(r"D:\froemelt\Documents\Andi\02_Doktorat\00_Data\HABE\HABE091011_Datenbeschreibung_131125UOe.xlsx")
# elapsed = time.time() - t
# print("elapsed time for initialization: {} s".format(round(elapsed, 1)))
#
# t = time.time()
# cons.create_consumptionfigs(wb, 'Konsumgueter', 3, 'count (-)')
# elapsed = time.time() - t
# print("elapsed time: {} s".format(round(elapsed, 1)))

# # Figures for "Mengen": these lines were run for levels 'all', 5, 6, 7 (ATTENTION: levels 5 and 6 are one level)
#
# t = time.time()
# wb = xlrd.open_workbook(r"D:\froemelt\Documents\Andi\02_Doktorat\00_Data\HABE\HABE091011_Datenbeschreibung_131125UOe.xlsx")
# levels = ['all', 5, 6, 7]
# elapsed = time.time() - t
# print("elapsed time for initialization: {} s".format(round(elapsed, 1)))
#
# for level in levels:
#     t = time.time()
#     cons.create_consumptionfigs(wb, 'Mengen', 4, 'kg or lt/month', level=level)
#     elapsed = time.time() - t
#     print("elapsed time for level {}: {} s".format(level, round(elapsed, 1)))

# # Figures for "Standard" (please note that not all figures will make sense as some variables are categorical):
#
# t = time.time()
# wb = xlrd.open_workbook(r"D:\froemelt\Documents\Andi\02_Doktorat\00_Data\HABE\HABE091011_Datenbeschreibung_131125UOe.xlsx")
# elapsed = time.time() - t
# print("elapsed time for initialization: {} s".format(round(elapsed, 1)))
#
# t = time.time()
# cons.create_consumptionfigs(wb, 'Standard', 3, 'CHF/month or count (-)')
# elapsed = time.time() - t
# print("elapsed time: {} s".format(round(elapsed, 1)))

# Figures for "Personen" (please note that not all figures will make sense as some variables are categorical):

# t = time.time()
# wb = xlrd.open_workbook(r"D:\froemelt\Documents\Andi\02_Doktorat\00_Data\HABE\HABE091011_Datenbeschreibung_131125UOe.xlsx")
# elapsed = time.time() - t
# print("elapsed time for initialization: {} s".format(round(elapsed, 1)))
#
# t = time.time()
# cons.create_consumptionfigs(wb, 'Personen', 3, 'count (-)')
# elapsed = time.time() - t
# print("elapsed time: {} s".format(round(elapsed, 1)))

#----------------------------------------------------------------------------------------------------------------------
# DEPRECATED: Data Pre-Processing: Preparation of HH-Predictors
#----------------------------------------------------------------------------------------------------------------------

# t = time.time()
# cons.habe_hh_predictor_builder()
# elapsed = time.time() - t
# print("elapsed time: \n{} s\n{} min\n{} h".format(round(elapsed, 1), round(elapsed/60, 1), round(elapsed/3600, 1)))

#----------------------------------------------------------------------------------------------------------------------
# Data Pre-Processing: Preparation of HH-data
#----------------------------------------------------------------------------------------------------------------------
# t = time.time()
# excel = {'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#          'overview': 'Overview',
#          'water': 'Water_prices',
#          'ww': 'Wastewater_prices',
#          'waste': 'Waste_prices',
#          'electricity': 'Electricity_prices',
#          'energy': 'Shares of heat energy carriers'}
# cons.habe_hhs_preparer(excel)
#
# cons.get_time("Time for preparing HH-data", t)  # last run: 00:02:57

#----------------------------------------------------------------------------------------------------------------------
# Create seasonal graphs to set correct filter attributes for pattern recognition
#----------------------------------------------------------------------------------------------------------------------

# tic = time.time()
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
# filtereddata = cons.SKLData(conn, 'working_tables.habe_hh_prepared',  meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('Pattern Recognition 1', 'Pattern Recognition 2')
#                                       })
# del filtereddata._conn_
# conn.close()
#
# attrs = [attr for attr in filtereddata.attributes_ if not attr.startswith("char_") and not attr.startswith("cg_")]
# housing_attrs = ['a571', 'a571100', 'a571201', 'a571202', 'a571203', 'a571204', 'a571205', 'a571301', 'a571302',
#                  'a571303', 'mx571202', 'mx571203', 'mx571302', 'mx571303']
# attrs += housing_attrs
# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\filtereddata_dwellingdata.pdf",
#                           attrs, filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:03:43

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\a50_a51.pdf",
#                           ['a50', 'a51', 'a511'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:12

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\meat.pdf",
#                           ['a511201', 'a511202', 'a511203', 'a511204', 'a511205', 'a511206', 'a511207', 'a511208', 'a511209',
#                            'a511210', 'a511211', 'a511212', 'a511213'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:35
#
# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\fish.pdf",
#                           ['a511301', 'a511302', 'a511303', 'a511304', 'a511305'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:14

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\speiseöle.pdf",
#                           ['a5115', 'a511501', 'a511502', 'a511503', 'a511504', 'a511505'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:16

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\fruits.pdf",
#                           ['a511601','a511602','a511603','a511604','a511605','a511606','a511607','a511608','a511609',
#                            'a511610','a511611','a511612','a511613'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:35

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\sweets.pdf",
#                           ['a511801','a511802','a511803','a511804','a511805','a511806','a511807'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:24

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\alcohol.pdf",
#                           ['a5211','a521101','a521102','a5212','a521201','a521202','a521203','a521204','a521205','a521206','a521207',
#                            'a521208','a521209','a5213','a521300'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:40

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\a53_a531.pdf",
#                           ['a53', 'a531'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:07

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\a561_a562.pdf",
#                           ['a561', 'a562'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:07

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\recreationalActivities.pdf",
#                           ['a66','a661','a6611','a661100','a6612','a661200','a6613','a661301','a661302','a6614','a661401',
#                            'a661402','a661403','a662','a6621','a662100','a6622','a662201','a662202','a662203','a662204',
#                            'a6623','a662301','a662302','a662303','a6624','a662400','a6625','a662501','a662502','a663','a6631',
#                            'a663101','a663102','a663103','a663104','a663105','a663106','a663107','a663108','a663109','a6632',
#                            'a663201','a663202','a663203','a663204','a663205','a663206','a663207','a663208','a663209','a6633',
#                            'a663300','a664','a6641','a664100','a6642','a664201','a664202','a6643','a664301','a664302',
#                            'a665','a6650','a665000'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:02:42

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\a360001.pdf",
#                           ['a360001'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:07

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\presents.pdf",
#                           ['a442','a4421','a442101','a442102','a4422','a442200','a4423','a442301','a442302','a442303','a442304',
#                            'a442305','a442306','a442307','a442308','a4424','a442401','a442402','a4425','a442500','a4426',
#                            'a442600','a4427','a442701','a442702','a442703','a442704','a442705','a442706','a4428','a442801',
#                            'a442802','a442803'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:01:21

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\meat_amounts.pdf",
#                           ['m5112','m511201','m511202','m511203','m511204','m511205','m511206','m511207','m511208','m511209',
#                            'm511210','m511211','m511212','m511213'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:35

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\fish_amounts.pdf",
#                           ['m5113','m511301','m511302','m511303','m511304','m511305'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:17

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\speiseöle_amounts.pdf",
#                           ['m5115','m511501','m511502','m511503','m511504','m511505'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:16

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\fruits_amounts.pdf",
#                           ['m5116', 'm511601','m511602', 'm511603', 'm511604', 'm511605','m511606','m511607','m511608',
#                            'm511609','m511610','m511611','m511612','m511613'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:35

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\sweets_amounts.pdf",
#                           ['m5118a','m511801','m511802','m511803','m511804','m511806'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:18

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\Consumption-Data-Screening (seasonal graphs)\alcohol_amounts.pdf",
#                           ['m5211','m521101','m521102','m5212','m521201','m521202','m521203','m521204','m521205','m521206',
#                            'm521207','m521208','m521209','m5213','m521300'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:40

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\03_Consumption-Data-Screening (seasonal graphs)\vegetables_amounts&exp.pdf",
#                           ['a5117', 'a511701', 'a511702', 'a511703', 'a511704', 'a511705', 'a511706', 'a511707', 'a511708',
#                            'a511709','a511710','a511711','a511712','a511713','a511714','a511715','m5117a','m511701','m511702',
#                            'm511704','m511705','m511706','m511707','m511708','m511709','m511710','m511711','m511712',
#                            'm511713','m511714','m511715'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:01:26

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\03_Consumption-Data-Screening (seasonal graphs)\non-alcoholic_amounts&exp.pdf",
#                           ['a5122','a512201','a512202','a512203','a512204','a512205','m5122','m512201','m512202','m512203',
#                            'm512204','m512205'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:32

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\03_Consumption-Data-Screening (seasonal graphs)\hotels&restaurants_details.pdf",
#                           ['a5311', 'a531101', 'a531102', 'a531103','a5312', 'a531201','a531202','a531203','a5313','a531301',
#                            'a531302','a531303', 'a532','a5320','a532001','a532002'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:43

# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\03_Consumption-Data-Screening (seasonal graphs)\housing_toplevels.pdf",
#                           ['a57', 'a571', 'a572', 'a573'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:13
#
# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\03_Consumption-Data-Screening (seasonal graphs)\public_transport.pdf",
#                           ['a622', 'a6221', 'a622101', 'a622102', 'a6222', 'a622201', 'a622202','a6223','a622300','a6224',
#                            'a622400','a6225','a622501','a622502','a622503','a622504','a6226','a622600'], filtereddata.varnames_)
# cons.get_time("Total Time", tic)  # last run: 00:00:47


#----------------------------------------------------------------------------------------------------------------------
# Data Pre-Processing: NK-Modelling
#----------------------------------------------------------------------------------------------------------------------
# ****************************
# "Pauschale Nebenkosten": Model-Selection
# ****************************
# tic = time.time()
# print("preparing and extracting data...")
# nk_ky = 'NK'
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\04_NK-Models\01_NK_pauschal"
#
# # 1st STEP: extract data and preparing input data--> at the moment only electricity
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# alldata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', meta='haushaltid',
#                              excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                     'sheet_with_codes': 'Overview',
#                                     'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                     'filter_column_names': ('NK-Model all LR KNN 1', 'NK-Model all')
#                                     })
# dict_of_data = cons.nk_data_preparer('alldata_nk', alldata_nk, nk_ky)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# filtereddata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter LR KNN 1', 'NK-Model-Filter')
#                                       })
#
# dict_of_data_filt = cons.nk_data_preparer('filtereddata_nk', filtereddata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_filt)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# basicdata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter basics LR KNN 1', 'NK-Model-Filter basics')
#                                       })
#
# dict_of_data_basic = cons.nk_data_preparer('basicdata_nk', basicdata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_basic)
#
# with open(os.path.join(savepath, "{}_LR-KNN-Data.pickle".format(nk_ky)), 'wb') as f:
#     pickle.dump(dict_of_data, f)
#
# cons.get_time("Time for preparing data", tic)  # last run: 00:00:16
#
# # 2nd STEP: Training of data / tuning hyperparameters
# t_mod = time.time()
# print("training regression-models...")
#
# lasso_df = pd.DataFrame(data=np.zeros((9, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'alpha', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("Lasso-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     lasso_nk = cons.Lasso_NK(dict_of_data[ky][0], dict_of_data[ky][1], lasso_df, ky)
#     with open(os.path.join(savepath, "{}_{}_LassoClass_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(lasso_nk, f)
#     cons.nk_model_basic_diagnostic_plot(lasso_nk, savepath, "{}_{}_Lasso_basic_diagnostic_plot".format(nk_ky, ky))
# lasso_df.to_excel(os.path.join(savepath, "{}_LassoResults_Modelselection.xlsx".format(nk_ky)), sheet_name='LassoResults', na_rep='-')
#
# lasso2_df = pd.DataFrame(data=np.zeros((9, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'alpha', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("PCA-Lasso-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     lasso2_nk = cons.Lasso_NK(dict_of_data[ky][0], dict_of_data[ky][1], lasso2_df, ky, pre_pca=True)
#     with open(os.path.join(savepath, "{}_{}_PCALassoClass_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(lasso2_nk, f)
#     cons.nk_model_basic_diagnostic_plot(lasso2_nk, savepath, "{}_{}_PCALasso_basic_diagnostic_plot".format(nk_ky, ky))
# lasso2_df.to_excel(os.path.join(savepath, "{}_PCALassoResults_Modelselection.xlsx".format(nk_ky)), sheet_name='PCALassoResults', na_rep='-')
#
# knn_df = pd.DataFrame(data=np.zeros((16, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'k', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max',
#                                'k=1', 'k=3', 'k=5', 'k=10', 'k=20', 'k=50', 'k=100'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("KNN-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     knn_nk = cons.KNN_NK(dict_of_data[ky][0], dict_of_data[ky][1], knn_df, ky, dist='euclidean')
#     with open(os.path.join(savepath, "{}_{}_KNNeuclClass_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(knn_nk, f)
#     cons.nk_model_basic_diagnostic_plot(knn_nk, savepath, "{}_{}_KNNeucl_basic_diagnostic_plot".format(nk_ky, ky))
# knn_df.to_excel(os.path.join(savepath, "{}_KNN-euclidean-Results_Modelselection.xlsx".format(nk_ky)), sheet_name='KNN-euclidean-Results', na_rep='-')
#
# # knn2_df = pd.DataFrame(data=np.zeros((9, len(dict_of_data)), dtype=float),
# #                         columns=[ky for ky in dict_of_data.keys()],
# #                         index=['R2', 'MSE', 'k', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max',
# #                               'k=1', 'k=3', 'k=5', 'k=10', 'k=20', 'k=50', 'k=100'])])
# #
# # for i, ky in enumerate(dict_of_data.keys()):
# #     print("KNN-Mahalanobis-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
# #     knn_nk = cons.KNN_NK(dict_of_data[ky][0], dict_of_data[ky][1], knn2_df, ky, dist='mahalanobis')
# # savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\NK-Models\NK_KNN-mahalanobis-Results_Modelselection.xlsx"
# # knn2_df.to_excel(savepath, sheet_name='KNN-mahalanobis-Results', na_rep='-')
#
# cons.get_time("time for training models", t_mod)  # last run: 01:22:09
#
# cons.get_time("Total time", tic)  # last run: 01:22:26

#****************************
# "Pauschale Nebenkosten": Model-Selection --> exporting data for RF-Modelling
#****************************
# tic = time.time()
# print("extracting and storing data...")
# nk_ky = 'NK'
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\04_NK-Models\01_NK_pauschal\Data_for_RF_on_Linux"
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# alldata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, meta='haushaltid',
#                              excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                     'sheet_with_codes': 'Overview',
#                                     'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                     'filter_column_names': ('NK-Model all RF 1', 'NK-Model all')
#                                     })
#
# dict_of_data = cons.nk_data_preparer('alldata_nk', alldata_nk, nk_ky)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# filtereddata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter RF 1', 'NK-Model-Filter')
#                                       })
#
# dict_of_data_filt = cons.nk_data_preparer('filtereddata_nk', filtereddata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_filt)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# basicdata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter basics RF 1', 'NK-Model-Filter basics')
#                                       })
#
# dict_of_data_basic = cons.nk_data_preparer('basicdata_nk', basicdata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_basic)
#
# with open(os.path.join(savepath, "dict_of_data.pickle"), 'wb') as f:
#     pickle.dump(dict_of_data, f)
#
# cons.get_time("Time for extracting/storing data", tic) # last run: 00:00:16

# ****************************
# Modelling "Pauschale Nebenkosten"
# ****************************
#
# tic = time.time()
# print("modelling...")
# nk_ky = 'NK'
#
# # 1st STEP: extract data and preparing input data
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# alldata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', meta='haushaltid',
#                              excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                     'sheet_with_codes': 'Overview',
#                                     'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                     'filter_column_names': ('NK-Model all LR KNN 1', 'NK-Model all')
#                                     })
#
# dict_of_data = cons.nk_data_preparer('alldata_nk', alldata_nk, nk_ky)
#
# # 2nd STEP: modelling "Pauschale Nebenkosten"
# lasso2_df = pd.DataFrame(data=np.zeros((9, 2), dtype=float),
#                         columns=['alldata_nk_owners','alldata_nk_renters'],
#                         index=['R2', 'MSE', 'alpha', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max'])
#
# lasso_nk_owners = cons.Lasso_NK(dict_of_data['alldata_nk_owners'][0], dict_of_data['alldata_nk_owners'][1], lasso2_df, 'alldata_nk_owners', pre_pca=True, modelparams={'alpha': 0.00107525891720122})
# lasso_nk_renters = cons.Lasso_NK(dict_of_data['alldata_nk_renters'][0], dict_of_data['alldata_nk_renters'][1], lasso2_df, 'alldata_nk_renters', pre_pca=True, modelparams={'alpha': 0.00266835868391905})
#
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\04_NK-Models\01_NK_pauschal\FinalModel"
#
# with open(os.path.join(savepath, "{}_{}_PCALassoClass_FinalModel.pickle".format(nk_ky, 'alldata_nk_owners')), 'wb') as f:
#         pickle.dump(lasso_nk_owners, f)
# cons.nk_model_basic_diagnostic_plot(lasso_nk_owners, savepath, "{}_{}_PCALasso_basic_diagnostic_plot".format(nk_ky, 'alldata_nk_owners'))
# with open(os.path.join(savepath, "{}_{}_PCALassoClass_FinalModel.pickle".format(nk_ky, 'alldata_nk_renters')),'wb') as f:
#     pickle.dump(lasso_nk_renters, f)
# cons.nk_model_basic_diagnostic_plot(lasso_nk_renters, savepath,"{}_{}_PCALasso_basic_diagnostic_plot".format(nk_ky, 'alldata_nk_renters'))
#
# lasso2_df.to_excel(os.path.join(savepath, "{}_PCALassoResults_FinalModel.xlsx".format(nk_ky)), sheet_name='PCALassoResults', na_rep='-')
#
# cons.get_time("Time for modelling", tic) # last run: 00:00:22
#
# # 3rd STEP: writing "Pauschale NK" to DB
# t_write = time.time()
# print('writing to pg-database...')
#
# cons.write_nk_results_to_pg([dict_of_data['alldata_nk_owners'][1], dict_of_data['alldata_nk_renters'][1]], [lasso_nk_owners.y_targ_destd_, lasso_nk_renters.y_targ_destd_], nk_ky)
#
# cons.get_time("Time for writing", t_write) # last run: 00:01:18
# cons.get_time("Total Time", tic) # last run: 00:01:40

#****************************
# "Kehricht": Model-Selection
#****************************
# tic = time.time()
# print("preparing and extracting data...")
# nk_ky = 'K'
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\04_NK-Models\02_Kehricht"
#
# # 1st STEP: extract data and preparing input data
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# alldata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', amount_as_target=True, meta='haushaltid',
#                              excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                     'sheet_with_codes': 'Overview',
#                                     'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                     'filter_column_names': ('NK-Model all (Kehricht)', 'NK-Model all')
#                                     })
# dict_of_data = cons.nk_data_preparer('alldata_nk', alldata_nk, nk_ky)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# filtereddata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', amount_as_target=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter 1 (Kehricht)', 'NK-Model-Filter 2 (Kehricht)')
#                                       })
#
# dict_of_data_filt = cons.nk_data_preparer('filtereddata_nk', filtereddata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_filt)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# basicdata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', amount_as_target=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter basics 1 (Kehricht)', 'NK-Model-Filter basics 2 (Kehricht)')
#                                       })
#
# dict_of_data_basic = cons.nk_data_preparer('basicdata_nk', basicdata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_basic)
#
# with open(os.path.join(savepath, "{}_LR-KNN-Data.pickle".format(nk_ky)), 'wb') as f:
#     pickle.dump(dict_of_data, f)
#
# cons.get_time("Time for preparing data", tic)  # last run: 00:00:16
#
# # 2nd STEP: Training of data / tuning hyperparameters
# t_mod = time.time()
# print("training regression-models...")
#
# lasso_df = pd.DataFrame(data=np.zeros((9, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'alpha', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("Lasso-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     lasso_nk = cons.Lasso_NK(dict_of_data[ky][0], dict_of_data[ky][1], lasso_df, ky)
#     with open(os.path.join(savepath, "{}_{}_LassoClass_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(lasso_nk, f)
#     cons.nk_model_basic_diagnostic_plot(lasso_nk, savepath, "{}_{}_Lasso_basic_diagnostic_plot".format(nk_ky, ky))
# lasso_df.to_excel(os.path.join(savepath, "{}_LassoResults_Modelselection.xlsx".format(nk_ky)), sheet_name='LassoResults', na_rep='-')
#
# lasso2_df = pd.DataFrame(data=np.zeros((9, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'alpha', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("PCA-Lasso-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     lasso2_nk = cons.Lasso_NK(dict_of_data[ky][0], dict_of_data[ky][1], lasso2_df, ky, pre_pca=True)
#     with open(os.path.join(savepath, "{}_{}_PCALassoClass_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(lasso2_nk, f)
#     cons.nk_model_basic_diagnostic_plot(lasso2_nk, savepath, "{}_{}_PCALasso_basic_diagnostic_plot".format(nk_ky, ky))
# lasso2_df.to_excel(os.path.join(savepath, "{}_PCALassoResults_Modelselection.xlsx".format(nk_ky)), sheet_name='PCALassoResults', na_rep='-')
#
# knn_df = pd.DataFrame(data=np.zeros((16, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'k', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max',
#                                'k=1', 'k=3', 'k=5', 'k=10', 'k=20', 'k=50', 'k=100'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("KNN-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     knn_nk = cons.KNN_NK(dict_of_data[ky][0], dict_of_data[ky][1], knn_df, ky, dist='euclidean')
#     with open(os.path.join(savepath, "{}_{}_KNNeuclClass_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(knn_nk, f)
#     cons.nk_model_basic_diagnostic_plot(knn_nk, savepath, "{}_{}_KNNeucl_basic_diagnostic_plot".format(nk_ky, ky))
# knn_df.to_excel(os.path.join(savepath, "{}_KNN-euclidean-Results_Modelselection.xlsx".format(nk_ky)), sheet_name='KNN-euclidean-Results', na_rep='-')
#
# # knn2_df = pd.DataFrame(data=np.zeros((9, len(dict_of_data)), dtype=float),
# #                         columns=[ky for ky in dict_of_data.keys()],
# #                         index=['R2', 'MSE', 'k', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max',
# #                               'k=1', 'k=3', 'k=5', 'k=10', 'k=20', 'k=50', 'k=100'])])
# #
# # for i, ky in enumerate(dict_of_data.keys()):
# #     print("KNN-Mahalanobis-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
# #     knn_nk = cons.KNN_NK(dict_of_data[ky][0], dict_of_data[ky][1], knn2_df, ky, dist='mahalanobis')
# # savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\NK-Models\K_KNN-mahalanobis-Results_Modelselection.xlsx"
# # knn2_df.to_excel(savepath, sheet_name='KNN-mahalanobis-Results', na_rep='-')
#
# cons.get_time("time for training models", t_mod)  # last run: 00:31:06
#
# cons.get_time("Total time", tic)  # last run: 00:31:22

#****************************
# "Kehricht": Model-Selection --> exporting data for RF-Modelling
#****************************
# tic = time.time()
# print("extracting and storing data...")
# nk_ky = 'K'
#
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\04_NK-Models\02_Kehricht\Data_for_RF_on_Linux"
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# alldata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, amount_as_target=True, meta='haushaltid',
#                              excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                     'sheet_with_codes': 'Overview',
#                                     'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                     'filter_column_names': ('NK-Model all (Kehricht)', 'NK-Model all')
#                                     })
#
# dict_of_data = cons.nk_data_preparer('alldata_nk', alldata_nk, nk_ky)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# filtereddata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, amount_as_target=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter 1 (Kehricht)', 'NK-Model-Filter 2 (Kehricht)')
#                                       })
#
# dict_of_data_filt = cons.nk_data_preparer('filtereddata_nk', filtereddata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_filt)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# basicdata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, amount_as_target=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter basics 1 (Kehricht)', 'NK-Model-Filter basics 2 (Kehricht)')
#                                       })
#
# dict_of_data_basic = cons.nk_data_preparer('basicdata_nk', basicdata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_basic)
#
# with open(os.path.join(savepath, "dict_of_data_kehricht.pickle"), 'wb') as f:
#     pickle.dump(dict_of_data, f)
#
# cons.get_time("Time for extracting/storing data", tic) # last run: 00:00:17

#****************************
# Modelling "Kehricht"
#****************************
#
# tic = time.time()
# print("modelling...")
# nk_ky = 'K'
#
# # 1st STEP: extract data and preparing input data
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# alldata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, amount_as_target=True, meta='haushaltid',
#                              excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                     'sheet_with_codes': 'Overview',
#                                     'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                     'filter_column_names': ('NK-Model all (Kehricht)', 'NK-Model all')
#                                     })
#
# dict_of_data = cons.nk_data_preparer('alldata_nk', alldata_nk, nk_ky)
#
# # 2nd STEP: modelling "Kehricht"
#
# rf2_df = pd.DataFrame(data=np.zeros((10, 1), dtype=float),
#                         columns=['alldata_nk'],
#                         index=['R2', 'MSE', 'r2_oob', 'noTrees', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max'])
#
# rf2_nk = cons.RF_NK(dict_of_data['alldata_nk'][0], dict_of_data['alldata_nk'][1], rf2_df, 'alldata_nk', scaling=False, modelparams={'n_estimators': 100})
#
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\04_NK-Models\02_Kehricht\FinalModel"
#
# with open(os.path.join(savepath, "{}_{}_RF_notscaled_Class_FinalModel.pickle".format(nk_ky, 'alldata_nk')),'wb') as f:
#     pickle.dump(rf2_nk, f)
# cons.nk_model_basic_diagnostic_plot(rf2_nk, savepath, "{}_{}_RF_notscaled_basic_diagnostic_plot".format(nk_ky, 'alldata_nk'))
#
# rf2_df.to_excel(os.path.join(savepath, "{}_RF_notscaled_FinalModel.xlsx".format(nk_ky)), sheet_name='RF_notscaled', na_rep='-')
#
# cons.get_time("Time for modelling", tic) # last run: 00:34:47
#
# # 3rd STEP: writing "Kehricht" to DB
# t_write = time.time()
# print('writing to pg-database...')
#
# cons.write_nk_results_to_pg([dict_of_data['alldata_nk'][1]], [rf2_nk.y_targ_destd_], nk_ky, amounts_as_targets=True, scaling=False)
#
# cons.get_time("Time for writing", t_write) # last run: 00:00:22
# cons.get_time("Total Time", tic) # last run: 00:35:09

#****************************
# "Electricity": Model-Selection
#****************************
# tic = time.time()
# print("preparing and extracting data...")
# nk_ky = 'El'
#
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\04_NK-Models\03_Electricity"
#
# # 1st STEP: extract data and preparing input data
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# alldata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', amount_as_target=True, meta='haushaltid',
#                              excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                     'sheet_with_codes': 'Overview',
#                                     'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                     'filter_column_names': ('NK-Model all LR KNN 1', 'NK-Model all')
#                                     })
#
# dict_of_data = cons.nk_data_preparer('alldata_nk', alldata_nk, nk_ky)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# filtereddata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', amount_as_target=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter LR KNN 1', 'NK-Model-Filter (Electricity)')
#                                       })
#
# dict_of_data_filt = cons.nk_data_preparer('filtereddata_nk', filtereddata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_filt)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# basicdata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', amount_as_target=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter basics LR KNN 1', 'NK-Model-Filter basics (Electricity)')
#                                       })
#
# dict_of_data_basic = cons.nk_data_preparer('basicdata_nk', basicdata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_basic)
#
# with open(os.path.join(savepath, "{}_LR-KNN-Data.pickle".format(nk_ky)), 'wb') as f:
#     pickle.dump(dict_of_data, f)
#
# cons.get_time("Time for preparing data", tic)  # last run: 00:00:15
#
# # 2nd STEP: Training of data / tuning hyperparameters
# t_mod = time.time()
# print("training regression-models...")
#
# lasso_df = pd.DataFrame(data=np.zeros((9, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'alpha', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("Lasso-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     lasso_nk = cons.Lasso_NK(dict_of_data[ky][0], dict_of_data[ky][1], lasso_df, ky)
#     with open(os.path.join(savepath, "{}_{}_LassoClass_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(lasso_nk, f)
#     cons.nk_model_basic_diagnostic_plot(lasso_nk, savepath, "{}_{}_Lasso_basic_diagnostic_plot".format(nk_ky, ky))
# lasso_df.to_excel(os.path.join(savepath, "{}_LassoResults_Modelselection.xlsx".format(nk_ky)), sheet_name='LassoResults', na_rep='-')
#
# lasso2_df = pd.DataFrame(data=np.zeros((9, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'alpha', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("PCA-Lasso-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     lasso2_nk = cons.Lasso_NK(dict_of_data[ky][0], dict_of_data[ky][1], lasso2_df, ky, pre_pca=True)
#     with open(os.path.join(savepath, "{}_{}_PCALassoClass_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(lasso2_nk, f)
#     cons.nk_model_basic_diagnostic_plot(lasso2_nk, savepath, "{}_{}_PCALasso_basic_diagnostic_plot".format(nk_ky, ky))
# lasso2_df.to_excel(os.path.join(savepath, "{}_PCALassoResults_Modelselection.xlsx".format(nk_ky)), sheet_name='PCALassoResults', na_rep='-')
#
# knn_df = pd.DataFrame(data=np.zeros((16, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'k', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max',
#                                'k=1', 'k=3', 'k=5', 'k=10', 'k=20', 'k=50', 'k=100'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("KNN-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     knn_nk = cons.KNN_NK(dict_of_data[ky][0], dict_of_data[ky][1], knn_df, ky, dist='euclidean')
#     with open(os.path.join(savepath, "{}_{}_KNNeuclClass_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(knn_nk, f)
#     cons.nk_model_basic_diagnostic_plot(knn_nk, savepath, "{}_{}_KNNeucl_basic_diagnostic_plot".format(nk_ky, ky))
# knn_df.to_excel(os.path.join(savepath, "{}_KNN-euclidean-Results_Modelselection.xlsx".format(nk_ky)), sheet_name='KNN-euclidean-Results', na_rep='-')
#
# # knn2_df = pd.DataFrame(data=np.zeros((9, len(dict_of_data)), dtype=float),
# #                         columns=[ky for ky in dict_of_data.keys()],
# #                         index=['R2', 'MSE', 'k', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max',
# #                               'k=1', 'k=3', 'k=5', 'k=10', 'k=20', 'k=50', 'k=100'])])
# #
# # for i, ky in enumerate(dict_of_data.keys()):
# #     print("KNN-Mahalanobis-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
# #     knn_nk = cons.KNN_NK(dict_of_data[ky][0], dict_of_data[ky][1], knn2_df, ky, dist='mahalanobis')
# # savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\NK-Models\K_KNN-mahalanobis-Results_Modelselection.xlsx"
# # knn2_df.to_excel(savepath, sheet_name='KNN-mahalanobis-Results', na_rep='-')
#
# cons.get_time("time for training models", t_mod)  # last run: 02:54:19
#
# cons.get_time("Total time", tic)  # last run: 02:54:35

#****************************
# "Electricity": Model-Selection --> exporting data for RF-Modelling
#****************************
# tic = time.time()
# print("extracting and storing data...")
# nk_ky = 'El'
#
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\04_NK-Models\03_Electricity\Data_for_RF_on_Linux"
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# alldata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, amount_as_target=True, meta='haushaltid',
#                              excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                     'sheet_with_codes': 'Overview',
#                                     'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                     'filter_column_names': ('NK-Model all RF 1', 'NK-Model all')
#                                     })
#
# dict_of_data = cons.nk_data_preparer('alldata_nk', alldata_nk, nk_ky)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# filtereddata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, amount_as_target=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter RF 1', 'NK-Model-Filter (Electricity)')
#                                       })
#
# dict_of_data_filt = cons.nk_data_preparer('filtereddata_nk', filtereddata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_filt)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# basicdata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, amount_as_target=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter basics RF 1', 'NK-Model-Filter basics (Electricity)')
#                                       })
#
# dict_of_data_basic = cons.nk_data_preparer('basicdata_nk', basicdata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_basic)
#
# with open(os.path.join(savepath, "dict_of_data_electricity.pickle"), 'wb') as f:
#     pickle.dump(dict_of_data, f)
#
# cons.get_time("Time for extracting/storing data", tic) # last run: 00:00:15

#****************************
# Modelling "Electricity"
#****************************
#
# tic = time.time()
# print("modelling...")
# nk_ky = 'El'
#
# # 1st STEP: extract data and preparing input data
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# alldata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, amount_as_target=True, meta='haushaltid',
#                              excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                     'sheet_with_codes': 'Overview',
#                                     'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                     'filter_column_names': ('NK-Model all RF 1', 'NK-Model all')
#                                     })
#
# dict_of_data = cons.nk_data_preparer('alldata_nk', alldata_nk, nk_ky)
#
# # 2nd STEP: modelling "Electricity"
# rf2_df = pd.DataFrame(data=np.zeros((10, 1), dtype=float),
#                         columns=['alldata_nk'],
#                         index=['R2', 'MSE', 'r2_oob', 'noTrees', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max'])
#
# rf2_nk = cons.RF_NK(dict_of_data['alldata_nk'][0], dict_of_data['alldata_nk'][1], rf2_df, 'alldata_nk', scaling=False, modelparams={'n_estimators': 100})
#
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\04_NK-Models\03_Electricity\FinalModel"
# with open(os.path.join(savepath, "{}_{}_RF_notscaled_Class_FinalModel.pickle".format(nk_ky, 'alldata_nk')),'wb') as f:
#     pickle.dump(rf2_nk, f)
# cons.nk_model_basic_diagnostic_plot(rf2_nk, savepath, "{}_{}_RF_notscaled_basic_diagnostic_plot".format(nk_ky, 'alldata_nk'))
#
# rf2_df.to_excel(os.path.join(savepath, "{}_RF_notscaled_FinalModel.xlsx".format(nk_ky)), sheet_name='RF_notscaled', na_rep='-')
#
# cons.get_time("Time for modelling", tic) # last run: 00:32:26
#
# with open(os.path.join(savepath, "{}_{}_RF_notscaled_Class_FinalModel.pickle".format(nk_ky, 'alldata_nk')), 'rb') as f:
#     rf2_nk = pickle.load(f)
#
# # 3rd STEP: writing "Electricity" to DB
# t_write = time.time()
# print('writing to pg-database...')
#
# cons.write_nk_results_to_pg([dict_of_data['alldata_nk'][1]], [rf2_nk.y_targ_destd_], nk_ky, amounts_as_targets=True, scaling=False)
#
# cons.get_time("Time for writing", t_write) # last run: 00:00:02
# cons.get_time("Total Time", tic) # last run: 00:32:28

#****************************
# "Heating": Model-Selection
#****************************
# tic = time.time()
# print("preparing and extracting data...")
# nk_ky = 'EnBr'
#
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\04_NK-Models\04_Heating"
#
# # 1st STEP: extract data and preparing input data
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# alldata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', nk_shr=True, meta='haushaltid',
#                              excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                     'sheet_with_codes': 'Overview',
#                                     'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                     'filter_column_names': ('NK-Model all LR KNN 1', 'NK-Model all')
#                                     })
#
# dict_of_data = cons.nk_data_preparer('alldata_nk', alldata_nk, nk_ky)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# filtereddata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', nk_shr=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter LR KNN 1', 'NK-Model-Filter')
#                                       })
#
# dict_of_data_filt = cons.nk_data_preparer('filtereddata_nk', filtereddata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_filt)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# basicdata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', nk_shr=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter basics LR KNN 1', 'NK-Model-Filter basics')
#                                       })
#
# dict_of_data_basic = cons.nk_data_preparer('basicdata_nk', basicdata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_basic)
#
# with open(os.path.join(savepath, "{}_LR-KNN-Data.pickle".format(nk_ky)), 'wb') as f:
#     pickle.dump(dict_of_data, f)
#
# cons.get_time("Time for preparing data", tic)  # last run: 00:00:42
#
# # 2nd STEP: Training of data / tuning hyperparameters
# t_mod = time.time()
# print("training regression-models...")
#
# lasso_df = pd.DataFrame(data=np.zeros((9, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'alpha', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("Lasso-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     lasso_nk = cons.Lasso_NK(dict_of_data[ky][0], dict_of_data[ky][1], lasso_df, ky)
#     with open(os.path.join(savepath, "{}_{}_LassoClass_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(lasso_nk, f)
#     cons.nk_model_basic_diagnostic_plot(lasso_nk, savepath, "{}_{}_Lasso_basic_diagnostic_plot".format(nk_ky, ky))
#
# lasso_df.to_excel(os.path.join(savepath, "{}_LassoResults_Modelselection.xlsx".format(nk_ky)), sheet_name='LassoResults', na_rep='-')
#
# lasso2_df = pd.DataFrame(data=np.zeros((9, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'alpha', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("PCA-Lasso-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     lasso2_nk = cons.Lasso_NK(dict_of_data[ky][0], dict_of_data[ky][1], lasso2_df, ky, pre_pca=True)
#     with open(os.path.join(savepath, "{}_{}_PCALassoClass_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(lasso2_nk, f)
#     cons.nk_model_basic_diagnostic_plot(lasso2_nk, savepath, "{}_{}_PCALasso_basic_diagnostic_plot".format(nk_ky, ky))
#
# lasso2_df.to_excel(os.path.join(savepath, "{}_PCALassoResults_Modelselection.xlsx".format(nk_ky)), sheet_name='PCALassoResults', na_rep='-')
#
# knn_df = pd.DataFrame(data=np.zeros((16, len(dict_of_data)), dtype=float),
#                         columns=[ky for ky in dict_of_data.keys()],
#                         index=['R2', 'MSE', 'k', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max',
#                                'k=1', 'k=3', 'k=5', 'k=10', 'k=20', 'k=50', 'k=100'])
#
# for i, ky in enumerate(dict_of_data.keys()):
#     print("KNN-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     knn_nk = cons.KNN_NK(dict_of_data[ky][0], dict_of_data[ky][1], knn_df, ky, dist='euclidean')
#     with open(os.path.join(savepath, "{}_{}_KNNeuclClass_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#         pickle.dump(knn_nk, f)
#     cons.nk_model_basic_diagnostic_plot(knn_nk, savepath, "{}_{}_KNNeucl_basic_diagnostic_plot".format(nk_ky, ky))
#
# knn_df.to_excel(os.path.join(savepath, "{}_KNN-euclidean-Results_Modelselection.xlsx".format(nk_ky)), sheet_name='KNN-euclidean-Results', na_rep='-')
#
# # knn2_df = pd.DataFrame(data=np.zeros((9, len(dict_of_data)), dtype=float),
# #                         columns=[ky for ky in dict_of_data.keys()],
# #                         index=['R2', 'MSE', 'k', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max',
# #                               'k=1', 'k=3', 'k=5', 'k=10', 'k=20', 'k=50', 'k=100'])])
# #
# # for i, ky in enumerate(dict_of_data.keys()):
# #     print("KNN-Mahalanobis-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
# #     knn_nk = cons.KNN_NK(dict_of_data[ky][0], dict_of_data[ky][1], knn2_df, ky, dist='mahalanobis')
# # savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\NK-Models\K_KNN-mahalanobis-Results_Modelselection.xlsx"
# # knn2_df.to_excel(savepath, sheet_name='KNN-mahalanobis-Results', na_rep='-')
#
# cons.get_time("time for training models", t_mod)  # last run: 00:33:04
#
# cons.get_time("Total time", tic)  # last run: 00:33:46

#****************************
# "Heating": Model-Selection --> exporting data for RF-Modelling
#****************************
# tic = time.time()
# print("extracting and storing data...")
# nk_ky = 'EnBr'
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\04_NK-Models\04_Heating\Data_for_RF_on_Linux"
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# alldata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, nk_shr=True, meta='haushaltid',
#                              excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                     'sheet_with_codes': 'Overview',
#                                     'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                     'filter_column_names': ('NK-Model all RF 1', 'NK-Model all')
#                                     })
#
# dict_of_data = cons.nk_data_preparer('alldata_nk', alldata_nk, nk_ky)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# filtereddata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, nk_shr=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter RF 1', 'NK-Model-Filter')
#                                       })
#
#
# dict_of_data_filt = cons.nk_data_preparer('filtereddata_nk', filtereddata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_filt)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# basicdata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, nk_shr=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter basics RF 1', 'NK-Model-Filter basics')
#                                       })
#
# dict_of_data_basic = cons.nk_data_preparer('basicdata_nk', basicdata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_basic)
#
# with open(os.path.join(savepath, "dict_of_data_heating.pickle"), 'wb') as f:
#     pickle.dump(dict_of_data, f)
#
# cons.get_time("Time for extracting/storing data", tic) # last run: 00:00:15

#****************************
# Modelling "Heating"
#****************************
#
# tic = time.time()
# print("modelling...")
# nk_ky = 'EnBr'
#
# # 1st STEP: extract data and preparing input data
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# filtereddata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, nk_shr=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter RF 1', 'NK-Model-Filter')
#                                       })
#
#
# dict_of_data = cons.nk_data_preparer('filtereddata_nk', filtereddata_nk, nk_ky)
#
# # 2nd STEP: modelling "heating"
# rf2_df = pd.DataFrame(data=np.zeros((10, 1), dtype=float),
#                         columns=['filtereddata_nk'],
#                         index=['R2', 'MSE', 'r2_oob', 'noTrees', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max'])
#
# rf2_nk = cons.RF_NK(dict_of_data['filtereddata_nk'][0], dict_of_data['filtereddata_nk'][1], rf2_df, 'filtereddata_nk', scaling=False, modelparams={'n_estimators': 100})
#
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\04_NK-Models\04_Heating\FinalModel"
#
# with open(os.path.join(savepath, "{}_{}_RF_notscaled_Class_FinalModel.pickle".format(nk_ky, 'filtereddata_nk')),'wb') as f:
#     pickle.dump(rf2_nk, f)
# cons.nk_model_basic_diagnostic_plot(rf2_nk, savepath, "{}_{}_RF_notscaled_basic_diagnostic_plot".format(nk_ky, 'filtereddata_nk'))
#
# rf2_df.to_excel(os.path.join(savepath, "{}_RF_notscaled_FinalModel.xlsx".format(nk_ky)), sheet_name='RF_notscaled', na_rep='-')
#
# cons.get_time("Time for modelling", tic) # last run: 00:03:04
#
# with open(os.path.join(savepath, "{}_{}_RF_notscaled_Class_FinalModel.pickle".format(nk_ky, 'filtereddata_nk')), 'rb') as f:
#     rf2_nk = pickle.load(f)
#
# # 3rd STEP: writing "heating" to DB
# t_write = time.time()
# print('writing to pg-database...')
#
# cons.write_nk_results_to_pg([dict_of_data['filtereddata_nk'][1]], [rf2_nk.y_targ_destd_], nk_ky, amounts_as_targets=False, scaling=False)
#
# cons.get_time("Time for writing", t_write) # last run: 00:00:29
# cons.get_time("Total Time", tic) # last run: 00:03:33


#****************************
# "Water": Model-Selection
#****************************
# if __name__ == "__main__":
#
#     tic = time.time()
#     print("preparing and extracting data...")
#     nk_ky = 'W'
#
#     savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\04_NK-Models\05_Water"
#
#     # 1st STEP: extract data and preparing input data
#     conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
#     alldata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', nk_shr=True, meta='haushaltid',
#                                  excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                         'sheet_with_codes': 'Overview',
#                                         'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                         'filter_column_names': ('NK-Model all LR KNN 1', 'NK-Model all')
#                                         })
#
#     dict_of_data = cons.nk_data_preparer('alldata_nk', alldata_nk, nk_ky)
#
#     conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
#     filtereddata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', nk_shr=True, meta='haushaltid',
#                                       excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                           'sheet_with_codes': 'Overview',
#                                           'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                           'filter_column_names': ('NK-Model-Filter LR KNN 1', 'NK-Model-Filter')
#                                           })
#
#     dict_of_data_filt = cons.nk_data_preparer('filtereddata_nk', filtereddata_nk, nk_ky)
#
#     dict_of_data.update(dict_of_data_filt)
#
#     conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
#     basicdata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', nk_shr=True, meta='haushaltid',
#                                       excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                           'sheet_with_codes': 'Overview',
#                                           'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                           'filter_column_names': ('NK-Model-Filter basics LR KNN 1', 'NK-Model-Filter basics')
#                                           })
#
#     dict_of_data_basic = cons.nk_data_preparer('basicdata_nk', basicdata_nk, nk_ky)
#
#     dict_of_data.update(dict_of_data_basic)
#
#     with open(os.path.join(savepath, "{}_LR-KNN-Data.pickle".format(nk_ky)), 'wb') as f:
#         pickle.dump(dict_of_data, f)
#
#     cons.get_time("Time for preparing data", tic)  # last run: 00:00:22
#
#     # 2nd STEP: Training of data / tuning hyperparameters
#     t_mod = time.time()
#     print("training regression-models...")
#
#     lasso_df = pd.DataFrame(data=np.zeros((9, len(dict_of_data)), dtype=float),
#                             columns=[ky for ky in dict_of_data.keys()],
#                             index=['R2', 'MSE', 'alpha', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max'])
#
#     for i, ky in enumerate(dict_of_data.keys()):
#         print("Lasso-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#         lasso_nk = cons.Lasso_NK(dict_of_data[ky][0], dict_of_data[ky][1], lasso_df, ky)
#         with open(os.path.join(savepath, "{}_{}_LassoClass_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#             pickle.dump(lasso_nk, f)
#         cons.nk_model_basic_diagnostic_plot(lasso_nk, savepath, "{}_{}_Lasso_basic_diagnostic_plot".format(nk_ky, ky))
#
#     lasso_df.to_excel(os.path.join(savepath, "{}_LassoResults_Modelselection.xlsx".format(nk_ky)), sheet_name='LassoResults', na_rep='-')
#
#     lasso2_df = pd.DataFrame(data=np.zeros((9, len(dict_of_data)), dtype=float),
#                             columns=[ky for ky in dict_of_data.keys()],
#                             index=['R2', 'MSE', 'alpha', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max'])
#
#     for i, ky in enumerate(dict_of_data.keys()):
#         print("PCA-Lasso-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#         lasso2_nk = cons.Lasso_NK(dict_of_data[ky][0], dict_of_data[ky][1], lasso2_df, ky, pre_pca=True)
#         with open(os.path.join(savepath, "{}_{}_PCALassoClass_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#             pickle.dump(lasso2_nk, f)
#         cons.nk_model_basic_diagnostic_plot(lasso2_nk, savepath, "{}_{}_PCALasso_basic_diagnostic_plot".format(nk_ky, ky))
#
#     lasso2_df.to_excel(os.path.join(savepath, "{}_PCALassoResults_Modelselection.xlsx".format(nk_ky)), sheet_name='PCALassoResults', na_rep='-')
#
#     knn_df = pd.DataFrame(data=np.zeros((16, len(dict_of_data)), dtype=float),
#                             columns=[ky for ky in dict_of_data.keys()],
#                             index=['R2', 'MSE', 'k', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max',
#                                    'k=1', 'k=3', 'k=5', 'k=10', 'k=20', 'k=50', 'k=100'])
#
#     for i, ky in enumerate(dict_of_data.keys()):
#         print("KNN-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#         knn_nk = cons.KNN_NK(dict_of_data[ky][0], dict_of_data[ky][1], knn_df, ky, dist='euclidean')
#         with open(os.path.join(savepath, "{}_{}_KNNeuclClass_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#             pickle.dump(knn_nk, f)
#         cons.nk_model_basic_diagnostic_plot(knn_nk, savepath, "{}_{}_KNNeucl_basic_diagnostic_plot".format(nk_ky, ky))
#
#     knn_df.to_excel(os.path.join(savepath, "{}_KNN-euclidean-Results_Modelselection.xlsx".format(nk_ky)), sheet_name='KNN-euclidean-Results', na_rep='-')
#
#     # knn2_df = pd.DataFrame(data=np.zeros((9, len(dict_of_data)), dtype=float),
#     #                         columns=[ky for ky in dict_of_data.keys()],
#     #                         index=['R2', 'MSE', 'k', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max',
#     #                               'k=1', 'k=3', 'k=5', 'k=10', 'k=20', 'k=50', 'k=100'])])
#     #
#     # for i, ky in enumerate(dict_of_data.keys()):
#     #     print("KNN-Mahalanobis-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     #     knn_nk = cons.KNN_NK(dict_of_data[ky][0], dict_of_data[ky][1], knn2_df, ky, dist='mahalanobis')
#     # savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\NK-Models\K_KNN-mahalanobis-Results_Modelselection.xlsx"
#     # knn2_df.to_excel(savepath, sheet_name='KNN-mahalanobis-Results', na_rep='-')
#
#     cons.get_time("time for training models", t_mod)  # last run: ca. 00:21:21
#
#     cons.get_time("Total time", tic)  # last run: ca. 00:21:42

#****************************
# "Water": Model-Selection --> exporting data for RF-Modelling
#****************************
# tic = time.time()
# print("extracting and storing data...")
# nk_ky = 'W'
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\04_NK-Models\05_Water\Data_for_RF_on_Linux"
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# alldata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, nk_shr=True, meta='haushaltid',
#                              excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                     'sheet_with_codes': 'Overview',
#                                     'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                     'filter_column_names': ('NK-Model all RF 1', 'NK-Model all')
#                                     })
#
# dict_of_data = cons.nk_data_preparer('alldata_nk', alldata_nk, nk_ky)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# filtereddata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, nk_shr=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter RF 1', 'NK-Model-Filter')
#                                       })
#
# dict_of_data_filt = cons.nk_data_preparer('filtereddata_nk', filtereddata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_filt)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# basicdata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, nk_shr=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter basics RF 1', 'NK-Model-Filter basics')
#                                       })
#
# dict_of_data_basic = cons.nk_data_preparer('basicdata_nk', basicdata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_basic)
#
# with open(os.path.join(savepath, "dict_of_data_water.pickle"), 'wb') as f:
#     pickle.dump(dict_of_data, f)
#
# cons.get_time("Time for extracting/storing data", tic) # last run: 00:00:15

#****************************
# Modelling "Water"
#****************************

# tic = time.time()
# print("modelling...")
# nk_ky = 'W'
#
# # 1st STEP: extract data and preparing input data
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# basicdata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, nk_shr=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter basics RF 1', 'NK-Model-Filter basics')
#                                       })
#
# dict_of_data = cons.nk_data_preparer('basicdata_nk', basicdata_nk, nk_ky)
#
# # 2nd STEP: modelling "water"
# rf2_df = pd.DataFrame(data=np.zeros((10, 1), dtype=float),
#                         columns=['basicdata_nk'],
#                         index=['R2', 'MSE', 'r2_oob', 'noTrees', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max'])
#
# rf2_nk = cons.RF_NK(dict_of_data['basicdata_nk'][0], dict_of_data['basicdata_nk'][1], rf2_df, 'basicdata_nk', scaling=False, modelparams={'n_estimators': 100})
#
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\04_NK-Models\05_Water\FinalModel"
#
# with open(os.path.join(savepath, "{}_{}_RF_notscaled_Class_FinalModel.pickle".format(nk_ky, 'basicdata_nk')),'wb') as f:
#     pickle.dump(rf2_nk, f)
# cons.nk_model_basic_diagnostic_plot(rf2_nk, savepath, "{}_{}_RF_notscaled_basic_diagnostic_plot".format(nk_ky, 'basicdata_nk'))
#
# rf2_df.to_excel(os.path.join(savepath, "{}_RF_notscaled_FinalModel.xlsx".format(nk_ky)), sheet_name='RF_notscaled', na_rep='-')
#
# cons.get_time("Time for modelling", tic) # last run: 00:00:33
#
# with open(os.path.join(savepath, "{}_{}_RF_notscaled_Class_FinalModel.pickle".format(nk_ky, 'basicdata_nk')), 'rb') as f:
#     rf2_nk = pickle.load(f)
#
# # 3rd STEP: writing "water" to DB
# t_write = time.time()
# print('writing to pg-database...')
#
# cons.write_nk_results_to_pg([dict_of_data['basicdata_nk'][1]], [rf2_nk.y_targ_destd_], nk_ky, amounts_as_targets=False, scaling=False)
#
# cons.get_time("Time for writing", t_write) # last run: 00:00:55
# cons.get_time("Total Time", tic) # last run: 00:01:28

#****************************
# "Wastewater": Model-Selection
#****************************
# if __name__ == "__main__":
#     tic = time.time()
#     print("preparing and extracting data...")
#     nk_ky = 'AW'
#     savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\04_NK-Models\06_Wastewater"
#
#     # 1st STEP: extract data and preparing input data
#     conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
#     alldata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', nk_shr=True, meta='haushaltid',
#                                  excel={
#                                      'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                      'sheet_with_codes': 'Overview',
#                                      'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                      'filter_column_names': ('NK-Model all LR KNN 1', 'NK-Model all')
#                                      })
#
#     dict_of_data = cons.nk_data_preparer('alldata_nk', alldata_nk, nk_ky)
#
#     conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
#     filtereddata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', nk_shr=True, meta='haushaltid',
#                                       excel={
#                                           'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                           'sheet_with_codes': 'Overview',
#                                           'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                           'filter_column_names': ('NK-Model-Filter LR KNN 1', 'NK-Model-Filter')
#                                           })
#
#     dict_of_data_filt = cons.nk_data_preparer('filtereddata_nk', filtereddata_nk, nk_ky)
#
#     dict_of_data.update(dict_of_data_filt)
#
#     conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
#     basicdata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', nk_shr=True, meta='haushaltid',
#                                    excel={
#                                        'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                        'sheet_with_codes': 'Overview',
#                                        'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                        'filter_column_names': (
#                                        'NK-Model-Filter basics LR KNN 1', 'NK-Model-Filter basics')
#                                        })
#
#     dict_of_data_basic = cons.nk_data_preparer('basicdata_nk', basicdata_nk, nk_ky)
#
#     dict_of_data.update(dict_of_data_basic)
#
#     with open(os.path.join(savepath, "{}_LR-KNN-Data.pickle".format(nk_ky)), 'wb') as f:
#         pickle.dump(dict_of_data, f)
#
#     cons.get_time("Time for preparing data", tic)  # last run: 00:00:17
#
#     # 2nd STEP: Training of data / tuning hyperparameters
#     t_mod = time.time()
#     print("training regression-models...")
#
#     lasso_df = pd.DataFrame(data=np.zeros((9, len(dict_of_data)), dtype=float),
#                             columns=[ky for ky in dict_of_data.keys()],
#                             index=['R2', 'MSE', 'alpha', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max'])
#
#     for i, ky in enumerate(dict_of_data.keys()):
#         print("Lasso-Regression: {}; {} of {}".format(ky, i + 1, len(dict_of_data)))
#         lasso_nk = cons.Lasso_NK(dict_of_data[ky][0], dict_of_data[ky][1], lasso_df, ky)
#         with open(os.path.join(savepath, "{}_{}_LassoClass_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#             pickle.dump(lasso_nk, f)
#         cons.nk_model_basic_diagnostic_plot(lasso_nk, savepath, "{}_{}_Lasso_basic_diagnostic_plot".format(nk_ky, ky))
#
#     lasso_df.to_excel(os.path.join(savepath, "{}_LassoResults_Modelselection.xlsx".format(nk_ky)),
#                       sheet_name='LassoResults', na_rep='-')
#
#     lasso2_df = pd.DataFrame(data=np.zeros((9, len(dict_of_data)), dtype=float),
#                             columns=[ky for ky in dict_of_data.keys()],
#                             index=['R2', 'MSE', 'alpha', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max'])
#
#     for i, ky in enumerate(dict_of_data.keys()):
#         print("PCA-Lasso-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#         lasso2_nk = cons.Lasso_NK(dict_of_data[ky][0], dict_of_data[ky][1], lasso2_df, ky, pre_pca=True)
#         with open(os.path.join(savepath, "{}_{}_PCALassoClass_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#             pickle.dump(lasso2_nk, f)
#         cons.nk_model_basic_diagnostic_plot(lasso2_nk, savepath, "{}_{}_PCALasso_basic_diagnostic_plot".format(nk_ky, ky))
#
#     lasso2_df.to_excel(os.path.join(savepath, "{}_PCALassoResults_Modelselection.xlsx".format(nk_ky)), sheet_name='PCALassoResults', na_rep='-')
#
#
#     knn_df = pd.DataFrame(data=np.zeros((16, len(dict_of_data)), dtype=float),
#                             columns=[ky for ky in dict_of_data.keys()],
#                             index=['R2', 'MSE', 'k', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max',
#                                    'k=1', 'k=3', 'k=5', 'k=10', 'k=20', 'k=50', 'k=100'])
#
#     for i, ky in enumerate(dict_of_data.keys()):
#         print("KNN-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#         knn_nk = cons.KNN_NK(dict_of_data[ky][0], dict_of_data[ky][1], knn_df, ky, dist='euclidean')
#         with open(os.path.join(savepath, "{}_{}_KNNeuclClass_Modelselection.pickle".format(nk_ky, ky)), 'wb') as f:
#             pickle.dump(knn_nk, f)
#         cons.nk_model_basic_diagnostic_plot(knn_nk, savepath, "{}_{}_KNNeucl_basic_diagnostic_plot".format(nk_ky, ky))
#
#     knn_df.to_excel(os.path.join(savepath, "{}_KNN-euclidean-Results_Modelselection.xlsx".format(nk_ky)), sheet_name='KNN-euclidean-Results', na_rep='-')
#
#     # knn2_df = pd.DataFrame(data=np.zeros((9, len(dict_of_data)), dtype=float),
#     #                         columns=[ky for ky in dict_of_data.keys()],
#     #                         index=['R2', 'MSE', 'k', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max',
#     #                               'k=1', 'k=3', 'k=5', 'k=10', 'k=20', 'k=50', 'k=100'])])
#     #
#     # for i, ky in enumerate(dict_of_data.keys()):
#     #     print("KNN-Mahalanobis-Regression: {}; {} of {}".format(ky, i+1, len(dict_of_data)))
#     #     knn_nk = cons.KNN_NK(dict_of_data[ky][0], dict_of_data[ky][1], knn2_df, ky, dist='mahalanobis')
#     # savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\Consumption\NK-Models\K_KNN-mahalanobis-Results_Modelselection.xlsx"
#     # knn2_df.to_excel(savepath, sheet_name='KNN-mahalanobis-Results', na_rep='-')
#
#     cons.get_time("time for training models", t_mod)  # last run: ca. 00:20:46
#
#     cons.get_time("Total time", tic)  # last run: ca. 00:21:03

#****************************
# "Wastewater": Model-Selection --> exporting data for RF-Modelling
#****************************
# tic = time.time()
# print("extracting and storing data...")
# nk_ky = 'AW'
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\04_NK-Models\06_Wastewater\Data_for_RF_on_Linux"
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# alldata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, nk_shr=True, meta='haushaltid',
#                              excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                     'sheet_with_codes': 'Overview',
#                                     'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                     'filter_column_names': ('NK-Model all RF 1', 'NK-Model all')
#                                     })
#
# dict_of_data = cons.nk_data_preparer('alldata_nk', alldata_nk, nk_ky)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# filtereddata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, nk_shr=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter RF 1', 'NK-Model-Filter')
#                                       })
#
# dict_of_data_filt = cons.nk_data_preparer('filtereddata_nk', filtereddata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_filt)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# basicdata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, nk_shr=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter basics RF 1', 'NK-Model-Filter basics')
#                                       })
#
# dict_of_data_basic = cons.nk_data_preparer('basicdata_nk', basicdata_nk, nk_ky)
#
# dict_of_data.update(dict_of_data_basic)
#
# with open(os.path.join(savepath, "dict_of_data_ww.pickle"), 'wb') as f:
#     pickle.dump(dict_of_data, f)
#
# cons.get_time("Time for extracting/storing data", tic) # last run: 00:00:15

#****************************
# Modelling "Wastewater"
#****************************

# tic = time.time()
# print("modelling...")
# nk_ky = 'AW'
#
# # 1st STEP: extract data and preparing input data
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# basicdata_nk = cons.SKLData_NK(nk_ky, conn, 'working_tables.habe_hh_prepared', add_month=True, nk_shr=True, meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('NK-Model-Filter basics RF 1', 'NK-Model-Filter basics')
#                                       })
#
# dict_of_data = cons.nk_data_preparer('basicdata_nk', basicdata_nk, nk_ky)
#
# # 2nd STEP: modelling "wastewater"
#
# rf2_df = pd.DataFrame(data=np.zeros((10, 1), dtype=float),
#                         columns=['basicdata_nk'],
#                         index=['R2', 'MSE', 'r2_oob', 'noTrees', 'pred_mean', 'pred_min', 'pred_max', 'orig_mean', 'orig_min', 'orig_max'])
#
# rf2_nk = cons.RF_NK(dict_of_data['basicdata_nk'][0], dict_of_data['basicdata_nk'][1], rf2_df, 'basicdata_nk', scaling=False, modelparams={'n_estimators': 100})
#
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\04_NK-Models\06_Wastewater\FinalModel"
#
# with open(os.path.join(savepath, "{}_{}_RF_notscaled_Class_FinalModel.pickle".format(nk_ky, 'basicdata_nk')),'wb') as f:
#     pickle.dump(rf2_nk, f)
# cons.nk_model_basic_diagnostic_plot(rf2_nk, savepath, "{}_{}_RF_notscaled_basic_diagnostic_plot".format(nk_ky, 'basicdata_nk'))
#
# rf2_df.to_excel(os.path.join(savepath, "{}_RF_notscaled_FinalModel.xlsx".format(nk_ky)), sheet_name='RF_notscaled', na_rep='-')
#
# cons.get_time("Time for modelling", tic) # last run: 00:00:29
#
# with open(os.path.join(savepath, "{}_{}_RF_notscaled_Class_FinalModel.pickle".format(nk_ky, 'basicdata_nk')), 'rb') as f:
#     rf2_nk = pickle.load(f)
#
# # 3rd STEP: writing "water" to DB
# t_write = time.time()
# print('writing to pg-database...')
#
# cons.write_nk_results_to_pg([dict_of_data['basicdata_nk'][1]], [rf2_nk.y_targ_destd_], nk_ky, amounts_as_targets=False, scaling=False)
#
# cons.get_time("Time for writing", t_write) # last run: 00:00:50
# cons.get_time("Total Time", tic) # last run: ca. 00:01:19

#****************************
# Clean up PG-Database
#****************************
# t = time.time()
# excel = {'path': r'D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx',
#          'overview': 'Overview',
#          'water': 'Water_prices',
#          'ww': 'Wastewater_prices',
#          'waste': 'Waste_prices',
#          'electricity': 'Electricity_prices',
#          'energy': 'Shares of heat energy carriers'}
# cons.clean_up_nk_pg(excel)
#
# cons.get_time("Time for cleaning up PG_Database", t)  # last run: 00:01:44

#----------------------------------------------------------------------------------------------------------------------
# Filtering features for Self-Organizing Map
#----------------------------------------------------------------------------------------------------------------------

#****************************
# Show seasonality graphs and compute ANOVA/Kruskal-Wallis to decide if correction for seasonality is necessary
#****************************
# tic = time.time()
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# filtered_data_for_som = cons.SKLData(conn, 'working_tables.habe_hh_prepared_imputed',  meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('Pattern Recognition 1', 'Pattern Recognition 2')
#                                       })
# del filtered_data_for_som._conn_
#
# attrs = [attr for attr in filtered_data_for_som.attributes_ if not attr.startswith('char_') and not attr.startswith('cg_')]
#
# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\01_Filtering\filtered_for_som_notcorr.pdf", attrs, not_orig_table='working_tables.habe_hh_prepared_imputed', varnames_dict=filtered_data_for_som.varnames_)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# filtered_data_for_som_full = cons.SKLData(conn, 'working_tables.habe_hh_prepared_imputed',  meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('Pattern Recognition (full) 1', 'Pattern Recognition (full) 2')
#                                       })
# del filtered_data_for_som_full._conn_
#
# attrs = [attr for attr in filtered_data_for_som_full.attributes_ if not attr.startswith('char_') and not attr.startswith('cg_')]
#
# cons.create_seasonal_figs(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\01_Filtering\filtered_for_som_full.pdf", attrs, not_orig_table='working_tables.habe_hh_prepared_imputed', varnames_dict=filtered_data_for_som_full.varnames_)
#
# cons.get_time("Time for creating seasonal graphs", tic)  # last run: 00:18:22

#****************************
# Prepare data for feature importance computations
#****************************
# tic=time.time()
# print("get data...")
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# # Extract the data:
# filtered_data_for_som = cons.SKLData_SOM(conn, 'working_tables.habe_hh_prepared_imputed',  meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('Pattern Recognition 1', 'Pattern Recognition 2')
#                                       })
#
# filtered_data_for_som_full = cons.SKLData_SOM(conn, 'working_tables.habe_hh_prepared_imputed',  meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('Pattern Recognition (full) 1', 'Pattern Recognition (full) 2')
#                                       })
#
# # The following extraction of data is actually an overkill as it is only used to determine which attributes need to be
# # deseasonalized
# tobedeseasonalized = cons.SKLData_SOM(conn, 'working_tables.habe_hh_prepared_imputed',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('Seasonality-Correction 1', 'Seasonality-Correction 2')
#                                       })
# cons.get_time('time for data extraction', tic)  # last run: 00:00:10
#
# # Before estimating the feature importance, we deseasonalize seasonal data:
# t_des = time.time()
# print("preparation/deseasonalizing...")
#
# tobedeseasonalized = [attr for attr in tobedeseasonalized.attributes_]
# tobedeseasonalized_notcorr = list(set(tobedeseasonalized).intersection(set(filtered_data_for_som.attributes_)))
# tobedeseasonalized_full = list(set(tobedeseasonalized).intersection(set(filtered_data_for_som_full.attributes_)))
#
# filtered_data_for_som.deseasonalize(tobedeseasonalized_notcorr, method='quantiles', plotpath=r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\01_Data\deseasonalizing_notcorr.pdf")
# filtered_data_for_som.standardize_data()
#
# filtered_data_for_som_full.deseasonalize(tobedeseasonalized_full, method='quantiles', plotpath=r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\01_Data\deseasonalizing_full.pdf")
# filtered_data_for_som_full.standardize_data()
#
# try:
#     del filtered_data_for_som._conn_
# except:
#     pass
# try:
#     del filtered_data_for_som_full._conn_
# except:
#     pass
#
# # Pickle the data for later use and/or export for Linux
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\01_Data\filtered_data_for_som_notcorr.pickle", 'wb') as f:
#     pickle.dump(filtered_data_for_som, f)
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\01_Data\filtered_data_for_som_full.pickle", 'wb') as f:
#     pickle.dump(filtered_data_for_som_full, f)
#
# cons.get_time('time for deseasonalizing', t_des)  # last run: 00:16:35
# cons.get_time('Total time for preparing', tic)  # last run: 00:16:46

#****************************
# Compute feature importance scores (here only LASSO/Randomized LASSO --> for RF see Linux)
#****************************
# tic=time.time()
#
# print("load data...")
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\01_Data\filtered_data_for_som_notcorr.pickle", 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\01_Data\filtered_data_for_som_full.pickle", 'rb') as f:
#     filtered_data_for_som_full = pickle.load(f)
#
# print("LASSO: not correlated dataset...")
# t_notcorr = time.time()
# lasso_fi_notcorr = cons.Lasso_FI(filtered_data_for_som, title='not_corr', rlasso=True,
#                     predictors=filtered_data_for_som.attributes_,
#                     targets=[t for t in filtered_data_for_som.attributes_ if t.startswith('a') or t.startswith('cg') or t.startswith('m')])
# cons.get_time("Time for not correlated dataset", t_notcorr)  # 00:28:19
#
# try:
#     del lasso_fi_notcorr.data_._conn_
# except:
#     pass
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\01_Data\lasso_fi_notcorr.pickle", 'wb') as f:
#     pickle.dump(lasso_fi_notcorr, f)
#
# lasso_fi_notcorr.save2excel(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\01_Data\lasso_fi_notcorr.xlsx")
#
# print("LASSO: full dataset...")
# t_full = time.time()
# lasso_fi_full = cons.Lasso_FI(filtered_data_for_som_full, title='full', rlasso=True,
#                     predictors=filtered_data_for_som_full.attributes_,
#                     targets=[t for t in filtered_data_for_som_full.attributes_ if t.startswith('a') or t.startswith('cg') or t.startswith('m')])
# cons.get_time("Time for not correlated dataset", t_full)  # 05:12:29
#
# try:
#     del lasso_fi_full.data_._conn_
# except:
#     pass
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\01_Data\lasso_fi_full.pickle", 'wb') as f:
#     pickle.dump(lasso_fi_full, f)
#
# lasso_fi_full.save2excel(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\01_Data\lasso_fi_full.xlsx")
#
# cons.get_time('time for FI-computing', tic)  # last run: 05:41:01

#****************************
# Analysis of feature importances
#****************************
#
# tic = time.time()
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
# print("analyzing feature importances...")
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\02_Results\lasso_fi_notcorr.pickle", 'rb') as f:
#     lasso_fi_notcorr = pickle.load(f)
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\02_Results\rf_fi_notcorr_light.pickle", 'rb') as f:
#     rf_fi_notcorr = pickle.load(f)
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\02_Results\lasso_fi_full.pickle", 'rb') as f:
#     lasso_fi_full = pickle.load(f)
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\02_Results\rf_fi_full_light.pickle", 'rb') as f:
#     rf_fi_full = pickle.load(f)
#
# t_notcorr = time.time()
# print("not correlated dataset")
# fi_analyzer_notcorr = cons.FI_Analyzer(conn, [lasso_fi_notcorr, rf_fi_notcorr], title='not_corr')
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\01_Data\fi_analyzer_notcorr.pickle", 'wb') as f:
#     pickle.dump(fi_analyzer_notcorr, f)
#
# fi_analyzer_notcorr.save2excel(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\02_Results\fi_analyzer_notcorr.xlsx")
# fi_analyzer_notcorr.create_plots(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\02_Results\fi_analyzer_notcorr.pdf")
#
# cons.get_time("Time for not correlated dataset", t_notcorr)  # last run: 00:00:29
#
# t_full = time.time()
# print("full dataset")
# fi_analyzer_full = cons.FI_Analyzer(conn, [lasso_fi_full, rf_fi_full], title='full')
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\01_Data\fi_analyzer_full.pickle", 'wb') as f:
#     pickle.dump(fi_analyzer_full, f)
#
# fi_analyzer_full.save2excel(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\02_Results\fi_analyzer_full.xlsx")
# fi_analyzer_full.create_plots(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\02_Results\fi_analyzer_full.pdf")
#
# cons.get_time("Time for full dataset", t_full)  # last run: 00:02:08
#
# cons.get_time('total time', tic)  # last run: not corr: 00:00:31 / full: 00:02:12

#----------------------------------------------------------------------------------------------------------------------
# Tuning Self-Organizing Maps --> exporting data for Linux
#----------------------------------------------------------------------------------------------------------------------

#****************************
# Computations for not-coupled/correlated dataset
#****************************

# # IF FEATURE IMPORTANCE IS NOT USED FOR FILTERING, THEN THE DATA PICKLED ABOVE (subchapter:
# # "Prepare data for feature importance computations") CAN BE USED.
#
# # Additional computations for setting up the SOM:
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance\01_Data\filtered_data_for_som_notcorr.pickle", 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# # Compute the number of neurons --> Vesanto 2000: m = 5 * sqrt(no. of samples)
#
# noneurons = 5 * (len(filtered_data_for_som.data_)**0.5)
# print("no. of neurons: {}".format(noneurons))
#
# # Compute the ratio of the side lengths --> according to Vesanto 2000: this corresponds to the ratio between the two greatest eigenvalues of covariance matrix of data
#
# data_cov = np.cov(filtered_data_for_som.data_scaled_, rowvar=False)
# eigv, _ = np.linalg.eig(data_cov)
# eigv.sort()
#
# ratio = float(eigv[-1]/eigv[-2])
# print("ratio of side lengths: {}".format(ratio))
#
# # Compute the two side length:
# l1 = (noneurons/ratio)**0.5
# print("length 1: {}".format(l1))
#
# l2 = ratio * l1
# print("length 2: {}".format(l2))
#
# print("definite lengths: {} : {}".format(int(round(l1,0)),int(round(l2, 0))))
# print("no of neurons: {}".format(int(round(l1,0))*int(round(l2,0))))
#
# # PRINT:
# # no. of neurons: 493.3051793768235
# # ratio of side lengths: 2.2506451553657376
# # length 1: 14.804860638896777
# # length 2: 33.32048787279793
# # definite lengths: 15 : 33
# # no of neurons: 495

# #****************************
# # Computations for full dataset
# #****************************
#
# # IF FEATURE IMPORTANCE IS NOT USED FOR FILTERING, THEN THE DATA PICKLED ABOVE (subchapter:
# # "Prepare data for feature importance computations") CAN BE USED.
#
# # Additional computations for setting up the SOM:
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance (finally not applied)\01_Data\filtered_data_for_som_full.pickle", 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# # Compute the number of neurons --> Vesanto 2000: m = 5 * sqrt(no. of samples)
#
# noneurons = 5 * (len(filtered_data_for_som.data_)**0.5)
# print("no. of neurons: {}".format(noneurons))
#
# # Compute the ratio of the side lengths --> according to Vesanto 2000: this corresponds to the ratio between the two greatest eigenvalues of covariance matrix of data
#
# data_cov = np.cov(filtered_data_for_som.data_scaled_, rowvar=False)
# eigv, _ = np.linalg.eig(data_cov)
# eigv.sort()
#
# ratio = float(eigv[-1]/eigv[-2])
# print("ratio of side lengths: {}".format(ratio))
#
# # Compute the two side length:
# l1 = (noneurons/ratio)**0.5
# print("length 1: {}".format(l1))
#
# l2 = ratio * l1
# print("length 2: {}".format(l2))
#
# print("definite lengths: {} : {}".format(int(round(l1,0)),int(round(l2, 0))))
# print("no of neurons: {}".format(int(round(l1,0))*int(round(l2,0))))
#
# # PRINT:
# # no. of neurons: 493.3051793768235
# # ratio of side lengths: 2.4972571667482706
# # length 1: 14.054849630798339
# # length 2: 35.09857396808044
# # definite lengths: 14 : 35
# # no of neurons: 490

# PLEASE NOTE: the SOM was written to PGDB after the clustering was accomplished --> see below how this was done


#----------------------------------------------------------------------------------------------------------------------
# Clustering (agglomerative clustering and K-Means --> DBSCAN on Linux-Server)
#----------------------------------------------------------------------------------------------------------------------
# if __name__ == '__main__':
#     tic = time.time()
#     print('loading data...')
#
#     # Load the SOM
#     with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\08_Best SOM\best_som.pickle", 'rb') as f:
#         som = pickle.load(f)
#
#     # We load arbitrarily the first "component map" to compute the connectivity (how the "pixels" are connected to
#     # each other).
#     som_map = som.codebook.matrix[:,0].reshape(som.codebook.mapsize[0],som.codebook.mapsize[1])
#     som_connectivity = grid_to_graph(*som_map.shape)
#
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
# #****************************
# # Hierarchical Clustering
# #****************************
#
#     print("Start Hierarchical Clustering...")
#     t_hc = time.time()
#
#     # Investigating the cluster number up to 500 is an overkill in view of the fact that there are only about 1000 neurons;
#     # Furthermore, in fact it would not be necessary to pass this cluster list to the agglomerative clustering since the
#     # whole dendrogram is computed anyway (but this way, it is just easier to directly retrieve the corresponding
#     # metrics)
#     noofclusterlist = [nc for nc in range(2,505)]
#
#     transl = {'Sil': 'silhouette_avg', 'CH': 'calinski_harabasz_score', 'Dendr': 'largest_dendro_gap'}
#
#     for i, ky in enumerate(hierarchical_params.keys()):
#         print("Variant {}: {} of {}".format(ky, i + 1, len(hierarchical_params)))
#         hierarchical_clusterer = AgglomerativeClustering(n_clusters=2, affinity=hierarchical_params[ky]['affinity'],
#                                                          connectivity=hierarchical_params[ky]['connectivity'],
#                                                          compute_full_tree=True,
#                                                          linkage=hierarchical_params[ky]['linkage'])
#         hierarchical_clustering = cons.Clustering_Analyzer(hierarchical_clusterer, som, {'n_clusters': noofclusterlist},
#                                                            scaling=hierarchical_params[ky]['scaling'],
#                                                            saving={'title': 'AggloClust {}'.format(ky),
#                                                             'savepath': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\01_Tuning\Hierarchical Clustering"})
#         df = hierarchical_clustering.make_score_dataframe()
#         df.to_excel(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\01_Tuning\Hierarchical Clustering\AggloClust {}.xlsx".format(ky))
#         for x in ('Sil', 'CH'):
#             df_x = df.copy()
#             df_x = df_x[df_x[transl[x]] == df_x[transl[x]].max()]
#             metalist = [df_x.loc[df_x.index[0],'silhouette_avg'], df_x.loc[df_x.index[0],'n_clusters'], df_x.loc[df_x.index[0],'calinski_harabasz_score'], df_x.loc[df_x.index[0],'largest_dendro_gap'], np.nan, np.nan, np.nan]
#             clustering_overview_df["{}_{}".format(ky, x)] = metalist
#         try:
#             df = df[df['n_clusters'] == df[transl['Dendr']]]
#             metalist = [df.loc[df.index[0], 'silhouette_avg'], df.loc[df.index[0], 'n_clusters'],
#                         df.loc[df.index[0], 'calinski_harabasz_score'], df.loc[df.index[0], 'largest_dendro_gap'], np.nan, np.nan, np.nan]
#         except:
#             metalist = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
#         clustering_overview_df["{}_{}".format(ky, 'Dendr')] = metalist
#     del hierarchical_clustering
#     del hierarchical_clusterer
#
#     cons.get_time("Time for hierarchical clustering", t_hc)
#
# #****************************
# # K-Means Clustering
# #****************************
#
#     print("Start K-Means Clustering...")
#     t_kmeans = time.time()
#
#     # We decided to decrease the list of clusters to be investigated compared to agglomerative clustering
#     noofclusterlist = [nc for nc in range(2, 205)]
#
#     for i, ky in enumerate(kmeans_params.keys()):
#         print("Variant {}: {} of {}".format(ky, i + 1, len(kmeans_params)))
#         kmeans_clusterer = KMeans(n_clusters=2, init='k-means++', n_init=100, copy_x=True, n_jobs=-1, algorithm='full')
#         kmeans_clustering = cons.Clustering_Analyzer(kmeans_clusterer, som, {'n_clusters': noofclusterlist},
#                                                      scaling=kmeans_params[ky]['scaling'],
#                                                     saving={'title': 'KMeans {}'.format(ky),
#                                                     'savepath': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\01_Tuning\K-Means"})
#         df = kmeans_clustering.make_score_dataframe()
#         df.to_excel(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\01_Tuning\K-Means\KMeans {}.xlsx".format(ky))
#         for x in ('Sil', 'CH'):
#             df_x = df.copy()
#             df_x = df_x[df_x[transl[x]] == df_x[transl[x]].max()]
#             metalist = [df_x.loc[df_x.index[0], 'silhouette_avg'], df_x.loc[df_x.index[0], 'n_clusters'],
#                         df_x.loc[df_x.index[0], 'calinski_harabasz_score'], np.nan, np.nan, np.nan, np.nan]
#             clustering_overview_df["{}_{}".format(ky, x)] = metalist
#     del kmeans_clustering
#     del kmeans_clusterer
#
#     cons.get_time("Time for K-Means clustering", t_kmeans)
#
# #****************************
# # DBSCAN Clustering  --> Maybe apply DBSCAN also on server with a more exhaustive eps-list
# #****************************
#
#     # print("Start DBSCAN Clustering...")
#     # t_dbscan = time.time()
#     #
#     # # The almost same reasons as for K-Means were taken to come up with the minsampleslist
#     # minsampleslist = np.arange(2,205,1)
#     #
#     # for i, ky in enumerate(dbscan_params.keys()):
#     #     print("Variant {}: {} of {}".format(ky, i + 1, len(kmeans_params)))
#     #
#     #     # In the following, we construct the eps-list based on the distances between the clustering data
#     #     # Please note: we start at the smallest distance, increase by the smallest distance and go up to half of the
#     #     # space in scaled data, but only to a fourth of the space with unscaled data.
#     #     if dbscan_params[ky]['scaling']:
#     #         clustering_data = som.codebook.matrix
#     #         scaler = 2
#     #     else:
#     #         clustering_data = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)
#     #         scaler = 4
#     #     dists = pairwise_distances(clustering_data, metric=dbscan_params[ky]['metric'], n_jobs=-1)
#     #     upper = np.triu(dists, k=1)
#     #     upper_flat = upper.reshape(1, upper.shape[0] * upper.shape[1])
#     #     upper_flat = upper_flat[upper_flat > 0]
#     #
#     #     # However, this procedure still came up with way too large eps-list for cosine, therefore, we inspected the data
#     #     # and decided for a 0.01 increase.
#     #     if not dbscan_params[ky]['metric'] == 'cosine':
#     #         epslist = np.arange(min(upper_flat), max(upper_flat) / scaler, min(upper_flat))
#     #     else:
#     #         epslist = np.arange(min(upper_flat), max(upper_flat/ scaler), 0.01)
#     #         continue  # Unfortunately, cosine did not work the way we wanted therefore, we skip this analysis
#     #
#     #     dbscan_clusterer = DBSCAN(eps=0.5, min_samples=5, metric=dbscan_params[ky]['metric'], n_jobs=-2)
#     #     dbscan_clustering = cons.Clustering_Analyzer(dbscan_clusterer, som, {'min_samples': minsampleslist, 'eps': epslist},
#     #                                                  scaling=dbscan_params[ky]['scaling'],
#     #                                                        saving={'title': 'DBSCAN {}'.format(ky),
#     #                                                                'savepath': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\01_Tuning\DBSCAN"})
#     #     df = dbscan_clustering.make_score_dataframe()
#     #     df.to_excel(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\01_Tuning\DBSCAN\DBSCAN {}.xlsx".format(ky))
#     #     for x in ('Sil', 'CH'):
#     #         df_x = df.copy()
#     #         df_x = df_x[df_x[transl[x]] == df_x[transl[x]].max()]
#     #         metalist = [df_x.loc[df_x.index[0], 'silhouette_avg'], df_x.loc[df_x.index[0], 'n_clusters'],
#     #                     df_x.loc[df_x.index[0], 'calinski_harabasz_score'], np.nan, df_x.loc[df_x.index[0], 'min_samples'], df_x.loc[df_x.index[0], 'eps'],
#     #                     df_x.loc[df_x.index[0], 'outlier_share']]
#     #         clustering_overview_df["{}_{}".format(ky, x)] = metalist
#     #
#     # del dbscan_clustering
#     # del dbscan_clusterer
#     #
#     # cons.get_time("Time for DBSCAN clustering", t_dbscan)
#
#     savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\01_Tuning\Clustering_Overview.xlsx"
#     clustering_overview_df.to_excel(savepath, sheet_name='Clustering_Overview', na_rep='-')
#
#     cons.get_time("Total Time", tic) # last run without DBSCAN: 17:00:19; Hierarchical Clustering: 12:38:39 / K-Means: 04:21:40


#****************************
# Evaluation of the clustering by means of the U-Matrix
#****************************

# #Agglomerative Clustering
# tic = time.time()
#
# #Load the SOM
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\08_Best SOM\best_som.pickle", 'rb') as f:
#     som = pickle.load(f)
#
# #Load Clusterer:
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\01_Tuning\Hierarchical Clustering\AggloClust ward_eucl_scaled_conn.pickle", 'rb') as f:
#     aggloclust = pickle.load(f)
#
# pp = PdfPages(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\02_Clustering-Evaluation\UMatrix_ward_eucl_scaled.pdf")
#
# nclusters = np.arange(5, 161, 1)
#
# for i, nc in enumerate(nclusters):
#     print("Cluster {} of {}".format(i+1, len(nclusters)))
#
#     aggloclusterer = getattr(aggloclust, 'hierarchical_clustering_{}_clusters_'.format(nc))
#
#     u = cons.UMatrix_impr(50, 50, 'umatrix', show_axis=False, text_size=8, show_text=True)
#     _ = u.show_impr(som, distance2=1, row_normalized=False, show_data=False,
#              contooor=True, labels=False, cmap='coolwarm_r', savepath=pp,
#                 clusters={'n_clusters': nc, 'clusterer': aggloclusterer})
#
# pp.close()
#
# cons.get_time("time for plotting U-Matrices", tic)
#
#
# # K-Means Clustering
# tic = time.time()
#
# #Load the SOM
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\08_Best SOM\best_som.pickle", 'rb') as f:
#     som = pickle.load(f)
#
# #Load Clusterer:
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\01_Tuning\K-Means\KMeans kmeans_scaled.pickle", 'rb') as f:
#     kmeansclust = pickle.load(f)
#
# pp = PdfPages(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\02_Clustering-Evaluation\UMatrix_kmeans_scaled.pdf")
#
# nclusters = np.arange(5, 94, 1)
#
# for i, nc in enumerate(nclusters):
#     print("Cluster {} of {}".format(i+1, len(nclusters)))
#
#     kmeansclusterer = getattr(kmeansclust, 'kmeans_{}_clusters_'.format(nc))
#
#     u = cons.UMatrix_impr(50, 50, 'umatrix', show_axis=False, text_size=8, show_text=True)
#     _ = u.show_impr(som, distance2=1, row_normalized=False, show_data=False,
#              contooor=True, labels=False, cmap='coolwarm_r', savepath=pp,
#                 clusters={'n_clusters': nc, 'clusterer': kmeansclusterer})
#
# pp.close()
#
# cons.get_time("time for plotting U-Matrices", tic)

# #DBSCAN Clustering
# tic = time.time()
#
# #Load the SOM
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\08_Best SOM\best_som.pickle", 'rb') as f:
#     som = pickle.load(f)
#
# #Euclidean:
# pp = PdfPages(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\02_Clustering-Evaluation\UMatrix_dbscan_eucl_scaled.pdf")
# for fl in os.listdir(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\01_Tuning\DBSCAN"):
#     if fl.endswith('pickle') and 'eucl' in fl:
#         with open(os.path.join(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\01_Tuning\DBSCAN", fl) , 'rb') as f:
#             dbscanclusterer = pickle.load(f)
#
#         u = cons.UMatrix_impr(50, 50, 'umatrix', show_axis=False, text_size=8, show_text=True)
#         _ = u.show_impr(som, distance2=1, row_normalized=False, show_data=False,
#                         contooor=True, labels=False, cmap='coolwarm_r', savepath=pp,
#                         clusters={'n_clusters': len(set(dbscanclusterer.labels_)), 'clusterer': dbscanclusterer})
#
#
#
# pp.close()
#
# cons.get_time("time for plotting U-Matrices", tic)  # last run: 00:00:26
#
# #L1:
# pp = PdfPages(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\02_Clustering-Evaluation\UMatrix_dbscan_l1_scaled.pdf")
# for fl in os.listdir(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\01_Tuning\DBSCAN"):
#     if fl.endswith('pickle') and 'l1' in fl:
#         with open(os.path.join(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\01_Tuning\DBSCAN", fl) , 'rb') as f:
#             dbscanclusterer = pickle.load(f)
#
#         u = cons.UMatrix_impr(50, 50, 'umatrix', show_axis=False, text_size=8, show_text=True)
#         _ = u.show_impr(som, distance2=1, row_normalized=False, show_data=False,
#                         contooor=True, labels=False, cmap='coolwarm_r', savepath=pp,
#                         clusters={'n_clusters': len(set(dbscanclusterer.labels_)), 'clusterer': dbscanclusterer})
#
# pp.close()
#
# cons.get_time("time for plotting U-Matrices", tic)  # last run: 00:00:26

#****************************
# In-depth Evaluation of the clustering
#****************************

# #Agglomerative Clustering
# tic = time.time()
#
# #Load the SOM
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\08_Best SOM\best_som.pickle", 'rb') as f:
#     som = pickle.load(f)
#
# #Load Clusterer:
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\01_Tuning\Hierarchical Clustering\AggloClust ward_eucl_scaled_conn.pickle", 'rb') as f:
#     aggloclust = pickle.load(f)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance (finally not applied)\01_Data\filtered_data_for_som_notcorr.pickle", 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# filtered_data_for_som_raw = cons.SKLData_SOM(conn, 'working_tables.habe_hh_prepared_imputed',  meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('Pattern Recognition 1', 'Pattern Recognition 2')
#                                       })
#
# assert np.array_equal(filtered_data_for_som.meta_, filtered_data_for_som_raw.meta_)
# assert np.array_equal(filtered_data_for_som.attributes_, filtered_data_for_som_raw.attributes_)
#
# idxlist = [i for i, a in enumerate(filtered_data_for_som.attributes_) if a not in filtered_data_for_som.seasonality_corrected_]
# assert np.array_equal(filtered_data_for_som.data_[:,idxlist], filtered_data_for_som_raw.data_[:,idxlist])
#
# #nclusters = np.array([20, 23, 26, 32, 34, 35, 36, 41, 44, 51, 54, 56, 99, 136, 160])  # first run
# nclusters = np.array([26, 34, 41, 54])  # second run
# pp = PdfPages(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\02_Clustering-Evaluation\Ward_eucl_overview.pdf")
#
# for i, nc in enumerate(nclusters):
#     print("Cluster {} of {}".format(i+1, len(nclusters)), end='\r')
#
#     aggloclusterer = getattr(aggloclust, 'hierarchical_clustering_{}_clusters_'.format(nc))
#     sil_avg = getattr(aggloclust, 'hierarchical_clustering_{}_clusters_silhouette_avg_'.format(nc))
#
#     evaluator = cons.SOM_Clustering_Evaluator(som, aggloclusterer, filtered_data_for_som_raw, conn)
#     evaluator.save2excel(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\02_Clustering-Evaluation\Ward_eucl_stats_{}.xlsx".format(nc))
#     evaluator.create_plot(pp, avg_silhouette=sil_avg)
#
#
# pp.close()
#
#
# cons.get_time("Time for agglomerative clustering", tic)  # first run: 00:05:27 / second run: 00:01:11

# #K-Means Clustering
# tic = time.time()
#
# #Load the SOM
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\08_Best SOM\best_som.pickle", 'rb') as f:
#     som = pickle.load(f)
#
# #Load Clusterer:
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\01_Tuning\K-Means\KMeans kmeans_scaled.pickle", 'rb') as f:
#     kmeansclust = pickle.load(f)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance (finally not applied)\01_Data\filtered_data_for_som_notcorr.pickle", 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# filtered_data_for_som_raw = cons.SKLData_SOM(conn, 'working_tables.habe_hh_prepared_imputed',  meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('Pattern Recognition 1', 'Pattern Recognition 2')
#                                       })
#
# assert np.array_equal(filtered_data_for_som.meta_, filtered_data_for_som_raw.meta_)
# assert np.array_equal(filtered_data_for_som.attributes_, filtered_data_for_som_raw.attributes_)
#
# idxlist = [i for i, a in enumerate(filtered_data_for_som.attributes_) if a not in filtered_data_for_som.seasonality_corrected_]
# assert np.array_equal(filtered_data_for_som.data_[:,idxlist], filtered_data_for_som_raw.data_[:,idxlist])
#
# #nclusters = np.array([18, 20, 22, 25, 28, 30, 31, 32, 34, 35, 39, 41, 45, 47, 48, 53, 54, 66, 77, 92])  # first run
# nclusters = np.array([28, 31, 34, 41, 48, 54])  # second run
# pp = PdfPages(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\02_Clustering-Evaluation\KMeans_scaled_overview.pdf")
#
# for i, nc in enumerate(nclusters):
#     print("Cluster {} of {}".format(i+1, len(nclusters)), end='\r')
#
#     kmeansclusterer = getattr(kmeansclust, 'kmeans_{}_clusters_'.format(nc))
#     sil_avg = getattr(kmeansclust, 'kmeans_{}_clusters_silhouette_avg_'.format(nc))
#
#     evaluator = cons.SOM_Clustering_Evaluator(som, kmeansclusterer, filtered_data_for_som_raw, conn)
#     evaluator.save2excel(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\02_Clustering-Evaluation\KMeans_scaled_stats_{}.xlsx".format(nc))
#     evaluator.create_plot(pp, avg_silhouette=sil_avg)
#
#
# pp.close()
#
#
# cons.get_time("Time for KMeans-clustering-Evaluation", tic)  # first run: 00:06:16 / second run: 00:01:53

#****************************
# Deep Analysis of Hierarchical Clustering
#****************************
#
# #Load Clusterer:
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\01_Tuning\Hierarchical Clustering\AggloClust ward_eucl_scaled_conn.pickle", 'rb') as f:
#     aggloclust = pickle.load(f)
#
# aggloclust.hierarchical_deep_analysis(aggloclust.hierarchical_clustering_34_clusters_, dendro_params={'truncate_mode':'lastp', 'p':54, 'leaf_rotation':90.,
#                                    'leaf_font_size':12., 'show_contracted':True, 'annotate_above':10, 'max_d': 22.95},
#                                    saving={'title': 'ward_scaled_conn_34clusters_truncatedat54', 'savepath': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\02_Clustering-Evaluation\Ward-Clustering"})


#----------------------------------------------------------------------------------------------------------------------
# Definitive Clustering
#----------------------------------------------------------------------------------------------------------------------

#****************************
# Compute Statistics
#****************************

# print("Compute statistics and perform definitive clustering")
# tic = time.time()
#
# # Load data and SOM:
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\08_Best SOM\best_som.pickle", 'rb') as f:
#     som = pickle.load(f)
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance (finally not applied)\01_Data\filtered_data_for_som_notcorr.pickle", 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# filtered_data_for_som_raw = cons.SKLData_SOM(conn, 'working_tables.habe_hh_prepared_imputed',  meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2'),
#                                       'filter_column_names': ('Pattern Recognition 1', 'Pattern Recognition 2')
#                                       })
#
# assert np.array_equal(som._data, filtered_data_for_som.data_scaled_)
# assert np.array_equal(som.data_raw, filtered_data_for_som.data_)
# assert np.array_equal(filtered_data_for_som.meta_, filtered_data_for_som_raw.meta_)
# assert np.array_equal(filtered_data_for_som.attributes_, filtered_data_for_som_raw.attributes_)
#
# idxlist = [i for i, a in enumerate(filtered_data_for_som.attributes_) if a not in filtered_data_for_som.seasonality_corrected_]
# assert np.array_equal(filtered_data_for_som.data_[:,idxlist], filtered_data_for_som_raw.data_[:,idxlist])
#
# # For later use, we also pickle the raw (not deseasonalized SKLData-class):
# del filtered_data_for_som_raw._conn_
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance (finally not applied)\01_Data\filtered_data_for_som_notcorr_raw_notdeseasonlized.pickle", 'wb') as f:
#     pickle.dump(filtered_data_for_som_raw,f)
#
# # Load the definitive clusterer:
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\01_Tuning\Hierarchical Clustering\AggloClust ward_eucl_scaled_conn.pickle", 'rb') as f:
#     ward_scaled_conn = pickle.load(f)
#
# # make a copy of the original labels:
# best_clusterer = copy.deepcopy(ward_scaled_conn.hierarchical_clustering_34_clusters_)
# best_clusterer.labels_old_ = best_clusterer.labels_.copy()
#
# # The "merge"-dict shall translate current labels into those with which they shall be merged:
# merge_dict = {
#     14: 13,
#     24: 3,
#     27: 1,
#     28: 6,
#     30: 12,
#     32: 33
# }
#
# # Merge the clusters and overwrite the old labels:
# new_cluster_labels = np.array([merge_dict[l] if l in merge_dict.keys() else l for l in best_clusterer.labels_])
# best_clusterer.labels_ = new_cluster_labels
# best_clusterer.merge_dict_ = merge_dict
#
# # Furthermore, we would like to give names in form of letters to the clusters (the two "outlier" clusters will start with 'O')
# lbls = set(new_cluster_labels)
# lbls.remove(4)
# lbls.remove(29)
# new_dict = {lbl: string.ascii_uppercase[i] for i, lbl in enumerate(lbls)}
# new_dict[4] = 'OA'
# new_dict[29] = 'OB'
#
# new_label_names = [new_dict[l] for l in new_cluster_labels]
# best_clusterer.label_names_ = new_label_names
# best_clusterer.names_dict_ = new_dict
#
# # In the following, we will compute the clustering statistics for our final choice:
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\03_Final Clustering"
# silhouette_avg, sample_silhouette_values = ward_scaled_conn._do_silhouette_analysis('final_clustering', best_clusterer.labels_, dbscan=False)
# sil_fig = ward_scaled_conn._silhouette_plot(best_clusterer, silhouette_avg, sample_silhouette_values, 'Final Clustering (Ward, Scaled, Connectivity)')
# sil_fig.savefig(os.path.join(savepath, 'FinalClustering_Silhouette_neuron_plot.pdf'), format='pdf', orientation='landscape', papertype='a4')
# plt.close(sil_fig)
# del sil_fig
#
# best_clusterer.silhouette_avg_neurons_ = silhouette_avg
# best_clusterer.sample_silhouette_values_neurons_ = sample_silhouette_values
#
# pp = PdfPages(os.path.join(savepath, "FinalClustering_Stats.pdf"))
# evaluator = cons.SOM_Clustering_Evaluator(som, best_clusterer, filtered_data_for_som_raw, conn)
# evaluator.save2excel(os.path.join(savepath, "FinalClustering_Stats.xlsx"))
# evaluator.create_plot(pp, avg_silhouette=silhouette_avg)
# pp.close()
#
# # We now would also like to compute the Silhouette-Score for the households (not only neurons). For this, we need to
# # hack our own class:
# ward_scaled_conn.clustering_data_ = filtered_data_for_som.data_scaled_
# best_clusterer.labels_temp_ = best_clusterer.labels_.copy()
#
# # list of household clusters (to which clusters the households belong to):
# hh_clusters = [evaluator._cluster_list_[int(evaluator._bmus_[hhidx])] for hhidx, _ in enumerate(evaluator.hh_data_.meta_)]
# best_clusterer.labels_ = hh_clusters
#
# silhouette_avg, sample_silhouette_values = ward_scaled_conn._do_silhouette_analysis('final_clustering_hhs', best_clusterer.labels_, dbscan=False)
# sil_fig = ward_scaled_conn._silhouette_plot(best_clusterer, silhouette_avg, sample_silhouette_values, 'Final Clustering (Ward, Scaled, Connectivity)')
# sil_fig.savefig(os.path.join(savepath, 'FinalClustering_Silhouette_allHHs_plot.pdf'), format='pdf', orientation='landscape', papertype='a4')
# plt.close(sil_fig)
# del sil_fig
#
# best_clusterer.hh_cluster_labels_ = best_clusterer.labels_.copy()
# best_clusterer.labels_ = best_clusterer.labels_temp_.copy()
# del best_clusterer.labels_temp_
#
# best_clusterer.silhouette_avg_hhs_ = silhouette_avg
# best_clusterer.sample_silhouette_values_hhs_ = sample_silhouette_values
#
# # for later use, we also pickle the adjusted clusterer-instance:
# with open(os.path.join(savepath, 'best_clusterer_manuallyadjusted.pickle'), 'wb') as f:
#     pickle.dump(best_clusterer, f)
#
# # Finally, we also would like to have a Hit-Map, a Hit-Map with the cluster-labels and the U-Matrix without cluster borders:
# pp = PdfPages(os.path.join(savepath, "Final_U-Matrix.pdf"))
# u = cons.UMatrix_impr(50, 50, 'U-Matrix', show_axis=False, text_size=8, show_text=True)
# _ = u.show_impr(som, distance2=1, row_normalized=False, show_data=False,
#              contooor=False, labels=False, cmap='coolwarm_r', savepath=pp)
#
# _ = u.show_impr(som, distance2=1, row_normalized=False, show_data=False,
#              contooor=False, labels=False, cmap='coolwarm_r', savepath=pp,
#                 clusters={'n_clusters': len(set(best_clusterer.labels_)), 'clusterer': best_clusterer})
#
# _ = u.show_impr(som, distance2=1, row_normalized=False, show_data=False,
#              contooor=True, labels=False, cmap='coolwarm_r', savepath=pp)
#
# pp.close()
#
# pp = PdfPages(os.path.join(savepath, "Final_Hits-Maps.pdf"))
# hits = cons.BmuHitsView_impr(50, 50, 'Hits-Map', text_size=8)
# _ = hits.show_impr(evaluator.som_, labelsize=12, cmap='nipy_spectral', savepath=pp)
#
# _ = hits.show(evaluator.som_, labelsize=12, cmap='Greys', savepath=pp)
#
# pp.close()
#
# cons.get_time("time for preparing definitive clustering", tic)  # last run: 00:00:44
#
# #****************************
# # Compute Archetypes and write to PGDB
# #****************************
#
# t_db = time.time()
# print("start writing to PGDB")
#
# # In a first step we create a translation-table with the following information: HHID - corresponding BMU/Neuron -
# # Original Cluster-Label (before merging certain clusters) - definitive cluster label (after merging) - name of cluster (string)
# hh_cluster_transl_dicts = []
# for hhidx, hhid in enumerate(evaluator.hh_data_.meta_):
#     hhbmu = int(evaluator._bmus_[hhidx])
#     hhclust_old = best_clusterer.labels_old_[hhbmu]
#     hhclust = evaluator._cluster_list_[hhbmu]
#     hh_cluster_transl_dicts.append({'haushaltid': int(hhid), 'bmu_index': hhbmu, 'cluster_label_orig': int(hhclust_old),
#                                     'cluster_label_def': int(hhclust), 'cluster_label_name': new_dict[hhclust]})
#
# writecur = conn.cursor()
# query = """
# DROP TABLE IF EXISTS working_tables.habe_clustering
# """
# writecur.execute(query)
# conn.commit()
#
# query = """
# CREATE TABLE working_tables.habe_clustering
# (haushaltid bigint, bmu_index int, cluster_label_orig int, cluster_label_def int, cluster_label_name varchar,
# CONSTRAINT habeclustering_pkey PRIMARY KEY (haushaltid))
# """
# writecur.execute(query)
# conn.commit()
#
# cols = ['haushaltid', 'bmu_index', 'cluster_label_orig', 'cluster_label_def', 'cluster_label_name']
# query = """
# INSERT INTO working_tables.habe_clustering(%s)
# VALUES (%s(%s)s)
# """ % (', '.join(cols), '%', ')s, %('.join(cols))
# writecur.executemany(query, hh_cluster_transl_dicts)
# conn.commit()
#
# writecur.execute("""
# CREATE INDEX habeclustering_hhid
# ON working_tables.habe_clustering
# USING btree
# (haushaltid);
# """)
# conn.commit()
#
# # In a next step, we will compute the demand of all archetypes. For this, we need to first to get a list of all attributes
# alldata = cons.SKLData(conn, 'working_tables.habe_hh_prepared_imputed',  meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2')
#                                       })
# attrlist = list(alldata.attributes_)
#
# # Now we will construct a list of the column names:
# attrlist2 = ['cluster_label_name varchar', 'cluster_label_def int']
# for a in attrlist:  # go through all attributes of a HH
#     a += ' float'  # please note that we will float also for char- and cg-attributes
#     attrlist2.append(a)
#
# query = """
# DROP TABLE IF EXISTS working_tables.habe_archetypes_weighted
# """
# writecur.execute(query)
# conn.commit()
#
# query = """
# DROP TABLE IF EXISTS working_tables.habe_archetypes_notweighted
# """
# writecur.execute(query)
# conn.commit()
#
# query = """
# CREATE TABLE working_tables.habe_archetypes_weighted
# ({}, CONSTRAINT habearchetypesweighted_pkey PRIMARY KEY (cluster_label_def))
# """.format(', '.join(attrlist2))
# writecur.execute(query)
# conn.commit()
#
# query = """
# CREATE TABLE working_tables.habe_archetypes_notweighted
# ({}, CONSTRAINT habearchetypesnotweighted_pkey PRIMARY KEY (cluster_label_def))
# """.format(', '.join(attrlist2))
# writecur.execute(query)
# conn.commit()
#
# query="""
# SELECT clust.cluster_label_name, clust.cluster_label_def, {} FROM working_tables.habe_hh_prepared_imputed habe LEFT JOIN
# working_tables.habe_clustering clust ON habe.haushaltid=clust.haushaltid LEFT JOIN original_data.habe_standard std ON
# habe.haushaltid=std.haushaltid
# GROUP BY clust.cluster_label_name, clust.cluster_label_def
# """.format(', '.join(['SUM(habe.{}*std.gewicht10_091011)/SUM(std.gewicht10_091011) AS {}'.format(attr, attr) for attr in attrlist]))
#
# cur = conn.cursor(cursor_factory=pge.RealDictCursor)
# cur.execute(query)
# archetypes_weighted = cur.fetchall()
# cur.close()
#
# query="""
# SELECT clust.cluster_label_name, clust.cluster_label_def, {} FROM working_tables.habe_hh_prepared_imputed habe LEFT JOIN
# working_tables.habe_clustering clust ON habe.haushaltid=clust.haushaltid
# GROUP BY clust.cluster_label_name, clust.cluster_label_def
# """.format(', '.join(['AVG(habe.{}) AS {}'.format(attr, attr) for attr in attrlist]))
#
# cur = conn.cursor(cursor_factory=pge.RealDictCursor)
# cur.execute(query)
# archetypes_not_weighted = cur.fetchall()
# cur.close()
#
# attrlist.append('cluster_label_name')
# attrlist.append('cluster_label_def')
#
# query = """
# INSERT INTO working_tables.habe_archetypes_weighted(%s)
# VALUES (%s(%s)s)
# """ % (', '.join(attrlist), '%', ')s, %('.join(attrlist))
# writecur.executemany(query, archetypes_weighted)
# conn.commit()
#
# query = """
# INSERT INTO working_tables.habe_archetypes_notweighted(%s)
# VALUES (%s(%s)s)
# """ % (', '.join(attrlist), '%', ')s, %('.join(attrlist))
# writecur.executemany(query, archetypes_not_weighted)
# conn.commit()
#
# writecur.execute("""
# CREATE INDEX habearchetypes_clusters
# ON working_tables.habe_archetypes_weighted
# USING btree
# (cluster_label_def);
# """)
# conn.commit()
#
# writecur.execute("""
# CREATE INDEX habearchetypes_clusters_nm
# ON working_tables.habe_archetypes_weighted
# USING btree
# (cluster_label_name);
# """)
# conn.commit()
#
# writecur.execute("""
# CREATE INDEX habearchetypes_clusters_notweighted
# ON working_tables.habe_archetypes_notweighted
# USING btree
# (cluster_label_def);
# """)
# conn.commit()
#
# writecur.execute("""
# CREATE INDEX habearchetypes_clusters_nm_notweighted
# ON working_tables.habe_archetypes_notweighted
# USING btree
# (cluster_label_name);
# """)
# conn.commit()
#
# writecur.close()
# conn.close()
#
# cons.get_time("Time for writing to PGDB", t_db)  # last run: 00:00:12
# cons.get_time("Total Time", tic)  # last run: 00:00:56

#****************************
# Finally and more for being complete and for documentation purposes, we also write the SOM to the PGDB
#****************************
# tic = time.time()
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\08_Best SOM\best_som.pickle", 'rb') as f:
#     som = pickle.load(f)
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\02_Feature Importance (finally not applied)\01_Data\filtered_data_for_som_notcorr.pickle", 'rb') as f:
#     filtered_data_for_som = pickle.load(f)
#
# assert np.array_equal(som.data_raw, filtered_data_for_som.data_)
# assert np.array_equal(som._data, filtered_data_for_som.data_scaled_)
#
# conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#
# writecur = conn.cursor()
# query = """
# DROP TABLE IF EXISTS working_tables.som_21to47
# """
# writecur.execute(query)
# conn.commit()
#
# writecur = conn.cursor()
# query = """
# DROP TABLE IF EXISTS working_tables.filtered_deseasonalized_som_data
# """
# writecur.execute(query)
# conn.commit()
#
# writecur = conn.cursor()
# query = """
# DROP TABLE IF EXISTS working_tables.filtered_deseasonalized_scaled_som_data
# """
# writecur.execute(query)
# conn.commit()
#
# attrlist = list(filtered_data_for_som.attributes_)
#
# attrlist2 = ['bmu_index int']
# for a in attrlist:
#     a += ' float'
#     attrlist2.append(a)
#
# query = """
# CREATE TABLE working_tables.som_21to47
# ({}, CONSTRAINT som_pkey PRIMARY KEY (bmu_index))
# """.format(', '.join(attrlist2))
# writecur.execute(query)
# conn.commit()
#
# attrlist3 = ['haushaltid bigint']
# for a in attrlist:
#     a += ' float'
#     attrlist3.append(a)
#
# query = """
# CREATE TABLE working_tables.filtered_deseasonalized_som_data
# ({}, CONSTRAINT somdata_pkey PRIMARY KEY (haushaltid))
# """.format(', '.join(attrlist3))
# writecur.execute(query)
# conn.commit()
#
# query = """
# CREATE TABLE working_tables.filtered_deseasonalized_scaled_som_data
# ({}, CONSTRAINT somdata_scaled_pkey PRIMARY KEY (haushaltid))
# """.format(', '.join(attrlist3))
# writecur.execute(query)
# conn.commit()
#
# attrlist4 = list(attrlist)
# attrlist.append('bmu_index')
# attrlist4.append('haushaltid')
#
# bmu_som_dict_list = []
# for bmu in range(som.codebook.matrix.shape[0]):
#     bmu_dict = {'bmu_index': bmu}
#     for ia, a in enumerate(filtered_data_for_som.attributes_):
#         bmu_dict[a] = som.codebook.matrix[bmu, ia]
#     bmu_som_dict_list.append(bmu_dict)
#
# query = """
# INSERT INTO working_tables.som_21to47(%s)
# VALUES (%s(%s)s)
# """ % (', '.join(attrlist), '%', ')s, %('.join(attrlist))
# writecur.executemany(query, bmu_som_dict_list)
# conn.commit()
#
# hh_som_dict_list = []
# for hhidx in range(som.data_raw.shape[0]):
#     hh_dict = {'haushaltid': filtered_data_for_som.meta_[hhidx]}
#     for ia, a in enumerate(filtered_data_for_som.attributes_):
#         hh_dict[a] = som.data_raw[hhidx, ia]
#     hh_som_dict_list.append(hh_dict)
#
# query = """
# INSERT INTO working_tables.filtered_deseasonalized_som_data(%s)
# VALUES (%s(%s)s)
# """ % (', '.join(attrlist4), '%', ')s, %('.join(attrlist4))
# writecur.executemany(query, hh_som_dict_list)
# conn.commit()
#
# hh_somscl_dict_list = []
# for hhidx in range(som._data.shape[0]):
#     hh_dict = {'haushaltid': filtered_data_for_som.meta_[hhidx]}
#     for ia, a in enumerate(filtered_data_for_som.attributes_):
#         hh_dict[a] = som._data[hhidx, ia]
#     hh_somscl_dict_list.append(hh_dict)
#
# query = """
# INSERT INTO working_tables.filtered_deseasonalized_scaled_som_data(%s)
# VALUES (%s(%s)s)
# """ % (', '.join(attrlist4), '%', ')s, %('.join(attrlist4))
# writecur.executemany(query, hh_somscl_dict_list)
# conn.commit()
#
# writecur.execute("""
# CREATE INDEX som_bmu
# ON working_tables.som_21to47
# USING btree
# (bmu_index);
# """)
# conn.commit()
#
# writecur.execute("""
# CREATE INDEX som_data_hhid
# ON working_tables.filtered_deseasonalized_som_data
# USING btree
# (haushaltid);
# """)
# conn.commit()
#
# writecur.execute("""
# CREATE INDEX som_data_scaled_hhid
# ON working_tables.filtered_deseasonalized_scaled_som_data
# USING btree
# (haushaltid);
# """)
# conn.commit()
#
# writecur.close()
# conn.close()
#
# cons.get_time("Time for Writing SOM to PGDB", tic)  # last run: 00:00:37

# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\06_Clustering\03_Final Clustering"
# with open(os.path.join(savepath, 'best_clusterer_manuallyadjusted.pickle'), 'rb') as f:
#     clusterer = pickle.load(f)
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\05_SOM\08_Best SOM\best_som.pickle", 'rb') as f:
#     som = pickle.load(f)
#
# conn = cons.get_pg_connection()
#
# writecur = conn.cursor()
# query = """
# DROP TABLE IF EXISTS working_tables.neurons_clustering
# """
# writecur.execute(query)
# conn.commit()
#
# query = """
# CREATE TABLE working_tables.neurons_clustering
# (bmu_index int, cluster_label_orig int, cluster_label_def int, cluster_label_name varchar, CONSTRAINT neuronclust_pkey PRIMARY KEY (bmu_index))
# """
# writecur.execute(query)
# conn.commit()
#
# bmu_som_dict_list = []
# for bmu in range(som.codebook.matrix.shape[0]):
#     bmu_dict = {'bmu_index': bmu, 'cluster_label_orig': int(clusterer.labels_old_[bmu]), 'cluster_label_def': int(clusterer.labels_[bmu]),'cluster_label_name': clusterer.label_names_[bmu]}
#     bmu_som_dict_list.append(bmu_dict)
#
# attrlist = ['bmu_index', 'cluster_label_orig', 'cluster_label_def', 'cluster_label_name']
#
# query = """
# INSERT INTO working_tables.neurons_clustering(%s)
# VALUES (%s(%s)s)
# """ % (', '.join(attrlist), '%', ')s, %('.join(attrlist))
# writecur.executemany(query, bmu_som_dict_list)
# conn.commit()
#
# writecur.execute("""
# CREATE INDEX neuronsclust_bmu
# ON working_tables.neurons_clustering
# USING btree
# (bmu_index);
# """)
# conn.commit()
#
# writecur.execute("""
# CREATE INDEX neuronsclust_lbl
# ON working_tables.neurons_clustering
# USING btree
# (cluster_label_def);
# """)
# conn.commit()
#
# writecur.execute("""
# CREATE INDEX neuronsclust_nms
# ON working_tables.neurons_clustering
# USING btree
# (cluster_label_name);
# """)
# conn.commit()
#
# writecur.close()
# conn.close()


#----------------------------------------------------------------------------------------------------------------------
# LCA of Archetypes
#----------------------------------------------------------------------------------------------------------------------

#****************************
# Computing the public transport and bike demand of the archetypes
#****************************
# tic = time.time()
#
# # In a first step, we would like get a list of attributes which are needed for computing the public transport and bike
# # demand by the microcensus; Getting this list via SKLData is a convenient way, but definitely an overkill
# conn = cons.get_pg_connection()
# alldata = cons.SKLData(conn, 'working_tables.habe_hh_prepared_imputed',  meta='haushaltid',
#                                   excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#                                       'sheet_with_codes': 'Overview',
#                                       'code_column_names': ('pg-code 1', 'pg-code 2')
#                                       })
# attrlist = list(alldata.attributes_)
# del alldata
# conn.close()
#
# querylist = ['char_nopers', 'e_bruttoeink']
# for a in attrlist:
#     if 'nofem' in a  or 'nomale' in a:
#         querylist.append(a)
#
# mcexcel = {
#     'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#     'sheet': 'PT_and_bike'
# }
#
# # In the following, we estimate the bike/pt-demand for each HABE-Household and construct a PG-Table
# cons.write_habe_pt_bike_to_pg(querylist, mcexcel)
#
# # Finally, we derive the estimate for each archetype and construct a new column to store it
# cons.write_archetypes_pt_bike_to_pg()
#
# cons.get_time("Total Time for PT/Bike-estimation", tic)  # last run: 00:00:11

#****************************
# Do the actual Archetype LCA:
#****************************

# tic = time.time()
#
# # In a first step, we have to perform an LCA in order to prepare the technosphere matrix. For a successful reuse of the
# # factorized LCI-Matrix, we need a functional unit which comprises processes from all databases needed.
# # Since the ecoinvent-based activities are in a different brightway-project than EXIOBASE-activities, we have to run
# # the two analyses in parallel
#
# print("Doing Pre-LCA...")
# pseudo_ei_fu = {
#     ('ecoinvent 3.3 cutoff', '9e32482c441075cbecb151627e5490c4'): 1,
#     ('Agribalyse 1.2', '6ff1d7d24c730753db935815dce15020'): 1,
#     ('heia', 'fruitnes'): 1
# }
#
# # In the methods-dict, we define different environmental indicators which shall be assessed. Note that the keys will
# # be used for further naming (e.g. in PG-tables)
# methods = {
#     'ipcc': ('IPCC 2013', 'climate change', 'GWP 100a'),
#     'rec_end_eco': ('ReCiPe Endpoint (H,A)', 'ecosystem quality', 'total'),
#     'rec_end_hh': ('ReCiPe Endpoint (H,A)', 'human health', 'total'),
#     'rec_end_res': ('ReCiPe Endpoint (H,A)', 'resources', 'total'),
#     'rec_end_tot': ('ReCiPe Endpoint (H,A)', 'total', 'total'),
#     'ubp_tot': ('ecological scarcity 2013', 'total', 'total')
# }
#
# met = methods['ipcc']
# bw2.projects.set_current('heia33')
# ei_lca = bw2.LCA(pseudo_ei_fu, method=met)
# ei_lca.lci()
# ei_lca.lcia()
#
# bw2.projects.set_current('exiobase_industry_workaround')
# pseudo_ex_fu = {('EXIOBASE 2.2', 'Manufacture of tobacco products:LV'): 1}
# ex_lca = bw2.LCA(pseudo_ex_fu, method=met)
# ex_lca.lci()
# ex_lca.lcia()
#
# cons.get_time("Time for Pre-LCA", tic) # last run: 00:00:01
#
# t_lca_prep = time.time()
# print("Computing LCA-multiplication factors...")
#
# # With the ConsumptionLCA-class we instantiate an object which contains multiplication factors for all processes and all
# # methods which need to be analyzed. This procedure turned out to be much faster than doing LCA directly with Brightway
# excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#    'sheet': 'LCA-Modelling'}
# conslca = cons.ConsumptionLCA(ei_lca, ex_lca, methods, excel, archetypes=True)
#
# # For Post-Analysis and for documentation purposes, we will store the ConsumptionLCA-class
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\07_Archetypes_LCA\archetype_consumption_lca.pickle", 'wb') as f:
#     pickle.dump(conslca, f)
#
# cons.get_time("Time for deriving multiplication factors", t_lca_prep) # last run: 00:00:23
#
# print("Extract archetype demands and do LCA...")
# t_lca = time.time()
#
# # For the following attributes, no LCA-scores exist. However, if we want to do an LCA of expenditures for secondary
# # homes, then we need also to pass these attributes.
# attributes_for_secondaryhomes = ['a572200', 'a572300', 'a571202', 'a571203', 'a571204', 'a571205', 'a571301','a571302', 'a571303']
#
# # In the next step, we will extract the archetypes-demands (weighted and not weighted)
# conn = cons.get_pg_connection()
# cur = conn.cursor(cursor_factory=pge.RealDictCursor)
# query = """
# SELECT cluster_label_def, cluster_label_name, {} FROM working_tables.habe_archetypes_weighted
# """.format(', '.join(conslca.attributes_list_ + attributes_for_secondaryhomes))
# cur.execute(query)
# archetypes_weighted = cur.fetchall()
#
# query = """
# SELECT cluster_label_def, cluster_label_name, {} FROM working_tables.habe_archetypes_notweighted
# """.format(', '.join(conslca.attributes_list_ + attributes_for_secondaryhomes))
# cur.execute(query)
# archetypes_notweighted = cur.fetchall()
# cur.close()
# conn.close()
#
# # Do the actual LCA and store the results directly in PG-Database
# cons.do_archetype_lca(conslca, archetypes_weighted, 'results.hh_archetypes_weighted')
#
# cons.do_archetype_lca(conslca, archetypes_notweighted, 'results.hh_archetypes_notweighted')
#
# cons.get_time("Time for Archetype-LCA", t_lca)  # 00:00:04
# cons.get_time("Total Time", tic)  # 00:00:28



#----------------------------------------------------------------------------------------------------------------------
# Training of the classifier
#----------------------------------------------------------------------------------------------------------------------

# tic = time.time()
#
# conn = cons.get_pg_connection()
#
# # In a first step, we retrieve all the data which will also be available in STATPOP
#
# attributes = [
#     'working_tables.habe_hh_prepared_imputed.haushaltid',
#     'char_nopers',
#     'char_georegion_ge',
#     'char_georegion_mit',
#     'char_georegion_nw',
#     'char_georegion_ost',
#     'char_georegion_ti',
#     'char_georegion_zen',
#     'char_georegion_zh',
#     'char_noausl',
#     'char_noch',
#     'char_nodiv',
#     'char_nomarried',
#     'char_nounwed',
#     'char_nowid',
#     'char_nofem0004',
#     'char_nofem0514',
#     'char_nofem1524',
#     'char_nofem2534',
#     'char_nofem3544',
#     'char_nofem4554',
#     'char_nofem5564',
#     'char_nofem6574',
#     'char_nofem7599',
#     'char_nomale0004',
#     'char_nomale0514',
#     'char_nomale1524',
#     'char_nomale2534',
#     'char_nomale3544',
#     'char_nomale4554',
#     'char_nomale5564',
#     'char_nomale6574',
#     'char_nomale7599'
# ]
#
# habe_hhs = cons.SKLData(conn, 'working_tables.habe_hh_prepared_imputed', attributes=attributes,
#                         meta='working_tables.habe_hh_prepared_imputed.haushaltid',
#                         joining={'joining_tables': ['working_tables.habe_clustering'],
#                                 'join_id': ['haushaltid'],
#                                 'attributes': [['cluster_label_def',]]}, excel=None)
#
# del habe_hhs._conn_
#
# X = habe_hhs.data_[:,:-1]
# y = habe_hhs.data_[:,-1]
#
# # In the next step, we also retrieve data which is available in external models that shall be merged with the consumption
# # models.
# # Therefore, we extract litres of diesel and petrol from the HABE-data and convert it to driven kilometers (which will be
# # used to match with MATSim-data) --> ATTENTION: we use the monthly data of the households --> no averaging within clusters!
#
# cur = conn.cursor(cursor_factory=pge.RealDictCursor)
# query= """
# SELECT haushaltid, m621501, m621502 FROM working_tables.habe_hh_prepared_imputed
# """
# cur.execute(query)
# mob_sql = cur.fetchall()
# cur.close()
# mob_dict = {x['haushaltid']: {'m621501': x['m621501'], 'm621502': x['m621502']} for x in mob_sql}
#
# # We now load the LCA-consumption class since in the FU-attributes we find the necessary information to convert litres
# # to km (based on ecoinvent 3 guidelines)
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\07_Archetypes_LCA\archetype_consumption_lca.pickle", 'rb') as f:
#     lca=pickle.load(f)
#
# petrolfact = 0
# for ky in lca.m621501_fu_.keys():
#     petrolfact += lca.m621501_fu_[ky]
#
# dieselfact = 0
# for ky in lca.m621502_fu_.keys():
#     dieselfact += lca.m621502_fu_[ky]
#
# # We then already compute the demand in km per HH and year
# # and also compute the total demands
#
# for hh in mob_dict.keys():
#     mob_dict[hh]['m621501'] *= petrolfact*12*1.6
#     mob_dict[hh]['m621502'] *= dieselfact*12*1.6
# mob_df = pd.DataFrame.from_dict(mob_dict, orient='index')
# mob_df['tot'] = mob_df.sum(axis=1)
#
# # The next for-loop ensures that the km are assigned to the correct households
#
# km_driven = []
# for hh in habe_hhs.meta_:
#     km_driven.append(mob_df.loc[hh, 'tot'])
# km_driven = np.array(km_driven)
#
# # # To be more robust, we decided to work with the percentiles instead of the absolute values
# # km_driven = []
# # for hh in habe_hhs.meta_:
# #     km_driven.append([stats.percentileofscore(mob_df['tot'], mob_df.loc[hh, 'tot']) / 100])
# # km_driven = np.array(km_driven)
#
# # Add the new "predictor" to the X-array
# X_mob = X.copy()
# X_mob = np.append(X_mob, km_driven.reshape(-1, 1), axis=1)
#
# # Then proceed similarly for the housing demand:
# cur = conn.cursor(cursor_factory=pge.RealDictCursor)
# query= """
# SELECT haushaltid, mx571302 FROM working_tables.habe_hh_prepared_imputed
# """
# cur.execute(query)
# hus_sql = cur.fetchall()
# cur.close()
#
# # Since the data is already in MJ we only need to multiply by the number of months
# hus_dict = {x['haushaltid']: {'tot': x['mx571302']*12} for x in hus_sql}
# hus_df = pd.DataFrame.from_dict(hus_dict, orient='index')
#
# # # The next for-loop ensures that the MJ are assigned to the correct households and again, we convert to quantiles
# # mj_heated = []
# # for hh in habe_hhs.meta_:
# #     mj_heated.append([stats.percentileofscore(hus_df['tot'], hus_df.loc[hh, 'tot']) / 100])
# # mj_heated = np.array(mj_heated)
#
# mj_heated = []
# for hh in habe_hhs.meta_:
#     mj_heated.append(hus_df.loc[hh, 'tot'])
# mj_heated = np.array(mj_heated)
#
# # Add the new "predictor" to the X-array
# X_mob_hus = X_mob.copy()
# X_mob_hus = np.append(X_mob_hus, mj_heated.reshape(-1, 1), axis=1)
#
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\09_Classification"
#
# # PROBLEM: we get out of memory --> COMPUTATIONS ARE DONE ON LINUX-SERVER --> we transfer the data to the server
# rf_data_dict = {'X_mob': X_mob, 'X_mob_hus': X_mob_hus, 'y': y, 'data': habe_hhs, 'mobility_demand': mob_df, 'housing_demand': hus_df}
# with open(os.path.join(savepath, 'rf_tuning_data.pickle'), 'wb') as f:
#     pickle.dump(rf_data_dict, f)
#
# # THE FOLLOWING COMPUTATIONS WERE DONE ON THE LINUX-SERVER BUT IN A MODIFIED/IMPROVED VERSION!
# # # Define the tuning parameters for the Random Forest-Classifier
# #
# # rf_tuning_params = {'default_gini': {'n_estimators': [100, 200, 300], 'criterion': 'gini', 'max_features': 'sqrt'},
# #                     'half_gini': {'n_estimators': [100, 200, 300], 'criterion': 'gini', 'max_features': round(0.5*(X.shape[1]**0.5))},
# #                     'twice_gini': {'n_estimators': [100, 200, 300], 'criterion': 'gini', 'max_features': round(2*(X.shape[1]**0.5))},
# #                     'max_gini': {'n_estimators': [100, 200, 300], 'criterion': 'gini', 'max_features': None},
# # }
# #
# #
# # # Define the "base estimator" and then tune the classifiers
# # rf_clf = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1, bootstrap=True,
# #                                 oob_score=True, n_jobs=-1, warm_start=False, class_weight='balanced')
# #
# # rf_clf_tuner = cons.RFClassifierTuner(rf_clf, rf_tuning_params, X, y)
# #
# # # Store the results to excel as well as to a pickle-file
# # savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\09_Classification"
# # rf_clf_tuner.save2excel(os.path.join(savepath, 'rf_gini_tuner.xlsx'))
# #
# # rf_clf_dict = {'rf_clf_tuner': rf_clf_tuner, 'data': habe_hhs, 'mobility_demand': mob_df}
# # with open(os.path.join(savepath, 'rf_gini_tuner.pickle'), 'wb') as f:
# #     pickle.dump(rf_clf_dict, f)
#
# conn.close()
# cons.get_time("Time for Training Classifier", tic)  # last run: 00:00:34

# #****************************
# # Tuning the probabilities (only mobility):
# #****************************
#
# # with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\09_Classification\twice_gini_2000.pickle", 'rb') as f:
# #     rf = pickle.load(f)
# #
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\09_Classification\gini_data.pickle", 'rb') as f:
#     data = pickle.load(f)
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\09_Classification\rf_tuning_data.pickle",'rb') as f:
#     tuning_data = pickle.load(f)
#
# # We will construct a small translator for the clusters for the pictures (we realized that because of memory issues, it
# # is better to construct the translator before calibrating the classifier
# conn = cons.get_pg_connection()
# cur = conn.cursor(cursor_factory=pge.RealDictCursor)
# query = """
# SELECT DISTINCT cluster_label_def, cluster_label_name FROM working_tables.habe_clustering
# """
# cur.execute(query)
# transl_sql = cur.fetchall()
# cur.close()
# conn.close()
# clusttransl = {x['cluster_label_def']: x['cluster_label_name'] for x in transl_sql}
#
#
# # cccv = CalibratedClassifierCV(rf, cv='prefit')
# # cccv.fit(data['X_test'], data['y_test'])
# #
# # del data, rf, tuning_data
# #
# # with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\09_Classification\calibrated_classifier.pickle", 'wb') as f:
# #     pickle.dump(cccv, f)
#
# # Because we were out of memory, we had to reload cccv later on
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\09_Classification\calibrated_classifier_isotonic.pickle", 'rb') as f:
#     cccv = pickle.load(f)
#
# # After calibrating the probabilities we will create two further evaluation plots:
#
# # PROBABILITY-HEATMAP
# pp = PdfPages(r"D:\froemelt\Desktop\HeatMaps_Classifier_Probas_isotonic.pdf")
#
# fig_width_cm = 21  # A4 page
# fig_height_cm = 29.7
# inches_per_cm = 1 / 2.58  # Convert cm to inches
# fig_width = fig_width_cm * inches_per_cm  # width in inches
# fig_height = fig_height_cm * inches_per_cm  # height in inches
# fig_size = [fig_height, fig_width]  # height and width are in the order needed for landscape
#
# plt.figure(figsize=fig_size)
# plt.title('Test-Dataset')
# probas = cccv.predict_proba(data['X_test'])
# cols = [clusttransl[c] for c in cccv.classes_]
# rows = ["{}_sample_{}".format(clusttransl[c], i) for i, c in enumerate(data['y_test'])]
# proba_df = pd.DataFrame(data=probas, columns=cols, index=rows)
# proba_df.sort_index(axis=1, inplace=True)
# sns.heatmap(proba_df.sort_index(), cmap='Blues', xticklabels=True)
# plt.yticks(rotation=0)
# fig = plt.gcf()
# fig.savefig(pp, format='pdf', papertype='a4')
# plt.close(fig)
# del fig
#
# plt.figure(figsize=fig_size)
# plt.title('Full Dataset')
# probas = cccv.predict_proba(tuning_data['X'])
# cols = [clusttransl[c] for c in cccv.classes_]
# rows = ["{}_sample_{}".format(clusttransl[c], i) for i, c in enumerate(tuning_data['y'])]
# proba_df = pd.DataFrame(data=probas, columns=cols, index=rows)
# proba_df.sort_index(axis=1, inplace=True)
# sns.heatmap(proba_df.sort_index(), cmap='Blues', xticklabels=True)
# plt.yticks(rotation=0)
# fig = plt.gcf()
# fig.savefig(pp, format='pdf', papertype='a4')
# plt.close(fig)
# del fig
#
# pp.close()
#
# # CONFUSION-MATRIX
# pp = PdfPages(r"D:\froemelt\Desktop\Confusion_Matrix_CalibratedClassifier_isotonic.pdf")
# plt.figure(figsize=fig_size)
# plt.title('Confusion Matrix (for Test-Dataset)')
# y_pred = cccv.predict(data['X_test'])
# cnf_matrix = confusion_matrix(data['y_test'], y_pred)
# cols = [clusttransl[c] for c in cccv.classes_]
# cnf_df = pd.DataFrame(data=cnf_matrix, columns=cols, index=cols)
# cnf_df.sort_index(axis=1, inplace=True)
# sns.heatmap(cnf_df.sort_index(), cmap='Reds', xticklabels=True, yticklabels=True, linewidths=1.0, annot=True, square=True, linecolor='grey')
# plt.yticks(rotation=0)
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# fig = plt.gcf()
# fig.savefig(pp, format='pdf', papertype='a4')
# plt.close(fig)
# del fig
#
# pp.close()

#****************************
# Tuning the probabilities (mobility & housing):
#****************************
#
# # with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\09_Classification\twice_gini_1000.pickle", 'rb') as f:
# #     rf = pickle.load(f)
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\09_Classification\gini_data.pickle", 'rb') as f:
#     data = pickle.load(f)
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\09_Classification\rf_tuning_data.pickle",'rb') as f:
#     tuning_data = pickle.load(f)
#
# # We will construct a small translator for the clusters for the pictures (we realized that because of memory issues, it
# # is better to construct the translator before calibrating the classifier
# conn = cons.get_pg_connection()
# cur = conn.cursor(cursor_factory=pge.RealDictCursor)
# query = """
# SELECT DISTINCT cluster_label_def, cluster_label_name FROM working_tables.habe_clustering
# """
# cur.execute(query)
# transl_sql = cur.fetchall()
# cur.close()
# conn.close()
# clusttransl = {x['cluster_label_def']: x['cluster_label_name'] for x in transl_sql}
#
#
# # cccv = CalibratedClassifierCV(rf, cv='prefit')
# # cccv.fit(data['X_test'], data['y_test'])
# #
# # del data, rf, tuning_data
# #
# # with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\09_Classification\calibrated_classifier.pickle", 'wb') as f:
# #     pickle.dump(cccv, f)
#
# # Because we were out of memory, we had to reload cccv later on
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\09_Classification\calibrated_classifier_isotonic.pickle", 'rb') as f:
#     cccv = pickle.load(f)
#
# # After calibrating the probabilities we will create two further evaluation plots:
#
# # PROBABILITY-HEATMAP
# pp = PdfPages(r"D:\froemelt\Desktop\HeatMaps_Classifier_Probas_isotonic.pdf")
#
# fig_width_cm = 21  # A4 page
# fig_height_cm = 29.7
# inches_per_cm = 1 / 2.58  # Convert cm to inches
# fig_width = fig_width_cm * inches_per_cm  # width in inches
# fig_height = fig_height_cm * inches_per_cm  # height in inches
# fig_size = [fig_height, fig_width]  # height and width are in the order needed for landscape
#
# plt.figure(figsize=fig_size)
# plt.title('Test-Dataset')
# probas = cccv.predict_proba(data['X_test'])
# cols = [clusttransl[c] for c in cccv.classes_]
# rows = ["{}_sample_{}".format(clusttransl[c], i) for i, c in enumerate(data['y_test'])]
# proba_df = pd.DataFrame(data=probas, columns=cols, index=rows)
# proba_df.sort_index(axis=1, inplace=True)
# sns.heatmap(proba_df.sort_index(), cmap='Blues', xticklabels=True)
# plt.yticks(rotation=0)
# fig = plt.gcf()
# fig.savefig(pp, format='pdf', papertype='a4')
# plt.close(fig)
# del fig
#
# plt.figure(figsize=fig_size)
# plt.title('Full Dataset')
# probas = cccv.predict_proba(tuning_data['X_mob_hus'])
# cols = [clusttransl[c] for c in cccv.classes_]
# rows = ["{}_sample_{}".format(clusttransl[c], i) for i, c in enumerate(tuning_data['y'])]
# proba_df = pd.DataFrame(data=probas, columns=cols, index=rows)
# proba_df.sort_index(axis=1, inplace=True)
# sns.heatmap(proba_df.sort_index(), cmap='Blues', xticklabels=True)
# plt.yticks(rotation=0)
# fig = plt.gcf()
# fig.savefig(pp, format='pdf', papertype='a4')
# plt.close(fig)
# del fig
#
# pp.close()
#
# # CONFUSION-MATRIX
# pp = PdfPages(r"D:\froemelt\Desktop\Confusion_Matrix_CalibratedClassifier_isotonic.pdf")
# plt.figure(figsize=fig_size)
# plt.title('Confusion Matrix (for Test-Dataset)')
# y_pred = cccv.predict(data['X_test'])
# cnf_matrix = confusion_matrix(data['y_test'], y_pred)
# cols = [clusttransl[c] for c in cccv.classes_]
# cnf_df = pd.DataFrame(data=cnf_matrix, columns=cols, index=cols)
# cnf_df.sort_index(axis=1, inplace=True)
# sns.heatmap(cnf_df.sort_index(), cmap='Reds', xticklabels=True, yticklabels=True, linewidths=1.0, annot=True, square=True, linecolor='grey')
# plt.yticks(rotation=0)
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# fig = plt.gcf()
# fig.savefig(pp, format='pdf', papertype='a4')
# plt.close(fig)
# del fig
#
# pp.close()


# #----------------------------------------------------------------------------------------------------------------------
# # Compute consumption demand  --> The actual computation is done on the LINUX-Server, here we prepare and export the PG-data
# #----------------------------------------------------------------------------------------------------------------------
#
# cpus = mp.cpu_count()   # count CPUs
#
# # Most of the following code was kindly provided by Rene Buffat
# def exportdem_worker(q, attributes, savepath, wid): # if I got it right: q is the queue (list of municipality numbers) while wid stands for the CPU
#
#     finished = False
#     while not finished:
#         try:
#
#             vals = q.get() # take a value from the queue
#
#             if vals == 'killitwithfire_worker':  # if-loop to terminate the processing
#                 logging.info(
#                     "worker {} finished - received good kill".format(wid))
#                 finished = True
#                 break
#             bfsnr = vals
#
#             cons.export_statpop_hhs(bfsnr, attributes, savepath)   # core of the function: calling the "actual" function
#
#         except Empty:
#             logging.info("worker {} finished - empty".format(wid))
#             finished = True
#             break
#         except Exception as e:
#             logging.exception("wid: {} / {}".format(wid, str(e)))
#
# #in the case of multiprocessing, the main-statement is necessary
# if __name__ == '__main__':
#     t = time.time()
#     print("Initialization/loading data...")
#
#     #Extract all municipality numbers from the PostGIS-DB:
#     conn = pg.connect("host=localhost port=5432 dbname=db_andi user=postgres  connect_timeout=2 password='postgres'")
#     cur = conn.cursor()
#     query = """
#     SELECT DISTINCT hh.bfs_nummer FROM working_tables.hh hh
#     """
#     cur.execute(query)
#     bfsnr_all = []
#     for row in cur:
#         bfsnr_all.append(row[0])
#     cur.close()
#
#     # We will directly save the data to an external hard drive which will then transfer the data to the server
#     savepath = r"E:\Doktorat-Zwischenspeicherung\Export4Linux"
#
#     with open(os.path.join(savepath, "bfsnrs.pickle"), 'wb') as f:
#         pickle.dump(bfsnr_all, f)
#
#     # The following try-statement would allow for interrupting the computation and restart afterwards
#     try:
#         bfsnr_deja = os.listdir(savepath)
#         bfsnr_deja = [int(f.split('_')[-1].replace('.pickle', '')) for f in bfsnr_deja if f.split('_')[-1].replace('.pickle', '').isnumeric()]
#         bfsnr_all = list(set(bfsnr_all) - set(bfsnr_deja))
#     except:
#         pass
#
#     # Extract the attributes which were used for the classifier to ensure the correct order
#     with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\09_Classification\rf_tuning_data.pickle", 'rb') as f:
#         rf_tuning_data = pickle.load(f)
#     attributes = rf_tuning_data['data']
#     attributes = attributes.attributes_[:-1]
#
#     # ATTENTION: the following lines are actually not needed for the data-preprocessing but they prepare data which will
#     # be needed for the consumption demand computations on the server:
#
#     # In order to ensure the same data also after interruption, we first test if data is already available (we test
#     # for habe_attributes since this is the only data for which the order is important:
#     try:
#         with open(os.path.join(savepath, "habe_attributes.pickle"), 'rb') as f:
#             habe_attributes = pickle.load(f)
#     except:
#         # Get all the column names of the Archetypes-table
#         query = """
#         select column_name from information_schema.columns where table_schema = 'working_tables' AND
#         table_name='habe_archetypes_weighted'"""
#         cur = conn.cursor()
#         cur.execute(query)
#         arche_cols = cur.fetchall()
#         arche_cols = [c[0] for c in arche_cols if not c[0].startswith('char')]
#         cur.close()
#
#         # Retrieve the monthly demands of the archetypes and create a dict for the clusters/archetype
#         cur = conn.cursor(cursor_factory=pge.RealDictCursor)
#         query = """
#         SELECT {} FROM working_tables.habe_archetypes_weighted
#         """.format(', '.join(arche_cols))
#         cur.execute(query)
#         archetypes = cur.fetchall()
#         archedict = {ar['cluster_label_def']: ar for ar in archetypes}
#         cur.close()
#         with open(os.path.join(savepath, "archedict.pickle"), 'wb') as f:
#             pickle.dump(archedict, f)
#
#         # Even though not really necessary (since we couple with the MATSim-based mobility model), we als retrieve the
#         # microcensus data to estimate public transport and bike demand
#         mc_data = cons.get_microcensus_data(excel={
#             'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#             'sheet': 'PT_and_bike'})
#         with open(os.path.join(savepath, "mc_data.pickle"), 'wb') as f:
#             pickle.dump(mc_data, f)
#
#         # the following lines construct a list of attributes which will be needed for the correct order of the demands
#         habe_attributes = arche_cols
#         idx = habe_attributes.index("cluster_label_def")
#         habe_attributes[idx] = 'householdid'
#         del habe_attributes[habe_attributes.index("cluster_label_name")]
#         conn.close()
#         with open(os.path.join(savepath, "habe_attributes.pickle"), 'wb') as f:
#             pickle.dump(habe_attributes, f)
#
#     cons.get_time("Time for intialization", t)  # last run: 00:00:18
#
#     t_dem = time.time()
#     print("start multi-processing...")
#
#     q = mp.Queue()  # create the queue
#     ps = [mp.Process(
#         target=exportdem_worker,
#         args=(q, attributes, savepath, i)) for i in range(cpus)]
#
#     #bfsnr_all = [4566, 4001, 230, 4946] #for debugging
#     for bfsnr in bfsnr_all:
#         q.put(bfsnr)
#
#     #for termination?
#     for _ in range(cpus):
#         q.put("killitwithfire_worker")
#
#     #start the multi-processing:
#     for p in ps:
#         p.start()
#
#     for p in ps:
#         p.join()
#
#     cons.get_time("Time exporting data", t_dem)  # last run: 76:26:59 for 2337 municipalities
#     cons.get_time("Total Time", t)  # last run: 76:27:17 for 2337 municipalities + 1:22:32 for the rest

#----------------------------------------------------------------------------------------------------------------------
# Write consumption demand to PG-DB
#----------------------------------------------------------------------------------------------------------------------
# print("Writing cluster choices...")
# t = time.time()
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\09_Classification\OnlyMobility\calibrated_classifier_mob.pickle", 'rb') as f:
#     clf = pickle.load(f)
#
# clf_classes = copy.deepcopy(clf.classes_)
# del clf
#
# conn = cons.get_pg_connection()
#
# clust_choice_attrs = ['cluster_label_def_{} float'.format(i) for i in clf_classes]
# dbstruct = ['householdid bigint', 'chosen_cluster_label_def int'] + clust_choice_attrs
#
# hh_cluster_choices = {
#     'filesloc': r"U:\HEIA\Results\Consumption\hh_cluster_choices",
#     'dbstruct' : dbstruct,
#     'tablename': 'working_tables.hh_cluster_choices'}
#
# csv2pg_cluster_choices = mob.CSV2PG(**hh_cluster_choices)
#
# csv2pg_cluster_choices.create_dbtable(conn)
#
# csv2pg_cluster_choices.insert_dbtable(conn)
#
# conn.close()
#
# cons.get_time("Time for writing cluster choices", t)  # last run: 00:05:26 (directly from Linux-Server, but at ETH)

# print("Writing demands...")
# t_dem = time.time()
#
# conn = cons.get_pg_connection()
#
# with open(r"U:\HEIA\Data\Consumption\Export4Linux\habe_attributes.pickle", 'rb') as f:
#     habe_attributes = pickle.load(f)
#
# dbstruct = ["{} float".format(attr) if not attr=='householdid' else "{} bigint".format(attr) for attr in habe_attributes if not attr.startswith('char')]
#
# hh_consumption_demands = {
#     'filesloc': r"U:\HEIA\Results\Consumption\hh_consumption_demand",
#     'dbstruct' : dbstruct,
#     'tablename': 'results.hh_consumption_demand'}
#
# csv2pg_consumption_dem = mob.CSV2PG(**hh_consumption_demands)
#
# csv2pg_consumption_dem.create_dbtable(conn)
#
# csv2pg_consumption_dem.insert_dbtable(conn)
#
# conn.close()
#
# cons.get_time("Time for writing demands", t_dem)  # last run: 00:45:02 (directly from Linux-Server, but at ETH, same for at home: 06:26:03)

#----------------------------------------------------------------------------------------------------------------------
# Consumption LCA:
#----------------------------------------------------------------------------------------------------------------------

#****************************
# Compute-Pre-LCA
#****************************

# # In a first step, we have to perform an LCA in order to prepare the technosphere matrix. For a successful reuse of the
# # factorized LCI-Matrix, we need a functional unit which comprises processes from all databases needed.
# # Since the ecoinvent-based activities are in a different brightway-project than EXIOBASE-activities, we have to run
# # the two analyses in parallel
#
# tlca = time.time()
# print("Doing Pre-LCA...")
# pseudo_ei_fu = {
#     ('ecoinvent 3.3 cutoff', '9e32482c441075cbecb151627e5490c4'): 1,
#     ('Agribalyse 1.2', '6ff1d7d24c730753db935815dce15020'): 1,
#     ('heia', 'fruitnes'): 1
# }
# # In the methods-dict, we define different environmental indicators which shall be assessed. Note that the keys will
# # be used for csv-files-names
# methods = {
#     'ipcc': ('IPCC 2013', 'climate change', 'GWP 100a'),
#     'rec_end_tot': ('ReCiPe Endpoint (H,A)', 'total', 'total'),
#     'ubp_tot': ('ecological scarcity 2013', 'total', 'total')
# }
#
# # not computed anymore:
# # 'rec_end_eco': ('ReCiPe Endpoint (H,A)', 'ecosystem quality', 'total'),
# #     'rec_end_hh': ('ReCiPe Endpoint (H,A)', 'human health', 'total'),
# #     'rec_end_res': ('ReCiPe Endpoint (H,A)', 'resources', 'total'),
#
# met = methods['ipcc']
# bw2.projects.set_current('heia33')
# ei_lca = bw2.LCA(pseudo_ei_fu, method=met)
# ei_lca.lci()
# ei_lca.lcia()
#
# bw2.projects.set_current('exiobase_industry_workaround')
# pseudo_ex_fu = {('EXIOBASE 2.2', 'Manufacture of tobacco products:LV'): 1}
# ex_lca = bw2.LCA(pseudo_ex_fu, method=met)
# ex_lca.lci(factorize=True)
# ex_lca.lcia()
#
# cons.get_time("Pre-LCA done", tlca)  # last run: 00:00:01
#
# tfact = time.time()
# print("Computing LCA-multiplication factors...")
#
# # With the ConsumptionLCA-class we instantiate an object which contains multiplication factors for all processes and all
# # methods which need to be analyzed. This procedure turned out to be much faster than doing LCA directly with Brightway
# excel={'path': r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\Consumption_Cockpit.xlsx",
#    'sheet': 'LCA-Modelling (STATPOP)',
#        'wwtp': 'WWTPs',
#        'heating': 'gws_building_energy'}
# conslca = cons.ConsumptionLCA(ei_lca, ex_lca, methods, excel, archetypes=False)
#
# # For Post-Analysis and for documentation purposes, we will store the ConsumptionLCA-class
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\11_Consumption_LCA\consumption_lca.pickle", 'wb') as f:
#     pickle.dump(conslca, f)
#
# cons.get_time("Computation of factors done", tfact)  # last run: 00:00:21
# cons.get_time("Total initialization time", tlca)  # last run: 00:00:24

#****************************
# Compute HH-consumption LCA
#****************************

# cpus = mp.cpu_count()   # count CPUs
#
# # Most of the following code was kindly provided by Rene Buffat
# def conslca_worker(q, conslca, savepath, wid): # if I got it right: q is the queue (list of municipality numbers) while wid stands for the CPU
#
#     finished = False
#     while not finished:
#         try:
#
#             vals = q.get() # take a value from the queue
#
#             if vals == 'killitwithfire_worker':  # if-loop to terminate the processing
#                 logging.info(
#                     "worker {} finished - received good kill".format(wid))
#                 finished = True
#                 break
#             bfsnr = vals
#
#             cons.do_consumption_lca(bfsnr, conslca, savepath)   # core of the function: calling the "actual" function
#
#         except Empty:
#             logging.info("worker {} finished - empty".format(wid))
#             finished = True
#             break
#         except Exception as e:
#             logging.exception("wid: {} / {}".format(wid, str(e)))
#
# #in the case of multiprocessing, the main-statement is necessary
# if __name__ == '__main__':
#     t = time.time()
#
#     #Extract all municipality numbers from the PostGIS-DB:
#     conn = cons.get_pg_connection()
#     cur = conn.cursor()
#     query = """
#     SELECT DISTINCT hh.bfs_nummer FROM working_tables.hh hh
#     """
#     cur.execute(query)
#     bfsnr_all = []
#     for row in cur:
#         bfsnr_all.append(row[0])
#     cur.close()
#     conn.close()
#
#     # The following try-statement tries to extract BFSNRs which were already computed. Furthermore, we exclude the 10 largest
#     # municipalities in this first attempt in order to speed up computations.
#     try:
#         bfsnr_deja = os.listdir(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\11_Consumption_LCA\Intermediate_Results_Repository\Results_Consumption_ipcc")
#         bfsnr_deja = [int(f.split('_')[-1].replace('.csv', '')) for f in bfsnr_deja]
#         bfsnr_all = list(set(bfsnr_all) - set(bfsnr_deja) - set([261, 6621, 2701, 5586, 351, 230, 1061, 3203, 5192, 371]))
#     except:
#         bfsnr_all = list(set(bfsnr_all) - set([261, 6621, 2701, 5586, 351, 230, 1061, 3203, 5192, 371]))
#
#     with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\11_Consumption_LCA\consumption_lca.pickle", 'rb') as f:
#         conslca = pickle.load(f)
#
#     print("start multiprocessing...")
#
#     savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\11_Consumption_LCA\Intermediate_Results_Repository\Results_Consumption_"
#
#     q = mp.Queue()  # create the queue
#     ps = [mp.Process(
#         target=conslca_worker,
#         args=(q, conslca, savepath, i)) for i in range(cpus)]
#
#     #bfsnr_all = [4566, 4001, 230, 4946] #for debugging
#     for bfsnr in bfsnr_all:
#         q.put(bfsnr)
#
#     #for termination?
#     for _ in range(cpus):
#         q.put("killitwithfire_worker")
#
#     #start the multi-processing:
#     for p in ps:
#         p.start()
#
#     for p in ps:
#         p.join()
#
#     cons.get_time("Total time", t)  # last run: 11:18:00 + 00:07:23

#****************************
# Compute HH-consumption LCA for the 10 largest municipalities
#****************************
#
# t = time.time()
#
# with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\11_Consumption_LCA\consumption_lca.pickle", 'rb') as f:
#     conslca = pickle.load(f)
#
# savepath = r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\11_Consumption_LCA\Intermediate_Results_Repository\Results_Consumption_"
#
# for bfsnr in [6621, 2701, 5586, 351, 230, 1061, 3203, 5192, 371, 261]:
#     t_muni = time.time()
#     print("computing {}".format(bfsnr))
#     cons.do_consumption_lca(bfsnr, conslca, savepath)
#     cons.get_time("time for {}".format(bfsnr), t_muni)
# cons.get_time("Total time", t)  # last run: 04:17:18
# # time for 6621: 00:38:27
# # time for 2701: 00:40:05
# # time for 5586: 00:31:22
# # time for 351: 00:24:53
# # time for 230: 00:16:35
# # time for 1061: 00:13:26
# # time for 3203: 00:11:02
# # time for 5192: 00:09:14
# # time for 371: 00:08:02
# # time for 261: 01:04:12

#****************************
# Write LCA-Results to PG-DB
#****************************

t = time.time()
conn = cons.get_pg_connection()

with open(r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\11_Consumption_LCA\consumption_lca.pickle", 'rb') as f:
    conslca = pickle.load(f)

methods = conslca.methods_

dbstruct = ['householdid bigint']
dbstruct += ["{} float".format(attr) for attr in conslca.attributes_list_]
dbstruct += ['mx5723 float']

methods2 = {'ipcc': methods['ipcc'],
            'rec_end_tot': methods['rec_end_tot'],
            'ubp_tot': methods['ubp_tot']
            }

for methodkey in methods2.keys():
    consumption_lca_res = {
        'filesloc':  r"D:\froemelt\Documents\Andi\02_Doktorat\03_Projects\05_HEIA\03_Computations\02_Consumption\11_Consumption_LCA\Intermediate_Results_Repository\Results_Consumption_"+methodkey,
        'dbstruct' : dbstruct,
        'tablename': 'results.hh_consumption_'+methodkey}

    csv2pg_consumption_lca = mob.CSV2PG(**consumption_lca_res)

    csv2pg_consumption_lca.create_dbtable(conn)

    csv2pg_consumption_lca.insert_dbtable(conn)

conn.close()

cons.get_time("Time for writing to PG-DB", t)  # last run: 01:04:03
