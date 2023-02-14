import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve

test01_pred_imrex   = pd.read_csv("../../data/predToolsData/Imrex/test01_pred.csv")
test02_pred_imrex   = pd.read_csv("../../data/predToolsData/Imrex/test02_pred.csv")
test03_pred_imrex   = pd.read_csv("../../data/predToolsData/Imrex/test03_pred.csv")
test04_pred_imrex   = pd.read_csv("../../data/predToolsData/Imrex/test04_pred.csv")
test05_pred_imrex   = pd.read_csv("../../data/predToolsData/Imrex/test05_pred.csv")
test06_pred_imrex   = pd.read_csv("../../data/predToolsData/Imrex/test06_pred.csv")
test07_pred_imrex   = pd.read_csv("../../data/predToolsData/Imrex/test07_pred.csv")
test08_pred_imrex   = pd.read_csv("../../data/predToolsData/Imrex/test08_pred.csv")
test09_pred_imrex   = pd.read_csv("../../data/predToolsData/Imrex/test09_pred.csv")
test10_pred_imrex   = pd.read_csv("../../data/predToolsData/Imrex/test10_pred.csv")
test11_pred_imrex   = pd.read_csv("../../data/predToolsData/Imrex/test11_pred.csv")
test12_pred_imrex   = pd.read_csv("../../data/predToolsData/Imrex/test12_pred.csv")
test13_pred_imrex   = pd.read_csv("../../data/predToolsData/Imrex/test13_pred.csv")
test14_pred_imrex   = pd.read_csv("../../data/predToolsData/Imrex/test14_pred.csv")
test15_pred_imrex   = pd.read_csv("../../data/predToolsData/Imrex/test15_pred.csv")

test01_pred_atmtcr2  = pd.read_csv("../../data/predToolsData/ATMTCR/Retraining/test01_pred.csv")
test02_pred_atmtcr2  = pd.read_csv("../../data/predToolsData/ATMTCR/Retraining/test02_pred.csv")
test03_pred_atmtcr2  = pd.read_csv("../../data/predToolsData/ATMTCR/Retraining/test03_pred.csv")
test04_pred_atmtcr2  = pd.read_csv("../../data/predToolsData/ATMTCR/Retraining/test04_pred.csv")
test05_pred_atmtcr2  = pd.read_csv("../../data/predToolsData/ATMTCR/Retraining/test05_pred.csv")
test06_pred_atmtcr2  = pd.read_csv("../../data/predToolsData/ATMTCR/Retraining/test06_pred.csv")
test07_pred_atmtcr2  = pd.read_csv("../../data/predToolsData/ATMTCR/Retraining/test07_pred.csv")
test08_pred_atmtcr2  = pd.read_csv("../../data/predToolsData/ATMTCR/Retraining/test08_pred.csv")
test09_pred_atmtcr2  = pd.read_csv("../../data/predToolsData/ATMTCR/Retraining/test09_pred.csv")
test10_pred_atmtcr2  = pd.read_csv("../../data/predToolsData/ATMTCR/Retraining/test10_pred.csv")
test11_pred_atmtcr2  = pd.read_csv("../../data/predToolsData/ATMTCR/Retraining/test11_pred.csv")
test12_pred_atmtcr2  = pd.read_csv("../../data/predToolsData/ATMTCR/Retraining/test12_pred.csv")
test13_pred_atmtcr2  = pd.read_csv("../../data/predToolsData/ATMTCR/Retraining/test13_pred.csv")
test14_pred_atmtcr2  = pd.read_csv("../../data/predToolsData/ATMTCR/Retraining/test14_pred.csv")
test15_pred_atmtcr2  = pd.read_csv("../../data/predToolsData/ATMTCR/Retraining/test15_pred.csv")

test01_pred_atmtcr  = pd.read_csv("../../data/predToolsData/ATMTCR/Pretrained/test01_pred.csv")
test02_pred_atmtcr  = pd.read_csv("../../data/predToolsData/ATMTCR/Pretrained/test02_pred.csv")
test03_pred_atmtcr  = pd.read_csv("../../data/predToolsData/ATMTCR/Pretrained/test03_pred.csv")
test04_pred_atmtcr  = pd.read_csv("../../data/predToolsData/ATMTCR/Pretrained/test04_pred.csv")
test05_pred_atmtcr  = pd.read_csv("../../data/predToolsData/ATMTCR/Pretrained/test05_pred.csv")
test06_pred_atmtcr  = pd.read_csv("../../data/predToolsData/ATMTCR/Pretrained/test06_pred.csv")
test07_pred_atmtcr  = pd.read_csv("../../data/predToolsData/ATMTCR/Pretrained/test07_pred.csv")
test08_pred_atmtcr  = pd.read_csv("../../data/predToolsData/ATMTCR/Pretrained/test08_pred.csv")
test09_pred_atmtcr  = pd.read_csv("../../data/predToolsData/ATMTCR/Pretrained/test09_pred.csv")
test10_pred_atmtcr  = pd.read_csv("../../data/predToolsData/ATMTCR/Pretrained/test10_pred.csv")
test11_pred_atmtcr  = pd.read_csv("../../data/predToolsData/ATMTCR/Pretrained/test11_pred.csv")
test12_pred_atmtcr  = pd.read_csv("../../data/predToolsData/ATMTCR/Pretrained/test12_pred.csv")
test13_pred_atmtcr  = pd.read_csv("../../data/predToolsData/ATMTCR/Pretrained/test13_pred.csv")
test14_pred_atmtcr  = pd.read_csv("../../data/predToolsData/ATMTCR/Pretrained/test14_pred.csv")
test15_pred_atmtcr  = pd.read_csv("../../data/predToolsData/ATMTCR/Pretrained/test15_pred.csv")

test01_pred_nettcr2  = pd.read_csv("../../data/predToolsData/nettcr/Retraining/test01_pred.csv")
test02_pred_nettcr2  = pd.read_csv("../../data/predToolsData/nettcr/Retraining/test02_pred.csv")
test03_pred_nettcr2  = pd.read_csv("../../data/predToolsData/nettcr/Retraining/test03_pred.csv")
test04_pred_nettcr2  = pd.read_csv("../../data/predToolsData/nettcr/Retraining/test04_pred.csv")
test05_pred_nettcr2  = pd.read_csv("../../data/predToolsData/nettcr/Retraining/test05_pred.csv")
test06_pred_nettcr2  = pd.read_csv("../../data/predToolsData/nettcr/Retraining/test06_pred.csv")
test07_pred_nettcr2  = pd.read_csv("../../data/predToolsData/nettcr/Retraining/test07_pred.csv")
test08_pred_nettcr2  = pd.read_csv("../../data/predToolsData/nettcr/Retraining/test08_pred.csv")
test09_pred_nettcr2  = pd.read_csv("../../data/predToolsData/nettcr/Retraining/test09_pred.csv")
test10_pred_nettcr2  = pd.read_csv("../../data/predToolsData/nettcr/Retraining/test10_pred.csv")
test11_pred_nettcr2  = pd.read_csv("../../data/predToolsData/nettcr/Retraining/test11_pred.csv")
test12_pred_nettcr2  = pd.read_csv("../../data/predToolsData/nettcr/Retraining/test12_pred.csv")
test13_pred_nettcr2  = pd.read_csv("../../data/predToolsData/nettcr/Retraining/test13_pred.csv")
test14_pred_nettcr2  = pd.read_csv("../../data/predToolsData/nettcr/Retraining/test14_pred.csv")
test15_pred_nettcr2  = pd.read_csv("../../data/predToolsData/nettcr/Retraining/test15_pred.csv")

test01_pred_nettcr  = pd.read_csv("../../data/predToolsData/nettcr/Pretrained/test01_pred.csv")
test02_pred_nettcr  = pd.read_csv("../../data/predToolsData/nettcr/Pretrained/test02_pred.csv")
test03_pred_nettcr  = pd.read_csv("../../data/predToolsData/nettcr/Pretrained/test03_pred.csv")
test04_pred_nettcr  = pd.read_csv("../../data/predToolsData/nettcr/Pretrained/test04_pred.csv")
test05_pred_nettcr  = pd.read_csv("../../data/predToolsData/nettcr/Pretrained/test05_pred.csv")
test06_pred_nettcr  = pd.read_csv("../../data/predToolsData/nettcr/Pretrained/test06_pred.csv")
test07_pred_nettcr  = pd.read_csv("../../data/predToolsData/nettcr/Pretrained/test07_pred.csv")
test08_pred_nettcr  = pd.read_csv("../../data/predToolsData/nettcr/Pretrained/test08_pred.csv")
test09_pred_nettcr  = pd.read_csv("../../data/predToolsData/nettcr/Pretrained/test09_pred.csv")
test10_pred_nettcr  = pd.read_csv("../../data/predToolsData/nettcr/Pretrained/test10_pred.csv")
test11_pred_nettcr  = pd.read_csv("../../data/predToolsData/nettcr/Pretrained/test11_pred.csv")
test12_pred_nettcr  = pd.read_csv("../../data/predToolsData/nettcr/Pretrained/test12_pred.csv")
test13_pred_nettcr  = pd.read_csv("../../data/predToolsData/nettcr/Pretrained/test13_pred.csv")
test14_pred_nettcr  = pd.read_csv("../../data/predToolsData/nettcr/Pretrained/test14_pred.csv")
test15_pred_nettcr  = pd.read_csv("../../data/predToolsData/nettcr/Pretrained/test15_pred.csv")

test01_pred_epitcr = pd.read_csv("../../data/predepTCRData/withoutMHC/test01_predict_proba.csv")
test02_pred_epitcr = pd.read_csv("../../data/predepTCRData/withoutMHC/test02_predict_proba.csv")
test03_pred_epitcr = pd.read_csv("../../data/predepTCRData/withoutMHC/test03_predict_proba.csv")
test04_pred_epitcr = pd.read_csv("../../data/predepTCRData/withoutMHC/test04_predict_proba.csv")
test05_pred_epitcr = pd.read_csv("../../data/predepTCRData/withoutMHC/test05_predict_proba.csv")
test06_pred_epitcr = pd.read_csv("../../data/predepTCRData/withoutMHC/test06_predict_proba.csv")
test07_pred_epitcr = pd.read_csv("../../data/predepTCRData/withoutMHC/test07_predict_proba.csv")
test08_pred_epitcr = pd.read_csv("../../data/predepTCRData/withoutMHC/test08_predict_proba.csv")
test09_pred_epitcr = pd.read_csv("../../data/predepTCRData/withoutMHC/test09_predict_proba.csv")
test10_pred_epitcr = pd.read_csv("../../data/predepTCRData/withoutMHC/test10_predict_proba.csv")
test11_pred_epitcr = pd.read_csv("../../data/predepTCRData/withoutMHC/test11_predict_proba.csv")
test12_pred_epitcr = pd.read_csv("../../data/predepTCRData/withoutMHC/test12_predict_proba.csv")
test13_pred_epitcr = pd.read_csv("../../data/predepTCRData/withoutMHC/test13_predict_proba.csv")
test14_pred_epitcr = pd.read_csv("../../data/predepTCRData/withoutMHC/test14_predict_proba.csv")
test15_pred_epitcr = pd.read_csv("../../data/predepTCRData/withoutMHC/test15_predict_proba.csv")

test01_pred_epitcr_mhc = pd.read_csv("../../data/predepTCRData/withMHC/test01_predict_proba.csv")
test02_pred_epitcr_mhc = pd.read_csv("../../data/predepTCRData/withMHC/test02_predict_proba.csv")
test03_pred_epitcr_mhc = pd.read_csv("../../data/predepTCRData/withMHC/test03_predict_proba.csv")
test04_pred_epitcr_mhc = pd.read_csv("../../data/predepTCRData/withMHC/test04_predict_proba.csv")
test05_pred_epitcr_mhc = pd.read_csv("../../data/predepTCRData/withMHC/test05_predict_proba.csv")
test06_pred_epitcr_mhc = pd.read_csv("../../data/predepTCRData/withMHC/test06_predict_proba.csv")
test07_pred_epitcr_mhc = pd.read_csv("../../data/predepTCRData/withMHC/test07_predict_proba.csv")
test08_pred_epitcr_mhc = pd.read_csv("../../data/predepTCRData/withMHC/test08_predict_proba.csv")
test09_pred_epitcr_mhc = pd.read_csv("../../data/predepTCRData/withMHC/test09_predict_proba.csv")

test01_pred_pmtnet = pd.read_csv("../../data/predToolsData/pMTnet/test01_pred.csv")
test02_pred_pmtnet = pd.read_csv("../../data/predToolsData/pMTnet/test02_pred.csv")
test03_pred_pmtnet = pd.read_csv("../../data/predToolsData/pMTnet/test03_pred.csv")
test04_pred_pmtnet = pd.read_csv("../../data/predToolsData/pMTnet/test04_pred.csv")
test05_pred_pmtnet = pd.read_csv("../../data/predToolsData/pMTnet/test05_pred.csv")
test06_pred_pmtnet = pd.read_csv("../../data/predToolsData/pMTnet/test06_pred.csv")
test07_pred_pmtnet = pd.read_csv("../../data/predToolsData/pMTnet/test07_pred.csv")
test08_pred_pmtnet = pd.read_csv("../../data/predToolsData/pMTnet/test08_pred.csv")
test09_pred_pmtnet = pd.read_csv("../../data/predToolsData/pMTnet/test09_pred.csv")

prob01_epitcr = test01_pred_epitcr["predict_proba"]
prob02_epitcr = test02_pred_epitcr["predict_proba"]
prob03_epitcr = test03_pred_epitcr["predict_proba"]
prob04_epitcr = test04_pred_epitcr["predict_proba"]
prob05_epitcr = test05_pred_epitcr["predict_proba"]
prob06_epitcr = test06_pred_epitcr["predict_proba"]
prob07_epitcr = test07_pred_epitcr["predict_proba"]
prob08_epitcr = test08_pred_epitcr["predict_proba"]
prob09_epitcr = test09_pred_epitcr["predict_proba"]
prob10_epitcr = test10_pred_epitcr["predict_proba"]
prob11_epitcr = test11_pred_epitcr["predict_proba"]
prob12_epitcr = test12_pred_epitcr["predict_proba"]
prob13_epitcr = test13_pred_epitcr["predict_proba"]
prob14_epitcr = test14_pred_epitcr["predict_proba"]
prob15_epitcr = test15_pred_epitcr["predict_proba"]


y_test01_epitcr = test01_pred_epitcr["binder"].to_numpy()
y_test02_epitcr = test02_pred_epitcr["binder"].to_numpy()
y_test03_epitcr = test03_pred_epitcr["binder"].to_numpy()
y_test04_epitcr = test04_pred_epitcr["binder"].to_numpy()
y_test05_epitcr = test05_pred_epitcr["binder"].to_numpy()
y_test06_epitcr = test06_pred_epitcr["binder"].to_numpy()
y_test07_epitcr = test07_pred_epitcr["binder"].to_numpy()
y_test08_epitcr = test08_pred_epitcr["binder"].to_numpy()
y_test09_epitcr = test09_pred_epitcr["binder"].to_numpy()
y_test10_epitcr = test10_pred_epitcr["binder"].to_numpy()
y_test11_epitcr = test11_pred_epitcr["binder"].to_numpy()
y_test12_epitcr = test12_pred_epitcr["binder"].to_numpy()
y_test13_epitcr = test13_pred_epitcr["binder"].to_numpy()
y_test14_epitcr = test14_pred_epitcr["binder"].to_numpy()
y_test15_epitcr = test15_pred_epitcr["binder"].to_numpy()


fpr01_epitcr, tpr01_epitcr, thresholds = roc_curve(y_test01_epitcr, prob01_epitcr, drop_intermediate=False)
fpr02_epitcr, tpr02_epitcr, thresholds = roc_curve(y_test02_epitcr, prob02_epitcr, drop_intermediate=False)
fpr03_epitcr, tpr03_epitcr, thresholds = roc_curve(y_test03_epitcr, prob03_epitcr, drop_intermediate=False)
fpr04_epitcr, tpr04_epitcr, thresholds = roc_curve(y_test04_epitcr, prob04_epitcr, drop_intermediate=False)
fpr05_epitcr, tpr05_epitcr, thresholds = roc_curve(y_test05_epitcr, prob05_epitcr, drop_intermediate=False)
fpr06_epitcr, tpr06_epitcr, thresholds = roc_curve(y_test06_epitcr, prob06_epitcr, drop_intermediate=False)
fpr07_epitcr, tpr07_epitcr, thresholds = roc_curve(y_test07_epitcr, prob07_epitcr, drop_intermediate=False)
fpr08_epitcr, tpr08_epitcr, thresholds = roc_curve(y_test08_epitcr, prob08_epitcr, drop_intermediate=False)
fpr09_epitcr, tpr09_epitcr, thresholds = roc_curve(y_test09_epitcr, prob09_epitcr, drop_intermediate=False)
fpr10_epitcr, tpr10_epitcr, thresholds = roc_curve(y_test10_epitcr, prob10_epitcr, drop_intermediate=False)
fpr11_epitcr, tpr11_epitcr, thresholds = roc_curve(y_test11_epitcr, prob11_epitcr, drop_intermediate=False)
fpr12_epitcr, tpr12_epitcr, thresholds = roc_curve(y_test12_epitcr, prob12_epitcr, drop_intermediate=False)
fpr13_epitcr, tpr13_epitcr, thresholds = roc_curve(y_test13_epitcr, prob13_epitcr, drop_intermediate=False)
fpr14_epitcr, tpr14_epitcr, thresholds = roc_curve(y_test14_epitcr, prob14_epitcr, drop_intermediate=False)
fpr15_epitcr, tpr15_epitcr, thresholds = roc_curve(y_test15_epitcr, prob15_epitcr, drop_intermediate=False)

auc_score01_epitcr = auc(fpr01_epitcr, tpr01_epitcr)
auc_score02_epitcr = auc(fpr02_epitcr, tpr02_epitcr)
auc_score03_epitcr = auc(fpr03_epitcr, tpr03_epitcr)
auc_score04_epitcr = auc(fpr04_epitcr, tpr04_epitcr)
auc_score05_epitcr = auc(fpr05_epitcr, tpr05_epitcr)
auc_score06_epitcr = auc(fpr06_epitcr, tpr06_epitcr)
auc_score07_epitcr = auc(fpr07_epitcr, tpr07_epitcr)
auc_score08_epitcr = auc(fpr08_epitcr, tpr08_epitcr)
auc_score09_epitcr = auc(fpr09_epitcr, tpr09_epitcr)
auc_score10_epitcr = auc(fpr10_epitcr, tpr10_epitcr)
auc_score11_epitcr = auc(fpr11_epitcr, tpr11_epitcr)
auc_score12_epitcr = auc(fpr12_epitcr, tpr12_epitcr)
auc_score13_epitcr = auc(fpr13_epitcr, tpr13_epitcr)
auc_score14_epitcr = auc(fpr14_epitcr, tpr14_epitcr)
auc_score15_epitcr = auc(fpr15_epitcr, tpr15_epitcr)

aucs_epitcr =  [auc_score01_epitcr,auc_score02_epitcr,auc_score03_epitcr,
                auc_score04_epitcr,auc_score05_epitcr,auc_score06_epitcr,
                auc_score07_epitcr,auc_score08_epitcr,auc_score09_epitcr,
                auc_score10_epitcr,auc_score11_epitcr,auc_score12_epitcr,
                auc_score13_epitcr,auc_score14_epitcr,auc_score15_epitcr]

prob01_Imrex = test01_pred_imrex["prediction_score"]
prob02_Imrex = test02_pred_imrex["prediction_score"]
prob03_Imrex = test03_pred_imrex["prediction_score"]
prob04_Imrex = test04_pred_imrex["prediction_score"]
prob05_Imrex = test05_pred_imrex["prediction_score"]
prob06_Imrex = test06_pred_imrex["prediction_score"]
prob07_Imrex = test07_pred_imrex["prediction_score"]
prob08_Imrex = test08_pred_imrex["prediction_score"]
prob09_Imrex = test09_pred_imrex["prediction_score"]
prob10_Imrex = test10_pred_imrex["prediction_score"]
prob11_Imrex = test11_pred_imrex["prediction_score"]
prob12_Imrex = test12_pred_imrex["prediction_score"]
prob13_Imrex = test13_pred_imrex["prediction_score"]
prob14_Imrex = test14_pred_imrex["prediction_score"]
prob15_Imrex = test15_pred_imrex["prediction_score"]


y_test01_Imrex = test01_pred_imrex["binder"].to_numpy()
y_test02_Imrex = test02_pred_imrex["binder"].to_numpy()
y_test03_Imrex = test03_pred_imrex["binder"].to_numpy()
y_test04_Imrex = test04_pred_imrex["binder"].to_numpy()
y_test05_Imrex = test05_pred_imrex["binder"].to_numpy()
y_test06_Imrex = test06_pred_imrex["binder"].to_numpy()
y_test07_Imrex = test07_pred_imrex["binder"].to_numpy()
y_test08_Imrex = test08_pred_imrex["binder"].to_numpy()
y_test09_Imrex = test09_pred_imrex["binder"].to_numpy()
y_test10_Imrex = test10_pred_imrex["binder"].to_numpy()
y_test11_Imrex = test11_pred_imrex["binder"].to_numpy()
y_test12_Imrex = test12_pred_imrex["binder"].to_numpy()
y_test13_Imrex = test13_pred_imrex["binder"].to_numpy()
y_test14_Imrex = test14_pred_imrex["binder"].to_numpy()
y_test15_Imrex = test15_pred_imrex["binder"].to_numpy()


fpr01_Imrex, tpr01_Imrex, thresholds = roc_curve(y_test01_Imrex, prob01_Imrex, drop_intermediate=False)
fpr02_Imrex, tpr02_Imrex, thresholds = roc_curve(y_test02_Imrex, prob02_Imrex, drop_intermediate=False)
fpr03_Imrex, tpr03_Imrex, thresholds = roc_curve(y_test03_Imrex, prob03_Imrex, drop_intermediate=False)
fpr04_Imrex, tpr04_Imrex, thresholds = roc_curve(y_test04_Imrex, prob04_Imrex, drop_intermediate=False)
fpr05_Imrex, tpr05_Imrex, thresholds = roc_curve(y_test05_Imrex, prob05_Imrex, drop_intermediate=False)
fpr06_Imrex, tpr06_Imrex, thresholds = roc_curve(y_test06_Imrex, prob06_Imrex, drop_intermediate=False)
fpr07_Imrex, tpr07_Imrex, thresholds = roc_curve(y_test07_Imrex, prob07_Imrex, drop_intermediate=False)
fpr08_Imrex, tpr08_Imrex, thresholds = roc_curve(y_test08_Imrex, prob08_Imrex, drop_intermediate=False)
fpr09_Imrex, tpr09_Imrex, thresholds = roc_curve(y_test09_Imrex, prob09_Imrex, drop_intermediate=False)
fpr10_Imrex, tpr10_Imrex, thresholds = roc_curve(y_test10_Imrex, prob10_Imrex, drop_intermediate=False)
fpr11_Imrex, tpr11_Imrex, thresholds = roc_curve(y_test11_Imrex, prob11_Imrex, drop_intermediate=False)
fpr12_Imrex, tpr12_Imrex, thresholds = roc_curve(y_test12_Imrex, prob12_Imrex, drop_intermediate=False)
fpr13_Imrex, tpr13_Imrex, thresholds = roc_curve(y_test13_Imrex, prob13_Imrex, drop_intermediate=False)
fpr14_Imrex, tpr14_Imrex, thresholds = roc_curve(y_test14_Imrex, prob14_Imrex, drop_intermediate=False)
fpr15_Imrex, tpr15_Imrex, thresholds = roc_curve(y_test15_Imrex, prob15_Imrex, drop_intermediate=False)

auc_score01_Imrex = auc(fpr01_Imrex, tpr01_Imrex)
auc_score02_Imrex = auc(fpr02_Imrex, tpr02_Imrex)
auc_score03_Imrex = auc(fpr03_Imrex, tpr03_Imrex)
auc_score04_Imrex = auc(fpr04_Imrex, tpr04_Imrex)
auc_score05_Imrex = auc(fpr05_Imrex, tpr05_Imrex)
auc_score06_Imrex = auc(fpr06_Imrex, tpr06_Imrex)
auc_score07_Imrex = auc(fpr07_Imrex, tpr07_Imrex)
auc_score08_Imrex = auc(fpr08_Imrex, tpr08_Imrex)
auc_score09_Imrex = auc(fpr09_Imrex, tpr09_Imrex)
auc_score10_Imrex = auc(fpr10_Imrex, tpr10_Imrex)
auc_score11_Imrex = auc(fpr11_Imrex, tpr11_Imrex)
auc_score12_Imrex = auc(fpr12_Imrex, tpr12_Imrex)
auc_score13_Imrex = auc(fpr13_Imrex, tpr13_Imrex)
auc_score14_Imrex = auc(fpr14_Imrex, tpr14_Imrex)
auc_score15_Imrex = auc(fpr15_Imrex, tpr15_Imrex)

aucs_Imrex = [auc_score01_Imrex,auc_score02_Imrex,auc_score03_Imrex,auc_score04_Imrex,
                auc_score05_Imrex,auc_score06_Imrex,auc_score07_Imrex,auc_score08_Imrex,
                auc_score09_Imrex,auc_score10_Imrex,auc_score11_Imrex,auc_score12_Imrex,
                auc_score13_Imrex,auc_score14_Imrex,auc_score15_Imrex]

prob01_nettcr = test01_pred_nettcr["binder_pred"]
prob02_nettcr = test02_pred_nettcr["binder_pred"]
prob03_nettcr = test03_pred_nettcr["binder_pred"]
prob04_nettcr = test04_pred_nettcr["binder_pred"]
prob05_nettcr = test05_pred_nettcr["binder_pred"]
prob06_nettcr = test06_pred_nettcr["binder_pred"]
prob07_nettcr = test07_pred_nettcr["binder_pred"]
prob08_nettcr = test08_pred_nettcr["binder_pred"]
prob09_nettcr = test09_pred_nettcr["binder_pred"]
prob10_nettcr = test10_pred_nettcr["binder_pred"]
prob11_nettcr = test11_pred_nettcr["binder_pred"]
prob12_nettcr = test12_pred_nettcr["binder_pred"]
prob13_nettcr = test13_pred_nettcr["binder_pred"]
prob14_nettcr = test14_pred_nettcr["binder_pred"]
prob15_nettcr = test15_pred_nettcr["binder_pred"]


y_test01_nettcr = test01_pred_nettcr["binder"].to_numpy()
y_test02_nettcr = test02_pred_nettcr["binder"].to_numpy()
y_test03_nettcr = test03_pred_nettcr["binder"].to_numpy()
y_test04_nettcr = test04_pred_nettcr["binder"].to_numpy()
y_test05_nettcr = test05_pred_nettcr["binder"].to_numpy()
y_test06_nettcr = test06_pred_nettcr["binder"].to_numpy()
y_test07_nettcr = test07_pred_nettcr["binder"].to_numpy()
y_test08_nettcr = test08_pred_nettcr["binder"].to_numpy()
y_test09_nettcr = test09_pred_nettcr["binder"].to_numpy()
y_test10_nettcr = test10_pred_nettcr["binder"].to_numpy()
y_test11_nettcr = test11_pred_nettcr["binder"].to_numpy()
y_test12_nettcr = test12_pred_nettcr["binder"].to_numpy()
y_test13_nettcr = test13_pred_nettcr["binder"].to_numpy()
y_test14_nettcr = test14_pred_nettcr["binder"].to_numpy()
y_test15_nettcr = test15_pred_nettcr["binder"].to_numpy()


fpr01_nettcr, tpr01_nettcr, thresholds = roc_curve(y_test01_nettcr, prob01_nettcr, drop_intermediate=False)
fpr02_nettcr, tpr02_nettcr, thresholds = roc_curve(y_test02_nettcr, prob02_nettcr, drop_intermediate=False)
fpr03_nettcr, tpr03_nettcr, thresholds = roc_curve(y_test03_nettcr, prob03_nettcr, drop_intermediate=False)
fpr04_nettcr, tpr04_nettcr, thresholds = roc_curve(y_test04_nettcr, prob04_nettcr, drop_intermediate=False)
fpr05_nettcr, tpr05_nettcr, thresholds = roc_curve(y_test05_nettcr, prob05_nettcr, drop_intermediate=False)
fpr06_nettcr, tpr06_nettcr, thresholds = roc_curve(y_test06_nettcr, prob06_nettcr, drop_intermediate=False)
fpr07_nettcr, tpr07_nettcr, thresholds = roc_curve(y_test07_nettcr, prob07_nettcr, drop_intermediate=False)
fpr08_nettcr, tpr08_nettcr, thresholds = roc_curve(y_test08_nettcr, prob08_nettcr, drop_intermediate=False)
fpr09_nettcr, tpr09_nettcr, thresholds = roc_curve(y_test09_nettcr, prob09_nettcr, drop_intermediate=False)
fpr10_nettcr, tpr10_nettcr, thresholds = roc_curve(y_test10_nettcr, prob10_nettcr, drop_intermediate=False)
fpr11_nettcr, tpr11_nettcr, thresholds = roc_curve(y_test11_nettcr, prob11_nettcr, drop_intermediate=False)
fpr12_nettcr, tpr12_nettcr, thresholds = roc_curve(y_test12_nettcr, prob12_nettcr, drop_intermediate=False)
fpr13_nettcr, tpr13_nettcr, thresholds = roc_curve(y_test13_nettcr, prob13_nettcr, drop_intermediate=False)
fpr14_nettcr, tpr14_nettcr, thresholds = roc_curve(y_test14_nettcr, prob14_nettcr, drop_intermediate=False)
fpr15_nettcr, tpr15_nettcr, thresholds = roc_curve(y_test15_nettcr, prob15_nettcr, drop_intermediate=False)

auc_score01_nettcr = auc(fpr01_nettcr, tpr01_nettcr)
auc_score02_nettcr = auc(fpr02_nettcr, tpr02_nettcr)
auc_score03_nettcr = auc(fpr03_nettcr, tpr03_nettcr)
auc_score04_nettcr = auc(fpr04_nettcr, tpr04_nettcr)
auc_score05_nettcr = auc(fpr05_nettcr, tpr05_nettcr)
auc_score06_nettcr = auc(fpr06_nettcr, tpr06_nettcr)
auc_score07_nettcr = auc(fpr07_nettcr, tpr07_nettcr)
auc_score08_nettcr = auc(fpr08_nettcr, tpr08_nettcr)
auc_score09_nettcr = auc(fpr09_nettcr, tpr09_nettcr)
auc_score10_nettcr = auc(fpr10_nettcr, tpr10_nettcr)
auc_score11_nettcr = auc(fpr11_nettcr, tpr11_nettcr)
auc_score12_nettcr = auc(fpr12_nettcr, tpr12_nettcr)
auc_score13_nettcr = auc(fpr13_nettcr, tpr13_nettcr)
auc_score14_nettcr = auc(fpr14_nettcr, tpr14_nettcr)
auc_score15_nettcr = auc(fpr15_nettcr, tpr15_nettcr)

aucs_nettcr = [  auc_score01_nettcr,auc_score02_nettcr,auc_score03_nettcr,auc_score04_nettcr,
                auc_score05_nettcr,auc_score06_nettcr,auc_score07_nettcr,auc_score08_nettcr,
                auc_score09_nettcr,auc_score10_nettcr,auc_score11_nettcr,auc_score12_nettcr,
                auc_score13_nettcr,auc_score14_nettcr,auc_score15_nettcr]

prob01_nettcr2 = test01_pred_nettcr2["prediction"]
prob02_nettcr2 = test02_pred_nettcr2["prediction"]
prob03_nettcr2 = test03_pred_nettcr2["prediction"]
prob04_nettcr2 = test04_pred_nettcr2["prediction"]
prob05_nettcr2 = test05_pred_nettcr2["prediction"]
prob06_nettcr2 = test06_pred_nettcr2["prediction"]
prob07_nettcr2 = test07_pred_nettcr2["prediction"]
prob08_nettcr2 = test08_pred_nettcr2["prediction"]
prob09_nettcr2 = test09_pred_nettcr2["prediction"]
prob10_nettcr2 = test10_pred_nettcr2["prediction"]
prob11_nettcr2 = test11_pred_nettcr2["prediction"]
prob12_nettcr2 = test12_pred_nettcr2["prediction"]
prob13_nettcr2 = test13_pred_nettcr2["prediction"]
prob14_nettcr2 = test14_pred_nettcr2["prediction"]
prob15_nettcr2 = test15_pred_nettcr2["prediction"]


y_test01_nettcr2 = test01_pred_nettcr2["binder"].to_numpy()
y_test02_nettcr2 = test02_pred_nettcr2["binder"].to_numpy()
y_test03_nettcr2 = test03_pred_nettcr2["binder"].to_numpy()
y_test04_nettcr2 = test04_pred_nettcr2["binder"].to_numpy()
y_test05_nettcr2 = test05_pred_nettcr2["binder"].to_numpy()
y_test06_nettcr2 = test06_pred_nettcr2["binder"].to_numpy()
y_test07_nettcr2 = test07_pred_nettcr2["binder"].to_numpy()
y_test08_nettcr2 = test08_pred_nettcr2["binder"].to_numpy()
y_test09_nettcr2 = test09_pred_nettcr2["binder"].to_numpy()
y_test10_nettcr2 = test10_pred_nettcr2["binder"].to_numpy()
y_test11_nettcr2 = test11_pred_nettcr2["binder"].to_numpy()
y_test12_nettcr2 = test12_pred_nettcr2["binder"].to_numpy()
y_test13_nettcr2 = test13_pred_nettcr2["binder"].to_numpy()
y_test14_nettcr2 = test14_pred_nettcr2["binder"].to_numpy()
y_test15_nettcr2 = test15_pred_nettcr2["binder"].to_numpy()


fpr01_nettcr2, tpr01_nettcr2, thresholds = roc_curve(y_test01_nettcr2, prob01_nettcr2, drop_intermediate=False)
fpr02_nettcr2, tpr02_nettcr2, thresholds = roc_curve(y_test02_nettcr2, prob02_nettcr2, drop_intermediate=False)
fpr03_nettcr2, tpr03_nettcr2, thresholds = roc_curve(y_test03_nettcr2, prob03_nettcr2, drop_intermediate=False)
fpr04_nettcr2, tpr04_nettcr2, thresholds = roc_curve(y_test04_nettcr2, prob04_nettcr2, drop_intermediate=False)
fpr05_nettcr2, tpr05_nettcr2, thresholds = roc_curve(y_test05_nettcr2, prob05_nettcr2, drop_intermediate=False)
fpr06_nettcr2, tpr06_nettcr2, thresholds = roc_curve(y_test06_nettcr2, prob06_nettcr2, drop_intermediate=False)
fpr07_nettcr2, tpr07_nettcr2, thresholds = roc_curve(y_test07_nettcr2, prob07_nettcr2, drop_intermediate=False)
fpr08_nettcr2, tpr08_nettcr2, thresholds = roc_curve(y_test08_nettcr2, prob08_nettcr2, drop_intermediate=False)
fpr09_nettcr2, tpr09_nettcr2, thresholds = roc_curve(y_test09_nettcr2, prob09_nettcr2, drop_intermediate=False)
fpr10_nettcr2, tpr10_nettcr2, thresholds = roc_curve(y_test10_nettcr2, prob10_nettcr2, drop_intermediate=False)
fpr11_nettcr2, tpr11_nettcr2, thresholds = roc_curve(y_test11_nettcr2, prob11_nettcr2, drop_intermediate=False)
fpr12_nettcr2, tpr12_nettcr2, thresholds = roc_curve(y_test12_nettcr2, prob12_nettcr2, drop_intermediate=False)
fpr13_nettcr2, tpr13_nettcr2, thresholds = roc_curve(y_test13_nettcr2, prob13_nettcr2, drop_intermediate=False)
fpr14_nettcr2, tpr14_nettcr2, thresholds = roc_curve(y_test14_nettcr2, prob14_nettcr2, drop_intermediate=False)
fpr15_nettcr2, tpr15_nettcr2, thresholds = roc_curve(y_test15_nettcr2, prob15_nettcr2, drop_intermediate=False)

auc_score01_nettcr2 = auc(fpr01_nettcr2, tpr01_nettcr2)
auc_score02_nettcr2 = auc(fpr02_nettcr2, tpr02_nettcr2)
auc_score03_nettcr2 = auc(fpr03_nettcr2, tpr03_nettcr2)
auc_score04_nettcr2 = auc(fpr04_nettcr2, tpr04_nettcr2)
auc_score05_nettcr2 = auc(fpr05_nettcr2, tpr05_nettcr2)
auc_score06_nettcr2 = auc(fpr06_nettcr2, tpr06_nettcr2)
auc_score07_nettcr2 = auc(fpr07_nettcr2, tpr07_nettcr2)
auc_score08_nettcr2 = auc(fpr08_nettcr2, tpr08_nettcr2)
auc_score09_nettcr2 = auc(fpr09_nettcr2, tpr09_nettcr2)
auc_score10_nettcr2 = auc(fpr10_nettcr2, tpr10_nettcr2)
auc_score11_nettcr2 = auc(fpr11_nettcr2, tpr11_nettcr2)
auc_score12_nettcr2 = auc(fpr12_nettcr2, tpr12_nettcr2)
auc_score13_nettcr2 = auc(fpr13_nettcr2, tpr13_nettcr2)
auc_score14_nettcr2 = auc(fpr14_nettcr2, tpr14_nettcr2)
auc_score15_nettcr2 = auc(fpr15_nettcr2, tpr15_nettcr2)

aucs_nettcr2 = [  auc_score01_nettcr2,auc_score02_nettcr2,auc_score03_nettcr2,auc_score04_nettcr2,
                auc_score05_nettcr2,auc_score06_nettcr2,auc_score07_nettcr2,auc_score08_nettcr2,
                auc_score09_nettcr2,auc_score10_nettcr2,auc_score11_nettcr2,auc_score12_nettcr2,
                auc_score13_nettcr2,auc_score14_nettcr2,auc_score15_nettcr2]

prob01_atmtcr = test01_pred_atmtcr["predict_proba"]
prob02_atmtcr = test02_pred_atmtcr["predict_proba"]
prob03_atmtcr = test03_pred_atmtcr["predict_proba"]
prob04_atmtcr = test04_pred_atmtcr["predict_proba"]
prob05_atmtcr = test05_pred_atmtcr["predict_proba"]
prob06_atmtcr = test06_pred_atmtcr["predict_proba"]
prob07_atmtcr = test07_pred_atmtcr["predict_proba"]
prob08_atmtcr = test08_pred_atmtcr["predict_proba"]
prob09_atmtcr = test09_pred_atmtcr["predict_proba"]
prob10_atmtcr = test10_pred_atmtcr["predict_proba"]
prob11_atmtcr = test11_pred_atmtcr["predict_proba"]
prob12_atmtcr = test12_pred_atmtcr["predict_proba"]
prob13_atmtcr = test13_pred_atmtcr["predict_proba"]
prob14_atmtcr = test14_pred_atmtcr["predict_proba"]
prob15_atmtcr = test15_pred_atmtcr["predict_proba"]


y_test01_atmtcr = test01_pred_atmtcr["binder"].to_numpy()
y_test02_atmtcr = test02_pred_atmtcr["binder"].to_numpy()
y_test03_atmtcr = test03_pred_atmtcr["binder"].to_numpy()
y_test04_atmtcr = test04_pred_atmtcr["binder"].to_numpy()
y_test05_atmtcr = test05_pred_atmtcr["binder"].to_numpy()
y_test06_atmtcr = test06_pred_atmtcr["binder"].to_numpy()
y_test07_atmtcr = test07_pred_atmtcr["binder"].to_numpy()
y_test08_atmtcr = test08_pred_atmtcr["binder"].to_numpy()
y_test09_atmtcr = test09_pred_atmtcr["binder"].to_numpy()
y_test10_atmtcr = test10_pred_atmtcr["binder"].to_numpy()
y_test11_atmtcr = test11_pred_atmtcr["binder"].to_numpy()
y_test12_atmtcr = test12_pred_atmtcr["binder"].to_numpy()
y_test13_atmtcr = test13_pred_atmtcr["binder"].to_numpy()
y_test14_atmtcr = test14_pred_atmtcr["binder"].to_numpy()
y_test15_atmtcr = test15_pred_atmtcr["binder"].to_numpy()


fpr01_atmtcr, tpr01_atmtcr, thresholds = roc_curve(y_test01_atmtcr, prob01_atmtcr, drop_intermediate=False)
fpr02_atmtcr, tpr02_atmtcr, thresholds = roc_curve(y_test02_atmtcr, prob02_atmtcr, drop_intermediate=False)
fpr03_atmtcr, tpr03_atmtcr, thresholds = roc_curve(y_test03_atmtcr, prob03_atmtcr, drop_intermediate=False)
fpr04_atmtcr, tpr04_atmtcr, thresholds = roc_curve(y_test04_atmtcr, prob04_atmtcr, drop_intermediate=False)
fpr05_atmtcr, tpr05_atmtcr, thresholds = roc_curve(y_test05_atmtcr, prob05_atmtcr, drop_intermediate=False)
fpr06_atmtcr, tpr06_atmtcr, thresholds = roc_curve(y_test06_atmtcr, prob06_atmtcr, drop_intermediate=False)
fpr07_atmtcr, tpr07_atmtcr, thresholds = roc_curve(y_test07_atmtcr, prob07_atmtcr, drop_intermediate=False)
fpr08_atmtcr, tpr08_atmtcr, thresholds = roc_curve(y_test08_atmtcr, prob08_atmtcr, drop_intermediate=False)
fpr09_atmtcr, tpr09_atmtcr, thresholds = roc_curve(y_test09_atmtcr, prob09_atmtcr, drop_intermediate=False)
fpr10_atmtcr, tpr10_atmtcr, thresholds = roc_curve(y_test10_atmtcr, prob10_atmtcr, drop_intermediate=False)
fpr11_atmtcr, tpr11_atmtcr, thresholds = roc_curve(y_test11_atmtcr, prob11_atmtcr, drop_intermediate=False)
fpr12_atmtcr, tpr12_atmtcr, thresholds = roc_curve(y_test12_atmtcr, prob12_atmtcr, drop_intermediate=False)
fpr13_atmtcr, tpr13_atmtcr, thresholds = roc_curve(y_test13_atmtcr, prob13_atmtcr, drop_intermediate=False)
fpr14_atmtcr, tpr14_atmtcr, thresholds = roc_curve(y_test14_atmtcr, prob14_atmtcr, drop_intermediate=False)
fpr15_atmtcr, tpr15_atmtcr, thresholds = roc_curve(y_test15_atmtcr, prob15_atmtcr, drop_intermediate=False)

auc_score01_atmtcr = auc(fpr01_atmtcr, tpr01_atmtcr)
auc_score02_atmtcr = auc(fpr02_atmtcr, tpr02_atmtcr)
auc_score03_atmtcr = auc(fpr03_atmtcr, tpr03_atmtcr)
auc_score04_atmtcr = auc(fpr04_atmtcr, tpr04_atmtcr)
auc_score05_atmtcr = auc(fpr05_atmtcr, tpr05_atmtcr)
auc_score06_atmtcr = auc(fpr06_atmtcr, tpr06_atmtcr)
auc_score07_atmtcr = auc(fpr07_atmtcr, tpr07_atmtcr)
auc_score08_atmtcr = auc(fpr08_atmtcr, tpr08_atmtcr)
auc_score09_atmtcr = auc(fpr09_atmtcr, tpr09_atmtcr)
auc_score10_atmtcr = auc(fpr10_atmtcr, tpr10_atmtcr)
auc_score11_atmtcr = auc(fpr11_atmtcr, tpr11_atmtcr)
auc_score12_atmtcr = auc(fpr12_atmtcr, tpr12_atmtcr)
auc_score13_atmtcr = auc(fpr13_atmtcr, tpr13_atmtcr)
auc_score14_atmtcr = auc(fpr14_atmtcr, tpr14_atmtcr)
auc_score15_atmtcr = auc(fpr15_atmtcr, tpr15_atmtcr)

aucs_atmtcr = [ auc_score01_atmtcr,auc_score02_atmtcr,auc_score03_atmtcr,auc_score04_atmtcr,
                auc_score05_atmtcr,auc_score06_atmtcr,auc_score07_atmtcr,auc_score08_atmtcr,
                auc_score09_atmtcr,auc_score10_atmtcr,auc_score11_atmtcr,auc_score12_atmtcr,
                auc_score13_atmtcr,auc_score14_atmtcr,auc_score15_atmtcr]

prob01_atmtcr2 = test01_pred_atmtcr2["prediction"]
prob02_atmtcr2 = test02_pred_atmtcr2["prediction"]
prob03_atmtcr2 = test03_pred_atmtcr2["prediction"]
prob04_atmtcr2 = test04_pred_atmtcr2["prediction"]
prob05_atmtcr2 = test05_pred_atmtcr2["prediction"]
prob06_atmtcr2 = test06_pred_atmtcr2["prediction"]
prob07_atmtcr2 = test07_pred_atmtcr2["prediction"]
prob08_atmtcr2 = test08_pred_atmtcr2["prediction"]
prob09_atmtcr2 = test09_pred_atmtcr2["prediction"]
prob10_atmtcr2 = test10_pred_atmtcr2["prediction"]
prob11_atmtcr2 = test11_pred_atmtcr2["prediction"]
prob12_atmtcr2 = test12_pred_atmtcr2["prediction"]
prob13_atmtcr2 = test13_pred_atmtcr2["prediction"]
prob14_atmtcr2 = test14_pred_atmtcr2["prediction"]
prob15_atmtcr2 = test15_pred_atmtcr2["prediction"]


y_test01_atmtcr2 = test01_pred_atmtcr2["binder"].to_numpy()
y_test02_atmtcr2 = test02_pred_atmtcr2["binder"].to_numpy()
y_test03_atmtcr2 = test03_pred_atmtcr2["binder"].to_numpy()
y_test04_atmtcr2 = test04_pred_atmtcr2["binder"].to_numpy()
y_test05_atmtcr2 = test05_pred_atmtcr2["binder"].to_numpy()
y_test06_atmtcr2 = test06_pred_atmtcr2["binder"].to_numpy()
y_test07_atmtcr2 = test07_pred_atmtcr2["binder"].to_numpy()
y_test08_atmtcr2 = test08_pred_atmtcr2["binder"].to_numpy()
y_test09_atmtcr2 = test09_pred_atmtcr2["binder"].to_numpy()
y_test10_atmtcr2 = test10_pred_atmtcr2["binder"].to_numpy()
y_test11_atmtcr2 = test11_pred_atmtcr2["binder"].to_numpy()
y_test12_atmtcr2 = test12_pred_atmtcr2["binder"].to_numpy()
y_test13_atmtcr2 = test13_pred_atmtcr2["binder"].to_numpy()
y_test14_atmtcr2 = test14_pred_atmtcr2["binder"].to_numpy()
y_test15_atmtcr2 = test15_pred_atmtcr2["binder"].to_numpy()


fpr01_atmtcr2, tpr01_atmtcr2, thresholds = roc_curve(y_test01_atmtcr2, prob01_atmtcr2, drop_intermediate=False)
fpr02_atmtcr2, tpr02_atmtcr2, thresholds = roc_curve(y_test02_atmtcr2, prob02_atmtcr2, drop_intermediate=False)
fpr03_atmtcr2, tpr03_atmtcr2, thresholds = roc_curve(y_test03_atmtcr2, prob03_atmtcr2, drop_intermediate=False)
fpr04_atmtcr2, tpr04_atmtcr2, thresholds = roc_curve(y_test04_atmtcr2, prob04_atmtcr2, drop_intermediate=False)
fpr05_atmtcr2, tpr05_atmtcr2, thresholds = roc_curve(y_test05_atmtcr2, prob05_atmtcr2, drop_intermediate=False)
fpr06_atmtcr2, tpr06_atmtcr2, thresholds = roc_curve(y_test06_atmtcr2, prob06_atmtcr2, drop_intermediate=False)
fpr07_atmtcr2, tpr07_atmtcr2, thresholds = roc_curve(y_test07_atmtcr2, prob07_atmtcr2, drop_intermediate=False)
fpr08_atmtcr2, tpr08_atmtcr2, thresholds = roc_curve(y_test08_atmtcr2, prob08_atmtcr2, drop_intermediate=False)
fpr09_atmtcr2, tpr09_atmtcr2, thresholds = roc_curve(y_test09_atmtcr2, prob09_atmtcr2, drop_intermediate=False)
fpr10_atmtcr2, tpr10_atmtcr2, thresholds = roc_curve(y_test10_atmtcr2, prob10_atmtcr2, drop_intermediate=False)
fpr11_atmtcr2, tpr11_atmtcr2, thresholds = roc_curve(y_test11_atmtcr2, prob11_atmtcr2, drop_intermediate=False)
fpr12_atmtcr2, tpr12_atmtcr2, thresholds = roc_curve(y_test12_atmtcr2, prob12_atmtcr2, drop_intermediate=False)
fpr13_atmtcr2, tpr13_atmtcr2, thresholds = roc_curve(y_test13_atmtcr2, prob13_atmtcr2, drop_intermediate=False)
fpr14_atmtcr2, tpr14_atmtcr2, thresholds = roc_curve(y_test14_atmtcr2, prob14_atmtcr2, drop_intermediate=False)
fpr15_atmtcr2, tpr15_atmtcr2, thresholds = roc_curve(y_test15_atmtcr2, prob15_atmtcr2, drop_intermediate=False)

auc_score01_atmtcr2 = auc(fpr01_atmtcr2, tpr01_atmtcr2)
auc_score02_atmtcr2 = auc(fpr02_atmtcr2, tpr02_atmtcr2)
auc_score03_atmtcr2 = auc(fpr03_atmtcr2, tpr03_atmtcr2)
auc_score04_atmtcr2 = auc(fpr04_atmtcr2, tpr04_atmtcr2)
auc_score05_atmtcr2 = auc(fpr05_atmtcr2, tpr05_atmtcr2)
auc_score06_atmtcr2 = auc(fpr06_atmtcr2, tpr06_atmtcr2)
auc_score07_atmtcr2 = auc(fpr07_atmtcr2, tpr07_atmtcr2)
auc_score08_atmtcr2 = auc(fpr08_atmtcr2, tpr08_atmtcr2)
auc_score09_atmtcr2 = auc(fpr09_atmtcr2, tpr09_atmtcr2)
auc_score10_atmtcr2 = auc(fpr10_atmtcr2, tpr10_atmtcr2)
auc_score11_atmtcr2 = auc(fpr11_atmtcr2, tpr11_atmtcr2)
auc_score12_atmtcr2 = auc(fpr12_atmtcr2, tpr12_atmtcr2)
auc_score13_atmtcr2 = auc(fpr13_atmtcr2, tpr13_atmtcr2)
auc_score14_atmtcr2 = auc(fpr14_atmtcr2, tpr14_atmtcr2)
auc_score15_atmtcr2 = auc(fpr15_atmtcr2, tpr15_atmtcr2)

aucs_atmtcr2 = [ auc_score01_atmtcr2,auc_score02_atmtcr2,auc_score03_atmtcr2,auc_score04_atmtcr2,
                auc_score05_atmtcr2,auc_score06_atmtcr2,auc_score07_atmtcr2,auc_score08_atmtcr2,
                auc_score09_atmtcr2,auc_score10_atmtcr2,auc_score11_atmtcr2,auc_score12_atmtcr2,
                auc_score13_atmtcr2,auc_score14_atmtcr2,auc_score15_atmtcr2]

#linestyle="dotted"
    
fig = plt.figure(figsize=(7,7))
ax  = fig.add_subplot(111)

ax.plot(fpr01_epitcr , tpr01_epitcr , label = 'epiTCR - Mean ROC (AUC = 0.980)',linewidth=1, color="r")
ax.plot(fpr02_epitcr , tpr02_epitcr , linewidth=1, color="r")
ax.plot(fpr03_epitcr , tpr03_epitcr , linewidth=1, color="r")
ax.plot(fpr04_epitcr , tpr04_epitcr , linewidth=1, color="r")
ax.plot(fpr05_epitcr , tpr05_epitcr , linewidth=1, color="r")
ax.plot(fpr06_epitcr , tpr06_epitcr , linewidth=1, color="r")
ax.plot(fpr07_epitcr , tpr07_epitcr , linewidth=1, color="r")
ax.plot(fpr08_epitcr , tpr08_epitcr , linewidth=1, color="r")
ax.plot(fpr09_epitcr , tpr09_epitcr , linewidth=1, color="r")
ax.plot(fpr10_epitcr , tpr10_epitcr , linewidth=1, color="r")
ax.plot(fpr11_epitcr , tpr11_epitcr , linewidth=1, color="r")
ax.plot(fpr12_epitcr , tpr12_epitcr , linewidth=1, color="r")
ax.plot(fpr13_epitcr , tpr13_epitcr , linewidth=1, color="r")
ax.plot(fpr14_epitcr , tpr14_epitcr , linewidth=1, color="r")
ax.plot(fpr15_epitcr , tpr15_epitcr , linewidth=1, color="r")

ax.plot(fpr01_Imrex , tpr01_Imrex , label = 'Imrex - Mean ROC (AUC = 0.551)',linewidth=1, color="g")
ax.plot(fpr02_Imrex , tpr02_Imrex , linewidth=1, color="g")
ax.plot(fpr03_Imrex , tpr03_Imrex , linewidth=1, color="g")
ax.plot(fpr04_Imrex , tpr04_Imrex , linewidth=1, color="g")
ax.plot(fpr05_Imrex , tpr05_Imrex , linewidth=1, color="g")
ax.plot(fpr06_Imrex , tpr06_Imrex , linewidth=1, color="g")
ax.plot(fpr07_Imrex , tpr07_Imrex , linewidth=1, color="g")
ax.plot(fpr08_Imrex , tpr08_Imrex , linewidth=1, color="g")
ax.plot(fpr09_Imrex , tpr09_Imrex , linewidth=1, color="g")
ax.plot(fpr10_Imrex , tpr10_Imrex , linewidth=1, color="g")
ax.plot(fpr11_Imrex , tpr11_Imrex , linewidth=1, color="g")
ax.plot(fpr12_Imrex , tpr12_Imrex , linewidth=1, color="g")
ax.plot(fpr13_Imrex , tpr13_Imrex , linewidth=1, color="g")
ax.plot(fpr14_Imrex , tpr14_Imrex , linewidth=1, color="g")
ax.plot(fpr15_Imrex , tpr15_Imrex , linewidth=1, color="g")

ax.plot(fpr01_nettcr , tpr01_nettcr , label = 'NetTCR - Mean ROC (AUC = 0.518)',linewidth=1, color="purple")
ax.plot(fpr02_nettcr , tpr02_nettcr , linewidth=1, color="purple")
ax.plot(fpr03_nettcr , tpr03_nettcr , linewidth=1, color="purple")
ax.plot(fpr04_nettcr , tpr04_nettcr , linewidth=1, color="purple")
ax.plot(fpr05_nettcr , tpr05_nettcr , linewidth=1, color="purple")
ax.plot(fpr06_nettcr , tpr06_nettcr , linewidth=1, color="purple")
ax.plot(fpr07_nettcr , tpr07_nettcr , linewidth=1, color="purple")
ax.plot(fpr08_nettcr , tpr08_nettcr , linewidth=1, color="purple")
ax.plot(fpr09_nettcr , tpr09_nettcr , linewidth=1, color="purple")
ax.plot(fpr10_nettcr , tpr10_nettcr , linewidth=1, color="purple")
ax.plot(fpr11_nettcr , tpr11_nettcr , linewidth=1, color="purple")
ax.plot(fpr12_nettcr , tpr12_nettcr , linewidth=1, color="purple")
ax.plot(fpr13_nettcr , tpr13_nettcr , linewidth=1, color="purple")
ax.plot(fpr14_nettcr , tpr14_nettcr , linewidth=1, color="purple")
ax.plot(fpr15_nettcr , tpr15_nettcr , linewidth=1, color="purple")

ax.plot(fpr01_nettcr2 , tpr01_nettcr2 , label = 'NetTCR* - Mean ROC (AUC = 0.931)',linewidth=1, color="orange")
ax.plot(fpr02_nettcr2 , tpr02_nettcr2 , linewidth=1, color="orange")
ax.plot(fpr03_nettcr2 , tpr03_nettcr2 , linewidth=1, color="orange")
ax.plot(fpr04_nettcr2 , tpr04_nettcr2 , linewidth=1, color="orange")
ax.plot(fpr05_nettcr2 , tpr05_nettcr2 , linewidth=1, color="orange")
ax.plot(fpr06_nettcr2 , tpr06_nettcr2 , linewidth=1, color="orange")
ax.plot(fpr07_nettcr2 , tpr07_nettcr2 , linewidth=1, color="orange")
ax.plot(fpr08_nettcr2 , tpr08_nettcr2 , linewidth=1, color="orange")
ax.plot(fpr09_nettcr2 , tpr09_nettcr2 , linewidth=1, color="orange")
ax.plot(fpr10_nettcr2 , tpr10_nettcr2 , linewidth=1, color="orange")
ax.plot(fpr11_nettcr2 , tpr11_nettcr2 , linewidth=1, color="orange")
ax.plot(fpr12_nettcr2 , tpr12_nettcr2 , linewidth=1, color="orange")
ax.plot(fpr13_nettcr2 , tpr13_nettcr2 , linewidth=1, color="orange")
ax.plot(fpr14_nettcr2 , tpr14_nettcr2 , linewidth=1, color="orange")
ax.plot(fpr15_nettcr2 , tpr15_nettcr2 , linewidth=1, color="orange")

ax.plot(fpr01_atmtcr , tpr01_atmtcr , label = 'ATM-TCR - Mean ROC (AUC = 0.494)',linewidth=1, color="c")
ax.plot(fpr02_atmtcr , tpr02_atmtcr , linewidth=1, color="c")
ax.plot(fpr03_atmtcr , tpr03_atmtcr , linewidth=1, color="c")
ax.plot(fpr04_atmtcr , tpr04_atmtcr , linewidth=1, color="c")
ax.plot(fpr05_atmtcr , tpr05_atmtcr , linewidth=1, color="c")
ax.plot(fpr06_atmtcr , tpr06_atmtcr , linewidth=1, color="c")
ax.plot(fpr07_atmtcr , tpr07_atmtcr , linewidth=1, color="c")
# ax.plot(fpr08_atmtcr , tpr08_atmtcr , linewidth=1, color="c")
# ax.plot(fpr09_atmtcr , tpr09_atmtcr , linewidth=1, color="c")
# ax.plot(fpr10_atmtcr , tpr10_atmtcr , linewidth=1, color="c")
# ax.plot(fpr11_atmtcr , tpr11_atmtcr , linewidth=1, color="c")
ax.plot(fpr12_atmtcr , tpr12_atmtcr , linewidth=1, color="c")
ax.plot(fpr13_atmtcr , tpr13_atmtcr , linewidth=1, color="c")
ax.plot(fpr14_atmtcr , tpr14_atmtcr , linewidth=1, color="c")
ax.plot(fpr15_atmtcr , tpr15_atmtcr , linewidth=1, color="c")

ax.plot(fpr08_atmtcr2 , tpr08_atmtcr2 , label = 'ATM-TCR* - Mean ROC (AUC = 0.494)',linewidth=1, color="salmon")
# ax.plot(fpr02_atmtcr2 , tpr02_atmtcr2 , linewidth=1, color="salmon")
# ax.plot(fpr03_atmtcr2 , tpr03_atmtcr2 , linewidth=1, color="salmon")
# ax.plot(fpr04_atmtcr2 , tpr04_atmtcr2 , linewidth=1, color="salmon")
# ax.plot(fpr05_atmtcr2 , tpr05_atmtcr2 , linewidth=1, color="salmon")
# ax.plot(fpr06_atmtcr2 , tpr06_atmtcr2 , linewidth=1, color="salmon")
# ax.plot(fpr07_atmtcr2 , tpr07_atmtcr2 , linewidth=1, color="salmon")
# ax.plot(fpr01_atmtcr2 , tpr01_atmtcr2 , linewidth=1, color="salmon")
ax.plot(fpr09_atmtcr2 , tpr09_atmtcr2 , linewidth=1, color="salmon")
ax.plot(fpr10_atmtcr2 , tpr10_atmtcr2 , linewidth=1, color="salmon")
ax.plot(fpr11_atmtcr2 , tpr11_atmtcr2 , linewidth=1, color="salmon")
# ax.plot(fpr12_atmtcr2 , tpr12_atmtcr2 , linewidth=1, color="salmon")
# ax.plot(fpr13_atmtcr2 , tpr13_atmtcr2 , linewidth=1, color="salmon")
# ax.plot(fpr14_atmtcr2 , tpr14_atmtcr2 , linewidth=1, color="salmon")
# ax.plot(fpr15_atmtcr2 , tpr15_atmtcr2 , linewidth=1, color="salmon")

# ax.plot([0, 1], [0, 1], linestyle="dashed", lw=1, color="k", label="Random guess", alpha=0.8)
plt.legend(loc="best")
# plt.title("ROC Curve without MHC", fontsize=11)
plt.xlabel("1 - Specificity", fontsize=11)
plt.ylabel("Sensitivity", fontsize=11)

plt.savefig("../../analysis/figures/benchmarkToolsWithoutMHC.png", dpi=600)
plt.savefig("../../analysis/figures/benchmarkToolsWithoutMHC.pdf", dpi=600)
plt.rcParams.update({'font.size': 11})

plt.show()


