import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve
from sklearn.metrics import classification_report


## epiTCR


test01 = pd.read_csv("../../data/splitData/withoutMHC/test/test01.csv")
test02 = pd.read_csv("../../data/splitData/withoutMHC/test/test02.csv")
test03 = pd.read_csv("../../data/splitData/withoutMHC/test/test03.csv")
test04 = pd.read_csv("../../data/splitData/withoutMHC/test/test04.csv")
test05 = pd.read_csv("../../data/splitData/withoutMHC/test/test05.csv")
test06 = pd.read_csv("../../data/splitData/withoutMHC/test/test06.csv")
test07 = pd.read_csv("../../data/splitData/withoutMHC/test/test07.csv")
test08 = pd.read_csv("../../data/splitData/withoutMHC/test/test08.csv")
test09 = pd.read_csv("../../data/splitData/withoutMHC/test/test09.csv")
test10 = pd.read_csv("../../data/splitData/withoutMHC/test/test10.csv")
test11 = pd.read_csv("../../data/splitData/withoutMHC/test/test11.csv")
test12 = pd.read_csv("../../data/splitData/withoutMHC/test/test12.csv")
test13 = pd.read_csv("../../data/splitData/withoutMHC/test/test13.csv")
test14 = pd.read_csv("../../data/splitData/withoutMHC/test/test14.csv")
test15 = pd.read_csv("../../data/splitData/withoutMHC/test/test15.csv")

test01_mhc = pd.read_csv("../../data/splitData/withoutMHC/test/test01.csv")
test02_mhc = pd.read_csv("../../data/splitData/withoutMHC/test/test02.csv")
test03_mhc = pd.read_csv("../../data/splitData/withoutMHC/test/test03.csv")
test04_mhc = pd.read_csv("../../data/splitData/withoutMHC/test/test04.csv")
test05_mhc = pd.read_csv("../../data/splitData/withoutMHC/test/test05.csv")
test06_mhc = pd.read_csv("../../data/splitData/withoutMHC/test/test06.csv")
test07_mhc = pd.read_csv("../../data/splitData/withoutMHC/test/test07.csv")
test08_mhc = pd.read_csv("../../data/splitData/withoutMHC/test/test08.csv")
test09_mhc = pd.read_csv("../../data/splitData/withoutMHC/test/test09.csv")

data_test = pd.concat([test01,test02,test03,test04,test05,test06,test07,
                       test08,test09,test10,test11,test12,test13,test14,test15], axis=0)
data_test = data_test.reset_index(drop=True)

data_test_mhc = pd.concat([test01_mhc,test02_mhc,test03_mhc,test04_mhc,test05_mhc,
                       test06_mhc,test07_mhc,test08_mhc,test09_mhc], axis=0)
data_test_mhc = data_test_mhc.reset_index(drop=True)

data_test_01 = data_test.sample(n=10000, random_state=42)
data_test_01 = data_test_01.reset_index(drop=True)

data_test_mhc_01 = data_test_mhc.sample(n=10000, random_state=42)
data_test_mhc_01 = data_test_mhc_01.reset_index(drop=True)

data_test_02 = data_test.sample(n=50000, random_state=42)
data_test_02 = data_test_02.reset_index(drop=True)

data_test_mhc_02 = data_test_mhc.sample(n=50000, random_state=42)
data_test_mhc_02 = data_test_mhc_02.reset_index(drop=True)

data_test_03 = data_test.sample(n=200000, random_state=42)
data_test_03 = data_test_03.reset_index(drop=True)

data_test_mhc_03 = data_test_mhc.sample(n=200000, random_state=42)
data_test_mhc_03 = data_test_mhc_03.reset_index(drop=True)

data_test_04 = data_test.sample(n=500000, random_state=42)
data_test_04 = data_test_04.reset_index(drop=True)

data_test_mhc_04 = data_test_mhc.sample(n=500000, random_state=42)
data_test_mhc_04 = data_test_mhc_04.reset_index(drop=True)

data_test_05 = data_test.sample(n=1000000, random_state=42)
data_test_05 = data_test_05.reset_index(drop=True)

data_test_01.to_csv("../../data/randomSampleData/epiTCR/withoutMHC/dataTest01.csv", index=False)
data_test_02.to_csv("../../data/randomSampleData/epiTCR/withoutMHC/dataTest02.csv", index=False)
data_test_03.to_csv("../../data/randomSampleData/epiTCR/withoutMHC/dataTest03.csv", index=False)
data_test_04.to_csv("../../data/randomSampleData/epiTCR/withoutMHC/dataTest04.csv", index=False)
data_test_05.to_csv("../../data/randomSampleData/epiTCR/withoutMHC/dataTest05.csv", index=False)

data_test_mhc_01.to_csv('../../data/randomSampleData/epiTCR/withMHC/dataTest10000.csv', index=False)
data_test_mhc_02.to_csv('../../data/randomSampleData/epiTCR/withMHC/dataTest50000.csv', index=False)
data_test_mhc_03.to_csv('../../data/randomSampleData/epiTCR/withMHC/dataTest200000.csv', index=False)
data_test_mhc_04.to_csv('../../data/randomSampleData/epiTCR/withMHC/dataTest500000.csv', index=False)

### pMTnet


test01_pmtnet = pd.read_csv("../../data/runTimeData/pMTnet/test01.csv")
test02_pmtnet = pd.read_csv("../../data/runTimeData/pMTnet/test02.csv")
test03_pmtnet = pd.read_csv("../../data/runTimeData/pMTnet/test03.csv")
test04_pmtnet = pd.read_csv("../../data/runTimeData/pMTnet/test04.csv")
test05_pmtnet = pd.read_csv("../../data/runTimeData/pMTnet/test05.csv")
test06_pmtnet = pd.read_csv("../../data/runTimeData/pMTnet/test06.csv")
test07_pmtnet = pd.read_csv("../../data/runTimeData/pMTnet/test07.csv")
test08_pmtnet = pd.read_csv("../../data/runTimeData/pMTnet/test08.csv")
test09_pmtnet = pd.read_csv("../../data/runTimeData/pMTnet/test09.csv")

y_data_pmtnet01 = test01_pmtnet.iloc[:, 3:]
y_data_pmtnet02 = test02_pmtnet.iloc[:, 3:]
y_data_pmtnet03 = test03_pmtnet.iloc[:, 3:]
y_data_pmtnet04 = test04_pmtnet.iloc[:, 3:]
y_data_pmtnet05 = test05_pmtnet.iloc[:, 3:]
y_data_pmtnet06 = test06_pmtnet.iloc[:, 3:]
y_data_pmtnet07 = test07_pmtnet.iloc[:, 3:]
y_data_pmtnet08 = test08_pmtnet.iloc[:, 3:]
y_data_pmtnet09 = test09_pmtnet.iloc[:, 3:]

data_test_pmtnet01 = pd.read_csv("../../data/runTimeData/pMTnet/normalizeData/test01.csv")
data_test_pmtnet02 = pd.read_csv("../../data/runTimeData/pMTnet/normalizeData/test02.csv")
data_test_pmtnet03 = pd.read_csv("../../data/runTimeData/pMTnet/normalizeData/test03.csv")
data_test_pmtnet04 = pd.read_csv("../../data/runTimeData/pMTnet/normalizeData/test04.csv")
data_test_pmtnet05 = pd.read_csv("../../data/runTimeData/pMTnet/normalizeData/test05.csv")
data_test_pmtnet06 = pd.read_csv("../../data/runTimeData/pMTnet/normalizeData/test06.csv")
data_test_pmtnet07 = pd.read_csv("../../data/runTimeData/pMTnet/normalizeData/test07.csv")
data_test_pmtnet08 = pd.read_csv("../../data/runTimeData/pMTnet/normalizeData/test08.csv")
data_test_pmtnet09 = pd.read_csv("../../data/runTimeData/pMTnet/normalizeData/test09.csv")

test_pMTnet01 = pd.concat([data_test_pmtnet01, y_data_pmtnet01], axis=1)
test_pMTnet02 = pd.concat([data_test_pmtnet02, y_data_pmtnet02], axis=1)
test_pMTnet03 = pd.concat([data_test_pmtnet03, y_data_pmtnet03], axis=1)
test_pMTnet04 = pd.concat([data_test_pmtnet04, y_data_pmtnet04], axis=1)
test_pMTnet05 = pd.concat([data_test_pmtnet05, y_data_pmtnet05], axis=1)
test_pMTnet06 = pd.concat([data_test_pmtnet06, y_data_pmtnet06], axis=1)
test_pMTnet07 = pd.concat([data_test_pmtnet07, y_data_pmtnet07], axis=1)
test_pMTnet08 = pd.concat([data_test_pmtnet08, y_data_pmtnet08], axis=1)
test_pMTnet09 = pd.concat([data_test_pmtnet09, y_data_pmtnet09], axis=1)

data_test_pmtnet = pd.concat([test_pMTnet01,test_pMTnet02,test_pMTnet03,test_pMTnet04,
                           test_pMTnet05,test_pMTnet06,test_pMTnet07,test_pMTnet08,test_pMTnet09], axis=0)
data_test_pmtnet = data_test_pmtnet.reset_index(drop=True)


data_test_pmtnet_01 = data_test_pmtnet.sample(n=10000, random_state=42)
data_test_pmtnet_01 = data_test_pmtnet_01.reset_index(drop=True)

data_test_pmtnet_02 = data_test_pmtnet.sample(n=50000, random_state=42)
data_test_pmtnet_02 = data_test_pmtnet_02.reset_index(drop=True)

data_test_pmtnet_03 = data_test_pmtnet.sample(n=200000, random_state=42)
data_test_pmtnet_03 = data_test_pmtnet_03.reset_index(drop=True)

data_test_pmtnet_04 = data_test_pmtnet.sample(n=500000, random_state=42)
data_test_pmtnet_04 = data_test_pmtnet_04.reset_index(drop=True)

X_data_test_pmtnet_01 = data_test_pmtnet_01.iloc[:, :3]
X_data_test_pmtnet_02 = data_test_pmtnet_02.iloc[:, :3]
X_data_test_pmtnet_03 = data_test_pmtnet_03.iloc[:, :3]
X_data_test_pmtnet_04 = data_test_pmtnet_04.iloc[:, :3]

X_data_test_pmtnet_01.to_csv('../../data/randomSampleData/pMTnet/dataTestpMTnet01.csv', index=False)
X_data_test_pmtnet_02.to_csv('../../data/randomSampleData/pMTnet/dataTestpMTnet02.csv', index=False)
X_data_test_pmtnet_03.to_csv('../../data/randomSampleData/pMTnet/dataTestpMTnet03.csv', index=False)
X_data_test_pmtnet_04.to_csv('../../data/randomSampleData/pMTnet/dataTestpMTnet04.csv', index=False)

### Imrex

imrex_test01 = pd.read_csv("../../data/runTimeData/Imrex/ptest01.csv")
imrex_test02 = pd.read_csv("../../data/runTimeData/Imrex/ptest02.csv")
imrex_test03 = pd.read_csv("../../data/runTimeData/Imrex/ptest03.csv")
imrex_test04 = pd.read_csv("../../data/runTimeData/Imrex/ptest04.csv")
imrex_test05 = pd.read_csv("../../data/runTimeData/Imrex/ptest05.csv")
imrex_test06 = pd.read_csv("../../data/runTimeData/Imrex/ptest06.csv")
imrex_test07 = pd.read_csv("../../data/runTimeData/Imrex/ptest07.csv")
imrex_test08 = pd.read_csv("../../data/runTimeData/Imrex/ptest08.csv")
imrex_test09 = pd.read_csv("../../data/runTimeData/Imrex/ptest09.csv")
imrex_test10 = pd.read_csv("../../data/runTimeData/Imrex/ptest10.csv")
imrex_test11 = pd.read_csv("../../data/runTimeData/Imrex/ptest11.csv")
imrex_test12 = pd.read_csv("../../data/runTimeData/Imrex/ptest12.csv")
imrex_test13 = pd.read_csv("../../data/runTimeData/Imrex/ptest13.csv")
imrex_test14 = pd.read_csv("../../data/runTimeData/Imrex/ptest14.csv")
imrex_test15 = pd.read_csv("../../data/runTimeData/Imrex/ptest15.csv")


imrex_data_test = pd.concat([imrex_test01,imrex_test02,imrex_test03,imrex_test04,
                             imrex_test05,imrex_test06,imrex_test07,imrex_test08,
                             imrex_test09,imrex_test10,imrex_test11,imrex_test12,
                             imrex_test13,imrex_test14,imrex_test15], axis=0)
imrex_data_test = imrex_data_test.reset_index(drop=True)

imrex_data_test_01 = imrex_data_test.sample(n=10000, random_state=42)
imrex_data_test_01 = imrex_data_test_01.reset_index(drop=True)

imrex_data_test_02 = imrex_data_test.sample(n=50000, random_state=42)
imrex_data_test_02 = imrex_data_test_02.reset_index(drop=True)

imrex_data_test_03 = imrex_data_test.sample(n=200000, random_state=42)
imrex_data_test_03 = imrex_data_test_03.reset_index(drop=True)

imrex_data_test_04 = imrex_data_test.sample(n=500000, random_state=42)
imrex_data_test_04 = imrex_data_test_04.reset_index(drop=True)

imrex_data_test_05 = imrex_data_test.sample(n=1000000, random_state=42)
imrex_data_test_05 = imrex_data_test_05.reset_index(drop=True)

imrex_data_test_01.to_csv('../../data/randomSampleData/Imrex/dataTestImrex10000.csv', index=False)
imrex_data_test_02.to_csv('../../data/randomSampleData/Imrex/dataTestImrex50000.csv', index=False)
imrex_data_test_03.to_csv('../../data/randomSampleData/Imrex/dataTestImrex200000.csv', index=False)
imrex_data_test_04.to_csv('../../data/randomSampleData/Imrex/dataTestImrex500000.csv', index=False)
imrex_data_test_05.to_csv('../../data/randomSampleData/Imrex/dataTestImrex1000000.csv', index=False)

### NetTCR

nettcr_test01 = pd.read_csv("../../data/runTimeData/NetTCR/test01.csv")
nettcr_test02 = pd.read_csv("../../data/runTimeData/NetTCR/test02.csv")
nettcr_test03 = pd.read_csv("../../data/runTimeData/NetTCR/test03.csv")
nettcr_test04 = pd.read_csv("../../data/runTimeData/NetTCR/test04.csv")
nettcr_test05 = pd.read_csv("../../data/runTimeData/NetTCR/test05.csv")
nettcr_test06 = pd.read_csv("../../data/runTimeData/NetTCR/test06.csv")
nettcr_test07 = pd.read_csv("../../data/runTimeData/NetTCR/test07.csv")
nettcr_test08 = pd.read_csv("../../data/runTimeData/NetTCR/test08.csv")
nettcr_test09 = pd.read_csv("../../data/runTimeData/NetTCR/test09.csv")
nettcr_test10 = pd.read_csv("../../data/runTimeData/NetTCR/test10.csv")
nettcr_test11 = pd.read_csv("../../data/runTimeData/NetTCR/test11.csv")
nettcr_test12 = pd.read_csv("../../data/runTimeData/NetTCR/test12.csv")
nettcr_test13 = pd.read_csv("../../data/runTimeData/NetTCR/test13.csv")
nettcr_test14 = pd.read_csv("../../data/runTimeData/NetTCR/test14.csv")
nettcr_test15 = pd.read_csv("../../data/runTimeData/NetTCR/test15.csv")

nettcr_test02.rename(columns = {'epitope':'peptide'}, inplace = True)
nettcr_test03.rename(columns = {'epitope':'peptide'}, inplace = True)
nettcr_test04.rename(columns = {'epitope':'peptide'}, inplace = True)
nettcr_test05.rename(columns = {'epitope':'peptide'}, inplace = True)
nettcr_test06.rename(columns = {'epitope':'peptide'}, inplace = True)
nettcr_test07.rename(columns = {'epitope':'peptide'}, inplace = True)
nettcr_test08.rename(columns = {'epitope':'peptide'}, inplace = True)
nettcr_test09.rename(columns = {'epitope':'peptide'}, inplace = True)
nettcr_test10.rename(columns = {'epitope':'peptide'}, inplace = True)
nettcr_test11.rename(columns = {'epitope':'peptide'}, inplace = True)
nettcr_test12.rename(columns = {'epitope':'peptide'}, inplace = True)
nettcr_test13.rename(columns = {'epitope':'peptide'}, inplace = True)
nettcr_test14.rename(columns = {'epitope':'peptide'}, inplace = True)
nettcr_test15.rename(columns = {'epitope':'peptide'}, inplace = True)

nettcr_test02 = nettcr_test02.loc[ :, ["CDR3b","peptide"]]
nettcr_test03 = nettcr_test03.loc[ :, ["CDR3b","peptide"]]
nettcr_test04 = nettcr_test04.loc[ :, ["CDR3b","peptide"]]
nettcr_test05 = nettcr_test05.loc[ :, ["CDR3b","peptide"]]
nettcr_test06 = nettcr_test06.loc[ :, ["CDR3b","peptide"]]
nettcr_test07 = nettcr_test07.loc[ :, ["CDR3b","peptide"]]
nettcr_test08 = nettcr_test08.loc[ :, ["CDR3b","peptide"]]
nettcr_test09 = nettcr_test09.loc[ :, ["CDR3b","peptide"]]
nettcr_test10 = nettcr_test10.loc[ :, ["CDR3b","peptide"]]
nettcr_test11 = nettcr_test11.loc[ :, ["CDR3b","peptide"]]
nettcr_test12 = nettcr_test12.loc[ :, ["CDR3b","peptide"]]
nettcr_test13 = nettcr_test13.loc[ :, ["CDR3b","peptide"]]
nettcr_test14 = nettcr_test14.loc[ :, ["CDR3b","peptide"]]
nettcr_test15 = nettcr_test15.loc[ :, ["CDR3b","peptide"]]


nettcr_data_test = pd.concat([nettcr_test01,nettcr_test02,nettcr_test03,
                              nettcr_test04,nettcr_test05,nettcr_test06,
                              nettcr_test07,nettcr_test08,nettcr_test09,
                              nettcr_test10,nettcr_test11,nettcr_test12,
                              nettcr_test13,nettcr_test14,nettcr_test15], axis=0)
nettcr_data_test = nettcr_data_test.reset_index(drop=True)

nettcr_data_test_01 = nettcr_data_test.sample(n=10000, random_state=42)
nettcr_data_test_01 = nettcr_data_test_01.reset_index(drop=True)

nettcr_data_test_02 = nettcr_data_test.sample(n=50000, random_state=42)
nettcr_data_test_02 = nettcr_data_test_02.reset_index(drop=True)

nettcr_data_test_03 = nettcr_data_test.sample(n=200000, random_state=42)
nettcr_data_test_03 = nettcr_data_test_03.reset_index(drop=True)

nettcr_data_test_04 = nettcr_data_test.sample(n=500000, random_state=42)
nettcr_data_test_04 = nettcr_data_test_04.reset_index(drop=True)

nettcr_data_test_05 = nettcr_data_test.sample(n=1000000, random_state=42)
nettcr_data_test_05 = nettcr_data_test_05.reset_index(drop=True)


nettcr_data_test_01.to_csv('../../data/randomSampleData/NetTCR/dataTestNetTCR_10000.csv', index=False)
nettcr_data_test_02.to_csv('../../data/randomSampleData/NetTCR/dataTestNetTCR_50000.csv', index=False)
nettcr_data_test_03.to_csv('../../data/randomSampleData/NetTCR/dataTestNetTCR_200000.csv', index=False)
nettcr_data_test_04.to_csv('../../data/randomSampleData/NetTCR/dataTestNetTCR_500000.csv', index=False)
nettcr_data_test_05.to_csv('../../data/randomSampleData/NetTCR/dataTestNetTCR_1000000.csv', index=False)


## Benchmark Runing time


tmp_dataset = pd.DataFrame(([["  10000",  18.817,   7.649,   2.662,  2489.677,   1.000,    1.431], 
                             ["  50000",  40.415,  17.154,  10.220, 12295.409,   4.509,    7.159], 
                             [" 200000", 132.958,  40.154,  22.604, 51264.141,  17.975,   27.638], 
                             [" 500000", 330.146,  94.591,  54.236,132951.125,  48.843,   70.597], 
                             ["1000000", 660.439, 179.598, 101.045,262892.017,  94.023,  142.194]]), 
                             columns=['count_sample', 'ATM-TCR', 'Imrex', 'NetTCR',
                                                      'pMTnet', 'epiTCR','epiTCR_mhc'])

tmp_dataset["ATM-TCR_log"] = np.log10(tmp_dataset["ATM-TCR"])
tmp_dataset["Imrex_log"] = np.log10(tmp_dataset["Imrex"])
tmp_dataset["NetTCR_log"] = np.log10(tmp_dataset["NetTCR"])
tmp_dataset["pMTnet_log"] = np.log10(tmp_dataset["pMTnet"])
tmp_dataset["epiTCR_log"] = np.log10(tmp_dataset["epiTCR"])
tmp_dataset["epiTCR_mhc_log"] = np.log10(tmp_dataset["epiTCR_mhc"])

plt.plot(tmp_dataset['count_sample'], tmp_dataset['ATM-TCR_log'],  label='ATM-TCR', marker='o')
plt.plot(tmp_dataset['count_sample'], tmp_dataset['Imrex_log'], label='Imrex', marker='o')
plt.plot(tmp_dataset['count_sample'], tmp_dataset['NetTCR_log'],  label='NetTCR', marker='o')
plt.plot(tmp_dataset['count_sample'], tmp_dataset['epiTCR_log'], label='epiTCR', marker='o')

plt.xlabel('Dataset size')
plt.ylabel('Runtime (log10(s))')
plt.legend()
plt.savefig("../../analysis/figures/benchmarkRunTime.png")
plt.savefig("../../analysis/figures/benchmarkRunTime.pdf")
plt.show()

import matplotlib.pyplot as plt

# Plot a simple line chart
plt.plot(tmp_dataset['count_sample'], tmp_dataset['pMTnet_log'],   color='g', label='pMTnet', marker='o')
plt.plot(tmp_dataset['count_sample'], tmp_dataset['epiTCR_mhc_log'], color='r', label='epiTCR', marker='o')

plt.xlabel('Dataset size')
plt.ylabel('Runtime (log10(s))')
plt.legend()
plt.savefig('../../analysis/figures/benchmarkRunTimeWithMHC.png')
plt.savefig('../../analysis/figures/benchmarkRunTimeWithMHC.pdf')
plt.show()
