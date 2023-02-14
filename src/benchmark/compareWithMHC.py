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

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (16,8)
fig,ax = plt.subplots(1)

mlib_acc    =  [0.89,0.88,0.89,0.89,0.88,0.89,0.88,0.89,0.88]
mlib_sens   =  [0.94,0.95,0.94,0.94,0.94,0.94,0.95,0.94,0.94]
mlib_spec   =  [0.88,0.88,0.88,0.88,0.89,0.88,0.88,0.88,0.87]
mlib_auc    =  [0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97]

pmtnet_acc  =  [0.89,0.89,0.89,0.89,0.89,0.88,0.89,0.89,0.89]
pmtnet_sens =  [0.03,0.03,0.04,0.04,0.04,0.03,0.03,0.03,0.03]
pmtnet_spec =  [0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98]
pmtnet_auc  =  [0.51,0.52,0.52,0.52,0.52,0.52,0.52,0.51,0.51]

algo = ["Testset 01","Testset 02","Testset 03","Testset 04","Testset 05",
        "Testset 06","Testset 07","Testset 08","Testset 09"]

df_002_acc = pd.DataFrame({"algo":algo,"epiTCR":mlib_acc, "pMTnet":pmtnet_acc})
df_002_acc = pd.melt(df_002_acc, id_vars="algo")

df_002_sens = pd.DataFrame({"algo":algo,"epiTCR":mlib_sens, "pMTnet":pmtnet_sens})
df_002_sens = pd.melt(df_002_sens, id_vars="algo")

df_002_spec = pd.DataFrame({"algo":algo,"epiTCR":mlib_spec, "pMTnet":pmtnet_spec})
df_002_spec = pd.melt(df_002_spec, id_vars="algo")

df_002_auc = pd.DataFrame({"algo":algo,"epiTCR":mlib_auc, "pMTnet":pmtnet_auc})
df_002_auc = pd.melt(df_002_auc, id_vars="algo")

splot = sns.barplot(data=df_002_acc, x="algo", y="value", hue="variable")
plt.legend(loc='best')

for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 10), size=15,
                   textcoords = 'offset points')
plt.ylim(0.0, 1.10)
ax.set_xticklabels([])
plt.xlabel('Test sets', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='16') 
plt.setp(ax.get_legend().get_title(), fontsize='16') 
plt.savefig("../../analysis/figures/comparisonWithmhcAcc002.png")
plt.savefig("../../analysis/figures/comparisonWithmhcAcc002.pdf")
plt.show()


plt.rcParams["figure.figsize"] = (16,8)
fig,ax = plt.subplots(1)

splot = sns.barplot(data=df_002_sens, x="algo", y="value", hue="variable")
plt.legend(loc='best')

for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 10), size=15,
                   textcoords = 'offset points')
plt.ylim(0.0, 1.15)
ax.set_xticklabels([])
plt.xlabel('Test sets', fontsize=20)
plt.ylabel('Sensitivity', fontsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='16') 
plt.setp(ax.get_legend().get_title(), fontsize='16') 
plt.savefig("../../analysis/figures/comparisonWithmhcSens002.png")
plt.savefig("../../analysis/figures/comparison_between_epitcr_pmtnet_withmhc_sens_002.pdf")
plt.show()


plt.rcParams["figure.figsize"] = (16,8)
fig,ax = plt.subplots(1)

splot = sns.barplot(data=df_002_spec, x="algo", y="value", hue="variable")
plt.legend(loc='best')

for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 10), size=15,
                   textcoords = 'offset points')
plt.ylim(0.0, 1.19)
ax.set_xticklabels([])
plt.xlabel('Test sets', fontsize=20)
plt.ylabel('Specificity', fontsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='16') 
plt.setp(ax.get_legend().get_title(), fontsize='16') 
plt.savefig("../../analysis/figures/comparisonWithmhcSpec002.png")
plt.savefig("../../analysis/figures/comparisonWithmhcSpec002.pdf")
plt.show()


plt.rcParams["figure.figsize"] = (16,8)
fig,ax = plt.subplots(1)

splot = sns.barplot(data=df_002_auc, x="algo", y="value", hue="variable")
plt.legend(loc='best')

for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 10), size=15,
                   textcoords = 'offset points')
plt.ylim(0.0, 1.19)
ax.set_xticklabels([])
plt.xlabel('Test sets', fontsize=20)
plt.ylabel('AUC', fontsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='16') 
plt.setp(ax.get_legend().get_title(), fontsize='16') 
plt.savefig("../../analysis/figures/comparisonWithmhcAUC002.png")
plt.savefig("../../analysis/figures/comparisonWithmhcAUC002.pdf")
plt.show()

####----------


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16,8)
fig,ax = plt.subplots(1)

mlib_acc    = [0.89,0.88,0.89,0.89,0.88,0.89,0.88,0.89,0.88]
mlib_sens   = [0.94,0.95,0.94,0.94,0.94,0.94,0.95,0.94,0.94]
mlib_spec   = [0.88,0.88,0.88,0.88,0.89,0.88,0.88,0.88,0.87]
mlib_auc    = [0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97]
pmtnet_auc  = [0.51,0.52,0.52,0.52,0.52,0.52,0.52,0.51,0.51]
pmtnet_acc  = [0.86,0.86,0.86,0.86,0.86,0.86,0.86,0.86,0.86]
pmtnet_sens = [0.08,0.08,0.08,0.09,0.08,0.07,0.08,0.07,0.07]
pmtnet_spec = [0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94]

algo = ["Testset 01","Testset 02","Testset 03","Testset 04","Testset 05",
        "Testset 06","Testset 07","Testset 08","Testset 09"]

df_005_acc = pd.DataFrame({"algo":algo,"epiTCR":mlib_acc, "pMTnet":pmtnet_acc})
df_005_acc = pd.melt(df_005_acc, id_vars="algo")

df_005_sens = pd.DataFrame({"algo":algo,"epiTCR":mlib_sens, "pMTnet":pmtnet_sens})
df_005_sens = pd.melt(df_005_sens, id_vars="algo")

df_005_spec = pd.DataFrame({"algo":algo,"epiTCR":mlib_spec, "pMTnet":pmtnet_spec})
df_005_spec = pd.melt(df_005_spec, id_vars="algo")

df_005_auc = pd.DataFrame({"algo":algo,"epiTCR":mlib_auc, "pMTnet":pmtnet_auc})
df_005_auc = pd.melt(df_005_auc, id_vars="algo")

splot = sns.barplot(data=df_005_acc, x="algo", y="value", hue="variable")
plt.legend(loc='best')

for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 10), size=15,
                   textcoords = 'offset points')
plt.ylim(0.0, 1.10)
ax.set_xticklabels([])
plt.xlabel('Test sets', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='16') 
plt.setp(ax.get_legend().get_title(), fontsize='16') 
plt.savefig("../../analysis/figures/comparisonWithmhcAcc005.png")
plt.savefig("../../analysis/figures/comparisonWithmhcAcc005.pdf")
plt.show()


plt.rcParams["figure.figsize"] = (16,8)
fig,ax = plt.subplots(1)

splot = sns.barplot(data=df_005_sens, x="algo", y="value", hue="variable")
plt.legend(loc='best')

for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 10), size=15,
                   textcoords = 'offset points')
plt.ylim(0.0, 1.14)
ax.set_xticklabels([])
plt.xlabel('Test sets', fontsize=20)
plt.ylabel('Sensitivity', fontsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='16') 
plt.setp(ax.get_legend().get_title(), fontsize='16') 
plt.savefig("../../analysis/figures/comparisonWithmhcSens005.png")
plt.savefig("../../analysis/figures/comparisonWithmhcSens005.pdf")
plt.show()


plt.rcParams["figure.figsize"] = (16,8)
fig,ax = plt.subplots(1)

splot = sns.barplot(data=df_005_spec, x="algo", y="value", hue="variable")
plt.legend(loc='best')

for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 10), size=15,
                   textcoords = 'offset points')
plt.ylim(0.0, 1.14)
ax.set_xticklabels([])
plt.xlabel('Test sets', fontsize=20)
plt.ylabel('Specificity', fontsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='16') 
plt.setp(ax.get_legend().get_title(), fontsize='16') 
plt.savefig("../../analysis/figures/comparisonWithmhcSpec005.png")
plt.savefig("../../analysis/figures/comparisonWithmhcSpec005.pdf")
plt.show()


plt.rcParams["figure.figsize"] = (16,8)
fig,ax = plt.subplots(1)

splot = sns.barplot(data=df_005_auc, x="algo", y="value", hue="variable")
plt.legend(loc='best')

for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 10), size=15,
                   textcoords = 'offset points')
plt.ylim(0.0, 1.19)
ax.set_xticklabels([])
plt.xlabel('Test sets', fontsize=20)
plt.ylabel('AUC', fontsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='16') 
plt.setp(ax.get_legend().get_title(), fontsize='16') 
plt.savefig("../../analysis/figures/comparisonWithmhcAUC005.png")
plt.savefig("../../analysis/figures/comparisonWithmhcAUC005.pdf")
plt.show()


####-----------------


# import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16,8)
fig,ax = plt.subplots(1)

mlib_acc    = [0.89,0.88,0.89,0.89,0.88,0.89,0.88,0.89,0.88]
mlib_sens   = [0.94,0.95,0.94,0.94,0.94,0.94,0.95,0.94,0.94]
mlib_spec   = [0.88,0.88,0.88,0.88,0.89,0.88,0.88,0.88,0.87]
mlib_auc    = [0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97]
pmtnet_acc  = [0.82,0.82,0.82,0.82,0.82,0.82,0.82,0.82,0.82]
pmtnet_sens = [0.15,0.14,0.14,0.15,0.14,0.13,0.13,0.13,0.13]
pmtnet_spec = [0.88,0.88,0.88,0.88,0.88,0.88,0.88,0.88,0.88]
pmtnet_auc  = [0.51,0.52,0.52,0.52,0.52,0.52,0.52,0.51,0.51]

algo = ["Testset 01","Testset 02","Testset 03","Testset 04","Testset 05",
        "Testset 06","Testset 07","Testset 08","Testset 09"]

df_01_acc = pd.DataFrame({"algo":algo,"epiTCR":mlib_acc, "pMTnet":pmtnet_acc})
df_01_acc = pd.melt(df_01_acc, id_vars="algo")

df_01_sens = pd.DataFrame({"algo":algo,"epiTCR":mlib_sens, "pMTnet":pmtnet_sens})
df_01_sens = pd.melt(df_01_sens, id_vars="algo")

df_01_spec = pd.DataFrame({"algo":algo,"epiTCR":mlib_spec, "pMTnet":pmtnet_spec})
df_01_spec = pd.melt(df_01_spec, id_vars="algo")

df_01_auc = pd.DataFrame({"algo":algo,"epiTCR":mlib_auc, "pMTnet":pmtnet_auc})
df_01_auc = pd.melt(df_01_auc, id_vars="algo")

splot = sns.barplot(data=df_01_acc, x="algo", y="value", hue="variable")
plt.legend(loc='best')

for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 10), size=15,
                   textcoords = 'offset points')
plt.ylim(0.0, 1.10)
ax.set_xticklabels([])
plt.xlabel('Test sets', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='16') 
plt.setp(ax.get_legend().get_title(), fontsize='16') 
plt.savefig("../../analysis/figures/comparisonWithmhcAUC01.png")
plt.savefig("../../analysis/figures/comparisonWithmhcAUC01.pdf")
plt.show()


plt.rcParams["figure.figsize"] = (16,8)
fig,ax = plt.subplots(1)

splot = sns.barplot(data=df_01_sens, x="algo", y="value", hue="variable")
plt.legend(loc='best')

for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 10), size=15,
                   textcoords = 'offset points')
plt.ylim(0.0, 1.14)
ax.set_xticklabels([])
plt.xlabel('Test sets', fontsize=20)
plt.ylabel('Sensitivity', fontsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='16') 
plt.setp(ax.get_legend().get_title(), fontsize='16') 
plt.savefig("../../analysis/figures/comparisonWithmhcSens01.png")
plt.savefig("../../analysis/figures/comparisonWithmhcSens01.pdf")
plt.show()


plt.rcParams["figure.figsize"] = (16,8)
fig,ax = plt.subplots(1)

splot = sns.barplot(data=df_01_spec, x="algo", y="value", hue="variable")
plt.legend(loc='best')

for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 10), size=15,
                   textcoords = 'offset points')
plt.ylim(0.0, 1.10)
ax.set_xticklabels([])
plt.xlabel('Test sets', fontsize=20)
plt.ylabel('Specificity', fontsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='16') 
plt.setp(ax.get_legend().get_title(), fontsize='16') 
plt.savefig("../../analysis/figures/comparisonWithmhcSpec01.png")
plt.savefig("../../analysis/figures/comparisonWithmhcSpec01.pdf")
plt.show()


plt.rcParams["figure.figsize"] = (16,8)
fig,ax = plt.subplots(1)

splot = sns.barplot(data=df_01_auc, x="algo", y="value", hue="variable")
plt.legend(loc='best')

for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 10), size=15,
                   textcoords = 'offset points')
plt.ylim(0.0, 1.19)
ax.set_xticklabels([])
plt.xlabel('Test sets', fontsize=20)
plt.ylabel('AUC', fontsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='16') 
plt.setp(ax.get_legend().get_title(), fontsize='16') 
plt.savefig("../../analysis/figures/comparisonWithmhcAUC01.png")
plt.savefig("../../analysis/figures/comparisonWithmhcAUC01.pdf")
plt.show()