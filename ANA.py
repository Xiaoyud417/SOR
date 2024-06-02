#basic packages
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import  combinations
from warnings import filterwarnings

#packages for figure
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # for displaying Chinese Characters
plt.rcParams['axes.unicode_minus'] = False # for displaying negative sign

#packages for supervised machine learning
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn.model_selection import StratifiedKFold as SKF
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.neural_network import MLPClassifier

#packages for unsupervised machine learning
from sklearn.decomposition import PCA as PCAA
from pyclust import KMedoids
from sklearn.manifold import TSNE, MDS

#packages for ML classifiers interpretation and feature selection
from minepy import MINE #maximal information coefficient (MIC)
from sklearn.cross_decomposition import CCA #canonical correlation analysis (CCA)

#packages for ML models save and read
import pickle

#packages for statistical analysis
from lifelines import KaplanMeierFitter,WeibullAFTFitter
from lifelines.statistics import logrank_test,multivariate_logrank_test

filterwarnings('ignore')

#Hyperparameter settings for 12 ML classifiers
models = [('LR', LR(max_iter=10000, tol=0.1)), ('LDA', LDA(solver='lsqr', shrinkage='auto')),
          ('QDA', QDA(store_covariance=True)),
          ('ET', ETC(n_estimators=10, min_samples_split=2, max_features="sqrt")),
          ('GBC', GBC(n_estimators=1000, learning_rate=1.0, max_depth=4, max_features='auto')),
          ('ABC', ABC(n_estimators=400)),
          ('KNN', KN(n_neighbors=6)), ('CART', DTC()), ('NB', GNB()),
          ('RSVM',SVC(C=.7, class_weight='balanced', kernel='rbf', cache_size=300, gamma='auto', probability=True)),
          ('RFC', RFC(min_samples_split=2, n_estimators=10, max_features="sqrt")),
          ('MLP', MLPClassifier(random_state=1, max_iter=3000, hidden_layer_sizes=(100, 100), batch_size=100,
                                solver='adam', warm_start=False, early_stopping=True, n_iter_no_change=100,
                                validation_fraction=0.2))]

# transform to a dictionary
dmodels=dict(zip([i[0] for i in models],[i[-1] for i in models]))

#parameters for MIC analysis
mine = MINE(alpha=0.6, c=15)

#Function for MIC value computation
def MIC_matirx0(dataframe, mine,nvar=['OC_OPC'], nf=['PCA_FCR_3_0.4_20']):
    # dataframe refers to the dataframe at least containing features in nvar and nf
    # mine refers to the function used for MIC value computation
    # nvar controls the features involved in MIC analysis in one dimension, here refers to disease information
    # nf controls the features involved in MIC analysis in the other dimension, here refers to PCA-transformed features
    result = np.zeros([len(nf), len(nvar)])
    for k, i in tqdm(enumerate(nf)):
        for kk, j in enumerate(nvar):
            mine.compute_score(dataframe[i].to_numpy(), dataframe[j].to_numpy())
            result[k, kk] = mine.mic()
        time.sleep
    RT = pd.DataFrame(result, index=nf, columns=nvar)
    return RT

std=StandardScaler()
mms = MinMaxScaler()
pca = PCAA()
cca = CCA(n_components=2, max_iter=300)

#function for cca score computation and feature selection via CCA methods
def CCA_selection(data, data1,bcd, set2, savepath,step=1,ratio=0.2):
    #data refers to the dataframe containing all feature values of training cohort
    #data1 refers to the dataframe containing all feature values of test cohort
    #bcd refers to the dataframe of binary coding disease states
    #set2 refers to the initial feature set
    #step controls the number of features deleted per iterative down sampling
    #ratio determines the anticipated percent of selected features
    #savepath refers to the absolute path for saving the selection procedure. This will help a lot in case of program crash.
    ppan=pd.concat([data,data1])
    std_ppan = pd.DataFrame(std.fit_transform(ppan), index=ppan.index.to_list(), columns=ppan.columns.to_list())
    std_code = pd.DataFrame(std.fit_transform(bcd), index=bcd.index.to_list(), columns=bcd.columns.to_list())
    set1=std_code.columns.to_list()
    pppan = pd.concat([std_code, std_ppan], axis=1)
    Data = pppan.loc[data.index.to_list()]
    Data1 = pppan.loc[data1.index.to_list()]
    pset = pd.DataFrame(columns=['CCA_coef'])
    pcoef = pd.DataFrame(columns=['CCA_coef'])
    original_num=len(set2)
    n = 1
    while n < (1-ratio) * original_num:
        for j in tqdm(list(combinations(set2, len(set2) - 1))):
            cca.fit(Data[set1].to_numpy(), Data[list(j)].to_numpy())
            coe = cca.score(X=Data1[set1].to_numpy(), y=Data1[list(j)].to_numpy())
            pset.loc['+'.join(j)] = coe
            time.sleep
        pset.sort_values('CCA_coef', inplace=True, ascending=False)
        #Sorting keeps the most related feature sets at the very first row. 
        # When down sampling don't improve the CCA score, the left iterations will be redundant.
        #Therefore, the final selected features are controlled by ratio and CCA score together.
        set2 = pset.index.to_list()[0].split('+')
        with open(savepath, 'a+') as f:
            f.write(str(len(set2)) + ' ' + str(pset.iloc[0, 0]) + ' ' + pset.index.to_list()[0] + '\n')
        pset = pset.iloc[:1, :1]
        pcoef.loc[len(set2)] = pset.iloc[0, 0]
        n += 1
    return pset,pcoef

#function for ML training without any data balancing strategy
score_set = ('accuracy', 'precision', 'average_precision', 'recall', 'f1', 'roc_auc')

def MLSS(data, data1,outpath,cols=[], var=['OC_OPC'], models=models, shuffle=True, model_output=True,
         model_name='OC_OPC',prefix=''):
    # data refers to the dataframe contains both feature values and disease information of training cohort
    # data1 refers to the dataframe contains both feature values and disease information of test cohort
    # cols controls the PCA-transformed acoustic feature sets used for machine learning
    # var refers to the disease label used for machine learning
    # models refers to 12 ML classifier which have been ready in terms of hyperparameter settings
    # model_output controls if save the trained ML classifiers
    # model_name and prefix allow for custom naming for saved model
    # val controls if figure output for ML classifier in terms of ROC curve (default) or precision-recall curve. 
    # one can name val='' to not to export any figures.
    # outpath controls where all outputs including models and figures to save
    data = data.astype({'Gender_no': 'category'})
    data1 = data1.astype({'Gender_no': 'category'})
    pdval = pd.DataFrame(columns=['Test_accuracy'])
    # this dataframe is used to record the test accuracy of each ML model in test cohorts
    pmodel_n = pd.DataFrame(columns=['CV_' + str(s) for s in range(1, 6)])
    # this dataframe is used to record the training metrics of each ML model in training cohorts
    # stratified 5-fold cross-validation is adopt in this study
    aval=data1[var]
    # this dataframe is used to record individual prediction outcome and probability
    for name, model in tqdm(models):
        for va in var:
            cv_results = model_selection.cross_validate(model, data[cols + ['Gender_no', 'Age']], data[va],
                                                        cv=SKF(n_splits=5, shuffle=shuffle), scoring=score_set)
            for sc in list(score_set):
                pmodel_n.loc[name + '_' + model_name + '_' + sc] = cv_results['test_' + sc]
            if model_output:
                model.fit(data[cols + ['Gender_no', 'Age']], data[va])
                with open(outpath + '\\' + prefix + '_' + name + '_' + model_name + '.pickle',
                          'wb') as f:
                    pickle.dump(model, f)
            for ix in data1.index.to_list():
                aval.loc[ix, 'Prediction_outcome_' + name + '_' + model_name] = model.predict(
                    data1.loc[ix][cols + ['Gender_no', 'Age']].to_numpy().reshape(1, -1)).flatten()
                aval.loc[ix, [p + '_' + name + '_' + model_name for p in ['PPP', 'NPP']]] = model.predict_proba(
                    data1.loc[ix][cols + ['Gender_no', 'Age']].to_numpy().reshape(1, -1)).flatten()
            X = data1[cols + ['Gender_no', 'Age']].to_numpy()
            y = data1[va].to_numpy()
            pdval.loc[name + "_" + model_name] = model.score(X, y)
        time.sleep
    return pmodel_n, pdval, aval

#package for SMOTE
from imblearn.over_sampling import BorderlineSMOTE
bsmote=BorderlineSMOTE(k_neighbors=5, random_state=2)

#function for ML training with SMOTE data balancing strategy
def MLSS_upo(data, data1, outpath, cols=[], var=['OC_OPC'], models=models, model_output=True,
             model_name='OC_OPC', prefix=''):
    # data refers to the dataframe contains both feature values and disease information of training cohort
    # data1 refers to the dataframe contains both feature values and disease information of test cohort
    # cols controls the PCA-transformed acoustic feature sets used for machine learning
    # var refers to the disease label used for machine learning
    # models refers to 12 ML classifier which have been ready in terms of hyperparameter settings
    # model_output controls if save the trained ML classifiers
    # model_name and prefix allow for custom naming for saved model
    # val controls if figure output for ML classifier in terms of ROC curve (default) or precision-recall curve.
    # one can name val='' to not to export any figures.
    # outpath controls where all outputs including models and figures to save
    X_res, y_res = bsmote.fit_resample(data[cols + ['Age', 'Gender_no']], data[var[0]])
    ocdata1 = pd.concat([X_res, y_res], axis=1)
    pdval = pd.DataFrame(columns=['Test_accuracy'])
    # this dataframe is used to record the test accuracy of each ML model in test cohorts
    pmodel_n = pd.DataFrame(columns=['CV_' + str(s) for s in range(1, 6)])
    # this dataframe is used to record the training metrics of each ML model in training cohorts
    # stratified 5-fold cross-validation is adopt in this study
    aval = data1[var]
    # this dataframe is used to record individual prediction outcome and probability
    for name, model in tqdm(models):
        for va in var:
            cv_results = model_selection.cross_validate(model, ocdata1[cols + ['Gender_no', 'Age']], ocdata1[va],
                                                        scoring=score_set)
            for sc in list(score_set):
                pmodel_n.loc[name + '_' + model_name + '_' + sc] = cv_results['test_' + sc]
            if model_output:
                model.fit(ocdata1[cols + ['Gender_no', 'Age']], ocdata1[va])
                with open(outpath + '\\' + prefix + '_' + name + '_' + model_name + '.pickle',
                          'wb') as f:
                    pickle.dump(model, f)
            for ix in data1.index.to_list():
                aval.loc[ix, 'Prediction_outcome_' + name + '_' + model_name] = model.predict(
                    data1.loc[ix][cols + ['Gender_no', 'Age']].to_numpy().reshape(1, -1)).flatten()
                aval.loc[ix, [p + '_' + name + '_' + model_name for p in ['PPP', 'NPP']]] = model.predict_proba(
                    data1.loc[ix][cols + ['Gender_no', 'Age']].to_numpy().reshape(1, -1)).flatten()
            X = data1[cols + ['Gender_no', 'Age']].to_numpy()
            y = data1[va].to_numpy()
            pdval.loc[name + "_" + model_name] = model.score(X, y)
        time.sleep
    return pmodel_n, pdval, aval

#function　for unsupervised learning using manifold learning and K-medoids clustering
def pam_cluster(phnc, pscc,savepath,lifepath,features_in,random_state=2):
    # phnc refers to the dataframe contains disease information of all OC and OPC patients
    # pscc refers to the dataframe contains disease information of all cancer patients whose pathology was squamous cell carcinoma (SCC).
    # savepath controls which directory all files export to 
    # lifepath refers to the directory containing files of disease variables and follow-up survival information 
    # features_in controls selected features used for manifold learning
    writer1 = pd.ExcelWriter(savepath + '\\TSNE+MDS_outcome.xlsx')
    ptdp=pd.read_csv(lifepath+'\\Progression_12.txt',index_col=0,sep='\t')
    ptdd=pd.read_csv(lifepath+'\\Death_52.txt',index_col=0,sep='\t')
    plogrank = pd.DataFrame(columns=['P_value'])
    for i in tqdm(range(2, 5)):
        for j in range(2, 4):
            for g, d in zip(['HNC', 'SCC'], [phnc, pscc]):
                data_std = std.fit_transform(d[features_in])
                td2 = MDS(n_components=j, max_iter=1000, random_state=random_state).fit_transform(data_std)
                k = KMedoids(n_clusters=i, distance='euclidean', max_iter=1000).fit_predict(td2)
                ptd2 = pd.DataFrame(td2, index=d.index.to_list(),
                                    columns=[g + '_ALL_MDS_' + str(x) for x in range(1, j + 1)])
                ptd2['Cluster_MDS'] = k+1
                td1 = TSNE(n_components=j, init="pca", learning_rate=250, random_state=random_state, perplexity=50, method='exact',
                           early_exaggeration=15).fit_transform(data_std)
                k = KMedoids(n_clusters=i, distance='euclidean', max_iter=1000).fit_predict(td1)
                ptd1 = pd.DataFrame(td1, index=d.index.to_list(),
                                    columns=[g + '_ALL_TSNE_' + str(x) for x in range(1, j + 1)])
                ptd1['Cluster_TSNE'] = k+1
                ptd12 = pd.concat([ptd1, ptd2, d], axis=1,join='inner')
                ptd_d = pd.concat([ptd12, ptdd], axis=1, join='inner')
                ptd_p = pd.concat([ptd12, ptdp], axis=1, join='inner')
                for x, y, title, dss in zip(['P', 'D'], [['PW', 'PM'], ['DW', 'DM']], ['Progression', 'Death'],
                                            [ptd_p, ptd_d]):
                    for yy in y:
                        T = dss[yy]
                        E = dss[x]
                        for m in ['TSNE', 'MDS']:
                            if i > 2:
                                lr = multivariate_logrank_test(T, dss['Cluster_' + m], E, alpha=.99)
                                plogrank.loc[str(i) + 'C_' + str(j) + 'D_ALL_' + m + '_' + x + '_' + yy[
                                    -1] + '_in_' + g] = lr.p_value
                                for co in list(combinations([cc for cc in range(1, i + 1)], 2)):
                                    lr = logrank_test(T[dss['Cluster_' + m] == co[0]],
                                                      T[dss['Cluster_' + m] == co[1]],
                                                      E[dss['Cluster_' + m] == co[0]],
                                                      E[dss['Cluster_' + m] == co[1]], alpha=.99)
                                    plogrank.loc[str(i) + 'C_' + str(j) + 'D_ALL_' + m + ''.join(
                                        [str(ccc) for ccc in co]) + '_' + x + '_' + yy[-1] + '_in_' + g] = lr.p_value
                            else:
                                group = (dss['Cluster_' + m] == 1)
                                lr = logrank_test(T[group], T[~group], E[group], E[~group], alpha=.99)
                                plogrank.loc[str(j) + 'C_' + str(j) + 'D_ALL_' + m + '_' + x + '_' + yy[
                                    -1] + '_in_' + g] = lr.p_value
                            ax = plt.subplot(111)
                            for ii in range(i):
                                group = (dss['Cluster_' + m] == ii + 1)
                                km = KaplanMeierFitter()
                                km.fit(T[group], event_observed=E[group], label="cluster {}".format(str(ii + 1)))
                                km.plot(ax=ax, color=sns.husl_palette(n_colors=i)[ii])
                            ax.set_title(title + ' in units of ' + dict(zip(['W', 'M'], ['week', 'month']))[yy[-1]],
                                         fontsize=20,
                                         fontweight='bold')
                            ax.set_xlabel(dict(zip(['W', 'M'], ['Weeks', 'Months']))[yy[-1]], fontsize=18,
                                          fontweight='bold')
                            plt.savefig(savepath+'\\Cluster' + str(i) + '_' + 'ALL_pam_in_' + g + '_' + str(
                                    j) + 'D_' + title + '_' + yy[-1] + '_KMP.svg', dpi=1200, bbox_inches='tight')
                            plt.close()
                ptd12.to_excel(writer1, g + '_D' + str(j) + '_C' + str(i))
                writer1.save()
        time.sleep
    plogrank.sort_values('P_value', inplace=True)
    plogrank.to_csv(savepath + '\\Logrank_pvalues.txt', sep='\t', encoding='utf_8_sig')
    return pd.ExcelFile(savepath + '\\TSNE+MDS_outcome.xlsx'),plogrank

#function for survival analysis including log rank and Weibull accelerated failure time model 
def survival_aft(phnc, pscc,path,lifepath,savepath,base_vars=['Age', 'Gender', 'CRP', 'SII', 'NLR', 'MLR']):
    # phnc refers to the dataframe contains disease information of all OC and OPC patients
    # pscc refers to the dataframe contains disease information of all cancer patients whose pathology was squamous cell carcinoma (SCC).
    # path refers to the excel file path of unsupervised learning outcome
    # lifepath refers to the directory containing files of disease variables and follow-up survival information
    # savepath controls which directory all files export to 
    # base_vars determines which disease variables are included as independent factors used for multivariate survival analysis.
    cds=pd.ExcelFile(path)
    ptd_p = pd.read_csv(lifepath + '\\Progression_12.txt', index_col=0, sep='\t')
    ptd_d = pd.read_csv(lifepath + '\\Death_52.txt', index_col=0, sep='\t')
    pd_waft_multi = pd.DataFrame(columns=['Bi_stage_p', 'P_value', 'AIC_partial', 'C_index', 'Log_LH'])
    for i in tqdm(range(2, 5)):
        for j in range(2, 4):
            for g, d in zip(['HNC', 'SCC'], [phnc.index.to_list(), pscc.index.to_list()]):
                ptd = cds.parse(sheet_name=g + '_D' + str(j) + '_C' + str(i), index_col=0)
                ptd['Bi_stage'] = np.asarray(
                    [0 if (ptd.loc[ix, 'Stage_3'] == 0 and ptd.loc[ix, 'Stage_4'] == 0) else 1 for ix in
                     ptd.index.to_list()])
                ptd = ptd.astype(dict(zip(['Gender', 'Bi_stage'], ['category'] * 2)))
                ptd1 = ptd1.astype(dict(zip(['Cluster_TSNE', 'Cluster_MDS'], [np.int] * 2)))
                ssa = base_vars+['Bi_stage']
                for m in ['TSNE', 'MDS']:
                    if i > 2:
                        ptd1 = pd.get_dummies(ptd1, columns=['Cluster_' + m], drop_first=True)
                        ssa += ['Cluster_' + m + '_' + str(c) for c in range(2, i + 1)]
                    else:
                        ssa += ['Cluster_' + m]
                    for x, y, title, dss in zip(['P', 'D'], [['PW', 'PM'], ['DW', 'DM']],
                                                ['Progression', 'Death'], [ptd_p, ptd_d]):
                        dss1 = ptd1.loc[[ix for ix in ptd1.index.to_list() if ix in dss.index.to_list()]]
                        for yy in y:
                            aft = WeibullAFTFitter()
                            aft.fit(dss1[[yy, x] + ssa], duration_col=yy, event_col=x)
                            f, ax = plt.subplots(figsize=(7, 4))
                            aft.plot(ax=ax)
                            plt.savefig(savepath + '\\' +m + '_AFT_multi_' + str(
                                    i) + 'C_' + str(j) + 'D_ALL_' + x + '_' + yy[
                                    -1] + '_in_' + g + '.svg',dpi=1200, bbox_inches='tight')
                            plt.close()
                            aft.summary.to_csv(savepath+'\\' + m + '_AFT_multi_' + str(
                                    i) + 'C_' + str(j) + 'D_ALL_' + x + '_' + yy[
                                    -1] + '_in_' + g + '.txt', sep='\t', encoding='utf_8_sig')
                            pd_waft_multi.loc[
                                m + '_' + str(i) + 'C_' + str(j) + 'D_ALL_' + x + '_' + yy[-1] + '_in_' + g] = [aft.summary.loc[('lambda_',
                                                                                                                       'Bi_stage'), 'p'],
                                                                                                                   aft.summary.loc[
                                                                                                                       [('lambda_',s)
                                                                                                                           for s in
                                                                                                                           ssa[7:]]][
                                                                                                                       'p'].to_list()] + [
                                                                                                                   aft.AIC_,
                                                                                                                   aft.concordance_index_,
                                                                                                                   aft.log_likelihood_]
        time.sleep
    pd_waft_multi['P_min'] = np.asarray([np.min(np.asarray(pd_waft_multi.loc[i, 'P_value'])) for i in pd_waft_multi.index.to_list()])
    pd_waft_multi.sort_values('P_min', inplace=True)
    pd_waft_multi.to_csv(savepath + '\\multi_aft_pvalues.txt', sep='\t',encoding='utf_8_sig')
    return pd_waft_multi
