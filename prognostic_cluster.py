import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import  combinations
from warnings import filterwarnings
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # for displaying Chinese Characters
plt.rcParams['axes.unicode_minus'] = False # for displaying negative sign

#packages for unsupervised machine learning
from sklearn.decomposition import PCA as PCAA
from pyclust import KMedoids
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler
std=StandardScaler()

#packages for ML models save and read
import pickle

#packages for survival analyses
from lifelines import KaplanMeierFitter,WeibullAFTFitter,LogLogisticAFTFitter,LogNormalAFTFitter
from lifelines.statistics import logrank_test,multivariate_logrank_test

filterwarnings('ignore')

#function for unsupervised learning using manifold learning and K-medoids clustering
def pam_cluster(phnc, pscc,savepath,lifepath,features_in,random_state=2,pic=True):
    # phnc refers to the dataframe contains disease information of all OC and OPC patients
    # pscc refers to the dataframe contains disease information of all cancer patients whose pathology was squamous cell carcinoma (SCC).
    # savepath controls which directory all files export to 
    # lifepath refers to the directory that stores files of disease variables and follow-up survival information 
    # features_in controls selected speech omics features used for manifold learning
    writer1 = pd.ExcelWriter(savepath + '\\TSNE+MDS_outcome.xlsx')
    ptdp=pd.read_csv(lifepath+'\\Progression_12.txt',index_col=0,sep='\t')
    ptdd=pd.read_csv(lifepath+'\\Death_52.txt',index_col=0,sep='\t')
    plogrank = pd.DataFrame(columns=['P_value'])
    for i in range(2, 5):
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
                            if pic:
                                ax = plt.subplot(111)
                                for ii in range(i):
                                    group = (dss['Cluster_' + m] == ii + 1)
                                    km = KaplanMeierFitter()
                                    km.fit(T[group], event_observed=E[group], label="cluster {}".format(str(ii + 1)))
                                    km.plot(ax=ax, color=sns.husl_palette(n_colors=i)[ii])
                                ax.set_title(title + ' in units of ' + dict(zip(['W', 'M'], ['week', 'month']))[yy[-1]],
                                            fontsize=20,fontweight='bold')
                                ax.set_xlabel(dict(zip(['W', 'M'], ['Weeks', 'Months']))[yy[-1]], fontsize=18,
                                            fontweight='bold')
                                plt.savefig(savepath+'\\Cluster' + str(i) + '_' + 'ALL_pam_in_' + g + '_' + str(
                                        j) + 'D_' + title + '_' + yy[-1] + '_KMP.svg', dpi=1200, bbox_inches='tight')
                                plt.close()
                ptd12.to_excel(writer1, g + '_D' + str(j) + '_C' + str(i))
                writer1._save()
    plogrank.sort_values('P_value', inplace=True)
    plogrank.to_csv(savepath + '\\Logrank_pvalues.txt', sep='\t', encoding='utf_8_sig')
    return pd.ExcelFile(savepath + '\\TSNE+MDS_outcome.xlsx'),plogrank

#function for survival analysis including log rank and Weibull accelerated failure time model 
def survival_aft(phnc, pscc,path,lifepath,savepath,base_vars=['Age', 'Gender_no', 'CRP', 'SII', 'NLR', 'MLR']):
    # phnc refers to the dataframe contains disease information of all OC and OPC patients
    # pscc refers to the dataframe contains disease information of all cancer patients whose pathology was squamous cell carcinoma (SCC).
    # path refers to the excel file path of unsupervised learning outcome
    # lifepath refers to the directory containing files of disease variables and follow-up survival information
    # savepath controls which directory all files export to 
    # base_vars determines which disease variables are included as independent factors used for multivariate survival analysis.
    cds=pd.ExcelFile(path)
    ptd_p = pd.read_csv(lifepath + '\\Progression_12.txt', index_col=0, sep='\t')
    ptd_p['PM']=ptd_p['PM']+0.01
    ptd_d = pd.read_csv(lifepath + '\\Death_52.txt', index_col=0, sep='\t')
    ptd_d['PM'] = ptd_d['PM'] + 0.01
    pd_aft_multi = pd.DataFrame(columns=['Distribution','Bi_stage_p', 'P_value', 'AIC_partial', 'C_index', 'Log_LH'])
    vl=len(base_vars)
    for i in range(2, 5):
        for j in range(2, 4):
            for g, d in zip(['HNC', 'SCC'], [phnc.index.to_list(), pscc.index.to_list()]):
                ptd = cds.parse(sheet_name=g + '_D' + str(j) + '_C' + str(i), index_col=0).loc[d]
                ptd['Bi_stage'] = np.asarray(
                    [0 if ptd.loc[ix, 'Stage'] < 3 else 1 for ix in
                     ptd.index.to_list()])
                ptd = ptd.astype(dict(zip(['Gender_no', 'Bi_stage'], ['category'] * 2)))
                ptd1 = ptd.astype(dict(zip(['Cluster_TSNE', 'Cluster_MDS'], ['int'] * 2)))
                for m in ['TSNE', 'MDS']:
                    ssa = base_vars + ['Bi_stage']
                    if i > 2:
                        ptd1 = pd.get_dummies(ptd1, columns=['Cluster_' + m], drop_first=True)
                        ssa += ['Cluster_' + m + '_' + str(c) for c in range(2, i + 1)]
                    else:
                        ssa += ['Cluster_' + m]
                    for x, y, title, dss in zip(['P', 'D'], [['PW', 'PM'], ['DW', 'DM']],
                                                ['Progression', 'Death'], [ptd_p, ptd_d]):
                        dss1 = pd.concat([ptd1,dss.iloc[:,-8:]],axis=1,join='inner')
                        for yy in y:
                            for aft,aft_model,cc in zip(['Weibull','LN','LL'],
                                [WeibullAFTFitter(),LogNormalAFTFitter(),LogLogisticAFTFitter()],
                                ['lambda','mu','alpha']):
                                aft_model.fit(dss1[[yy, x] + ssa], duration_col=yy, event_col=x)
                                f, ax = plt.subplots(figsize=(4, 10))
                                aft_model.plot(ax=ax)
                                plt.savefig(savepath + '\\{}_{}AFT_multi_{}C_{}D_{}_{}_in_{}.svg'.format(m,aft,str(i),str(j),x,yy[-1],g),
                                dpi=1200, bbox_inches='tight')
                                plt.close()
                                aft_model.summary.to_csv(savepath+'\\{}_{}AFT_multi_{}C_{}D_{}_{}_in_{}.txt'.format(m,aft,str(i),str(j),x,yy[-1],g),
                                sep='\t', encoding='utf_8_sig')
                                pd_aft_multi.loc['{}_{}AFT_multi_{}C_{}D_{}_{}_in_{}'.format(m,aft,str(i),str(j),x,yy[-1],g)] = [aft]+[aft_model.summary.loc[(cc+'_','Bi_stage'), 'p'],aft_model.summary.loc[[(cc+'_',s)for s in ssa[vl+1:]]]['p'].to_list()]+[aft_model.AIC_,aft_model.concordance_index_,aft_model.log_likelihood_]
    pd_aft_multi['P_min'] = np.asarray([np.min(np.asarray(pd_aft_multi.loc[i, 'P_value'])) for i in pd_aft_multi.index.to_list()])
    pd_aft_multi.sort_values('P_min', inplace=True)
    pd_aft_multi.to_csv(savepath + '\\multi_aft_pvalues.txt', sep='\t',encoding='utf_8_sig')
    return pd_aft_multi