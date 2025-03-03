import pandas as pd
import numpy as np
from pyclust import KMedoids
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter, WeibullAFTFitter, LogLogisticAFTFitter, LogNormalAFTFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # for displaying Chinese Characters
plt.rcParams['axes.unicode_minus'] = False  # For displaying negative sign


def pam_cluster(phnc: pd.DataFrame, pscc: pd.DataFrame, savepath: str, lifepath: str,
                features_in: list, random_state: int = 2, pic: bool = True) -> tuple:
    """
    Perform partitioning around medoids (PAM) clustering and log-rank test.

    Args:
        phnc (pd.DataFrame): DataFrame containing disease information of all OC and OPC patients.
        pscc (pd.DataFrame): DataFrame containing disease information of all cancer patients whose pathology
                             was squamous cell carcinoma (SCC).
        savepath (str): Directory to save files.
        lifepath (str): Directory that stores files of disease variables and follow-up survival information.
        features_in (list): Selected speech omics features used for manifold learning.
        random_state (int, optional): Random state for KMedoids. Defaults to 2.
        pic (bool, optional): Whether to plot Kaplan-Meier curves. Defaults to True.

    Returns:
        tuple: A tuple containing the Excel file of clustering outcomes and log-rank p-values.
    """
    writer1 = pd.ExcelWriter(savepath + '\\TSNE+MDS_outcome.xlsx')
    ptdp = pd.read_csv(lifepath + '\\Progression_12.txt', index_col=0, sep='\t')
    ptdd = pd.read_csv(lifepath + '\\Death_52.txt', index_col=0, sep='\t')

    plogrank = pd.DataFrame(columns=['P_value'])

    for i in range(2, 5):
        for j in range(2, 4):
            for g, d in zip(['HNC', 'SCC'], [phnc, pscc]):
                data_std = StandardScaler().fit_transform(d[features_in])
                td2 = MDS(n_components=j, max_iter=1000, random_state=random_state).fit_transform(data_std)
                k = KMedoids(n_clusters=i, distance='euclidean', max_iter=1000).fit_predict(td2)
                ptd2 = pd.DataFrame(td2, index=d.index.to_list(),
                                    columns=[g + '_ALL_MDS_' + str(x) for x in range(1, j + 1)])
                ptd2['Cluster_MDS'] = k + 1
                td1 = TSNE(n_components=j, init="pca", learning_rate=250, random_state=random_state,
                           perplexity=50, method='exact', early_exaggeration=15).fit_transform(data_std)
                k = KMedoids(n_clusters=i, distance='euclidean', max_iter=1000).fit_predict(td1)
                ptd1 = pd.DataFrame(td1, index=d.index.to_list(),
                                    columns=[g + '_ALL_TSNE_' + str(x) for x in range(1, j + 1)])
                ptd1['Cluster_TSNE'] = k + 1
                ptd12 = pd.concat([ptd1, ptd2, d], axis=1, join='inner')
                ptd_d = pd.concat([ptd12, ptdd], axis=1, join='inner')
                ptd_p = pd.concat([ptd12, ptdp], axis=1, join='inner')
                for x, y, title, dss in zip(['P', 'D'],
                                            [['PW', 'PM'], ['DW', 'DM']],
                                            ['Progression', 'Death'],
                                            [ptd_p, ptd_d]):
                    for yy in y:
                        ev_time = dss[yy]
                        evt = dss[x]
                        for m in ['TSNE', 'MDS']:
                            if i > 2:
                                lr = multivariate_logrank_test(ev_time, dss['Cluster_' + m], evt, alpha=.99)
                                plogrank.loc[
                                    str(i) + 'C_' + str(j) + 'D_ALL_' + m + '_' + x + '_' + yy[-1] + '_in_' + g] = \
                                    lr.p_value
                                for co in list(combinations([cc for cc in range(1, i + 1)], 2)):
                                    lr = logrank_test(ev_time[dss['Cluster_' + m] == co[0]],
                                                      ev_time[dss['Cluster_' + m] == co[1]],
                                                      evt[dss['Cluster_' + m] == co[0]],
                                                      evt[dss['Cluster_' + m] == co[1]], alpha=.99)
                                    plogrank.loc[
                                        str(i) + 'C_' + str(j) + 'D_ALL_' + m + ''.join(
                                            [str(ccc) for ccc in co]) + '_' + x + '_' + yy[-1] + '_in_' + g] = \
                                        lr.p_value
                            else:
                                group = (dss['Cluster_' + m] == 1)
                                lr = logrank_test(ev_time[group],
                                                  ev_time[~group],
                                                  evt[group],
                                                  evt[~group], alpha=.99)
                                plogrank.loc[
                                    str(j) + 'C_' + str(j) + 'D_ALL_' + m + '_' + x + '_' + yy[-1] + '_in_' + g] = \
                                    lr.p_value
                            if pic:
                                ax = plt.subplot(111)
                                for ii in range(i):
                                    group = (dss['Cluster_' + m] == ii + 1)
                                    km = KaplanMeierFitter()
                                    km.fit(ev_time[group],
                                           event_observed=evt[group],
                                           label="cluster {}".format(str(ii + 1)))
                                    km.plot_survival_function(ax=ax, color=sns.husl_palette(n_colors=i)[ii])
                                ax.set_title(title + ' in units of ' + dict(zip(['W', 'M'], ['week', 'month']))[yy[-1]],
                                             fontsize=20, fontweight='bold')
                                ax.set_xlabel(dict(zip(['W', 'M'], ['Weeks', 'Months']))[yy[-1]],
                                              fontsize=18, fontweight='bold')
                                plt.savefig(
                                    savepath + '\\Cluster' + str(i) + '_' + 'ALL_pam_in_' +
                                    g + '_' + str(j) + 'D_' + title + '_' + yy[-1] + '_KMP.svg',
                                    dpi=1200, bbox_inches='tight')
                                plt.close()
                ptd12.to_excel(writer1, g + '_D' + str(j) + '_C' + str(i))
                writer1._save()
    plogrank.sort_values('P_value', inplace=True)
    plogrank.to_csv(savepath + '\\Logrank_pvalues.txt', sep='\t', encoding='utf_8_sig')
    return pd.ExcelFile(savepath + '\\TSNE+MDS_outcome.xlsx'), plogrank


# Function for survival analysis including log rank and Weibull accelerated failure time model


def survival_aft(phnc: pd.DataFrame, pscc: pd.DataFrame, path: str,
                 lifepath: str, savepath: str, base_vars: list = None) -> pd.DataFrame:
    """
    Perform multivariate survival analysis using accelerated failure time (AFT) model.

    Args：
        phnc (pd.DataFrame): DataFrame containing disease information of all OC and OPC patients.
        pscc (pd.DataFrame): DataFrame containing disease information of all cancer patients whose pathology
                             was squamous cell carcinoma (SCC).
        path (str): Path of unsupervised learning outcome.
        lifepath (str): Directory containing files of disease variables and follow-up survival information.
        savepath (str): Directory to save files.
        base_vars (list, optional): Disease variables to include as independent factors used for
                                    multivariate survival analysis.

    Returns:
        pd.DataFrame: A dataframe containing the results of multivariate survival analysis.
    """
    cds = pd.ExcelFile(path)
    ptd_p = pd.read_csv(
        lifepath +
        '\\Progression_12.txt',
        index_col=0,
        sep='\t')
    ptd_d = pd.read_csv(lifepath + '\\Death_52.txt', index_col=0, sep='\t')
    pd_aft_multi = pd.DataFrame(
        columns=[
            'Bi_stage_p',
            'P_value',
            'AIC_partial',
            'C_index',
            'Log_LH'])
    vl = len(base_vars)
    for i in range(2, 5):
        for j in range(2, 4):
            for g, d in zip(['HNC', 'SCC'],
                            [phnc.index.to_list(), pscc.index.to_list()]):
                ptd = cds.parse(sheet_name=g + '_D' + str(j) + '_C' + str(i), index_col=0)
                ptd['Bi_stage'] = np.asarray(
                    [0 if (ptd.loc[ix, 'Stage_3'] == 0 and ptd.loc[ix, 'Stage_4'] == 0) else 1 for ix in
                     ptd.index.to_list()])
                ptd = ptd.astype(
                    dict(zip(['Gender', 'Bi_stage'], ['category'] * 2)))
                ptd1 = ptd.astype(
                    dict(zip(['Cluster_TSNE', 'Cluster_MDS'], [np.int] * 2)))
                ssa = base_vars + ['Bi_stage']
                for m in ['TSNE', 'MDS']:
                    if i > 2:
                        ptd1 = pd.get_dummies(
                            ptd1, columns=[
                                'Cluster_' + m], drop_first=True)
                        ssa += ['Cluster_' + m + '_' +
                                str(c) for c in range(2, i + 1)]
                    else:
                        ssa += ['Cluster_' + m]
                    for x, y, title, dss in zip(['P', 'D'], [['PW', 'PM'], ['DW', 'DM']],
                                                ['Progression', 'Death'], [ptd_p, ptd_d]):
                        dss1 = pd.concat([ptd1, dss.iloc[:, -8:]], axis=1, join='inner')
                        for yy in y:
                            for aft, aft_model, cc in zip(['Weibull', 'LN', 'LL'],
                                                          [WeibullAFTFitter(),
                                                           LogNormalAFTFitter(),
                                                           LogLogisticAFTFitter()],
                                                          ['lambda', 'mu', 'alpha']):
                                aft_model.fit(dss1[[yy, x] + ssa], duration_col=yy, event_col=x)
                                f, ax = plt.subplots(figsize=(4, 10))
                                aft_model.plot(ax=ax)
                                plt.savefig(savepath + '\\{}_{}AFT_multi_{}C_{}D_{}_{}_in_{}.svg'.format(
                                    m, aft, str(i), str(j), x, yy[-1], g), dpi=1200, bbox_inches='tight')
                                plt.close()
                                aft_model.summary.to_csv(
                                    savepath+'\\{}_{}AFT_multi_{}C_{}D_{}_{}_in_{}.txt'.format(
                                        m, aft, str(i), str(j), x, yy[-1], g), sep='\t', encoding='utf_8_sig')
                                pd_aft_multi.loc[
                                    '{}_{}AFT_multi_{}C_{}D_{}_{}_in_{}'.format(
                                        m, aft, str(i), str(j), x, yy[-1], g)] = [aft] + \
                                                                                 [aft_model.summary.loc[
                                                                                      (cc+'_', 'Bi_stage'), 'p'],
                                                                                  aft_model.summary.loc[
                                                                                      [(cc+'_', s) for s in
                                                                                       ssa[vl+1:]]]['p'].to_list()] +\
                                                                                 [aft_model.AIC_,
                                                                                  aft_model.concordance_index_,
                                                                                  aft_model.log_likelihood_]
    pd_aft_multi['P_min'] = np.asarray([np.min(np.asarray(pd_aft_multi.loc[i, 'P_value'])) for i in
                                        pd_aft_multi.index.to_list()])
    pd_aft_multi.sort_values('P_min', inplace=True)
    pd_aft_multi.to_csv(savepath + '\\multi_aft_pvalues.txt',
                        sep='\t', encoding='utf_8_sig')
    return pd_aft_multi
