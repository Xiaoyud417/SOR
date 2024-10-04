import numpy as np
import joblib,os,time
from tqdm import tqdm
from sklearn.cross_decomposition import CCA
from .SOR import *
from .omics_attri import CCA_modularity
import pandas as pd
from itertools import combinations
from warnings import filterwarnings
filterwarnings('ignore')
fc=['ALL','FCR','VSA','F2R']+[''.join([d,str(j),v]) for v in ['a','i','u'] for d in ['F','B'] for j in range(1,4)]
cca = CCA(n_components=2, max_iter=300)

def sor_fit(fpath,outpath,ev_case,pcapath,sdrpath,pdr=pd.DataFrame()):
    epo = SOR(path=fpath,outpath=outpath, case=ev_case, thresholds=1, intervals=1, pdr=pdr)
    pkde = epo.get_kde(kpath='\\'.join(fpath.split('\\')[:-1])+'\\2.FormantExcel')
    pca_data=[]
    for th in range(2, 10):
        threshold = th / 10
        epo = SOR(path=fpath, outpath=outpath, case=ev_case, thresholds=threshold, intervals=1, pdr=pdr)
        pdn = epo.get_pdn(ipath='\\'.join(fpath.split('\\')[:-1]) + '\\3.DerivedMetric')
        for thr in range(10, 101, 10):
            interval = thr
            sdr0=pd.read_csv(sdrpath+'\\SDR_{}_{}.txt'.format(str(threshold),str(interval)),sep='\t',index_col=0)
            sdr_dic=dict(zip(fc,[sdr0.to_numpy()]+[sdr0.to_numpy()[:,interval*xx:interval*(xx+1)] for xx in range(21)]))
            epo = SOR(path=fpath,outpath=outpath, case=ev_case, thresholds=threshold, intervals=interval, pdr=pdr)
            pdl = epo.get_pdl(pdn,spath='\\'.join(fpath.split('\\')[:-1]) +'\\4.SDRdata')
            npd0, pddata = epo.get_ALL(pdl,spath='\\'.join(fpath.split('\\')[:-1]) +'\\4.SDRdata')
            x_transform = []
            o_pca=pd.read_csv(pcapath+'\\PCA_transform_{}_{}.txt'.format(str(threshold),str(interval)),sep='\t',index_col=0)
            for fd,fdd in zip(fc,[pddata.to_numpy()] + npd0):
                nfeat=len([col for col in o_pca.columns.to_list() if col.split('_')[1]==fd])
                pca_model=joblib.load(pcapath+'\\PCA_{}_{}_{}.pickle'.format(str(threshold),str(interval),fd))
                X_pca = pca_model.fit_transform(np.vstack((fdd,sdr_dic[fd])))[:len(ev_case),:nfeat]
                ppca = pd.DataFrame(X_pca, index=pddata.index.to_list(),columns=['PCA_{}_{}_{}_{}'.format(fd,od,str(threshold),str(interval)) for od in range(1, nfeat + 1)])
                x_transform.append(ppca)
            pxt = pd.concat(x_transform, axis=1)
            pca_data.append(pxt)
    ev_sor=pd.concat(pca_data,axis=1)
    cmo=CCA_modularity(fc=fc,data=ev_sor)
    md=cmo.min_pc()
    pcca=pd.DataFrame()
    for (f1, f2) in list(combinations(fc, 2)):
        for name in ev_case:
            cca_coe = cmo.CCA_m(min_dic=md, name=name, f1=f1, f2=f2)
            pcca.loc[name, '-'.join([f1, f2])] = cca_coe
    pev_sor=pd.concat([ev_sor,pcca],axis=1)
    pev_sor.to_csv(outpath + '\\EV_features.txt',sep='\t', encoding='utf_8_sig')
    return pev_sor

tsk=['OC_OPC','Type','Malignancy','Lesion','SCC','Ttfc_1','Ttfc_2','Ttfc_3','T_1','T_2','T_3',
    'Ntfc_12','Nscc_12','N_0','N_1','N_2','Stage_1','Stage_2','Stage_3','Ntfc_0','Ntfc_1','Ntfc_2',
    'Stagetfc_1','Stagetfc_2','Stagetfc_3','HCOPML-T1','OPML-T1']
dtm=dict(zip(tsk,['CCA-FCR_ET_OC_OPC','CCA-F1a_ET_Type','0.7_80_GBC_Malignancy+Concatenated','CCA-B1u_LDA_Lesion','0.2_90_ET_SCC+Concatenated',
                  'CCA-F2a_ET_Ttfc_1','0.8_40_GBC_Ttfc_2','OFS_ABC_Ttfc_3','0.3_100_GBC_T_1+Concatenated','0.8_10_GBC_T_2+Concatenated','OFS_NB_T_3',
                  '0.3_100_GBC_Ntfc_12+Concatenated','0.6_90_GBC_Nscc_12+Merged','0.6_50_ET_N_0','OFS_LR_N_1','OFS_LR_N_2','0.8_70_ET_Stage_1','0.9_10_LR_Stage_2',
                  'OFS_ABC_Stage_3','0.7_20_ET_Ntfc_0+Concatenated','0.5_30_MLP_Ntfc_1+Concatenated','OFS_RFC_Ntfc_2','0.3_100_LDA_Stagetfc_1+Concatenated',
                  '0.7_20_ET_Stagetfc_2+Merged','0.8_60_MLP_Stagetfc_3+Merged','0.7_100_LDA_HCOPML-T1+Concatenated','0.2_100_GBC_OPML-T1+Concatenated']))

def ev_fit(model_path,task,ev_data):
    ev_data['Age']=[int(ix[5:8]) for ix in ev_data.index.to_list()]
    ev_data['Gender_no']=[int(ix[9]) for ix in ev_data.index.to_list()]
    mm=joblib.load(model_path+'\\{}.pickle'.format(dtm[task]))
    features=list(mm.feature_names_in_)
    y_pred=mm.predict(ev_data[features].to_numpy())
    y_prob = mm.predict_proba(ev_data[features].to_numpy())
    return y_pred,y_prob
    