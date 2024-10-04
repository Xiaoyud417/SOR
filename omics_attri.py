#required packages
import networkx as nx
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from minepy import MINE #maximal information coefficient (MIC)
import rcca #canonical correlation analysis (CCA)
from sklearn.preprocessing import StandardScaler
from itertools import combinations

plt.rcParams['font.sans-serif'] = ['SimHei']  # for displaying Chinese Characters
plt.rcParams['axes.unicode_minus'] = False # for displaying negative sign

fc=['ALL','FCR','VSA','F2R']+[''.join([d,str(j),v]) for v in ['a','i','u'] for d in ['F','B'] for j in range(1,4)]

dfc=dict(zip(fc,['#00ffff','#be03fd','#aaff32','#0165fc']+
             sns.light_palette('#047495')[-3:]+sns.light_palette('#0485d1')[-3:]+
             sns.light_palette('#cb416b')[-3:]+sns.light_palette('#ffd1df')[-3:]+
             sns.light_palette('#029386')[-3:]+sns.light_palette('#04d8b2')[-3:]))

#function for feature visualization by weight graph
def make_wg(data, feat_dim, savepath, out_op=True, ratio=0.2):
    #data: a dataframe of Pearson correlation matrix
    #feat_dim: feature dimension, namely F1a, F2a, etc.
    #savepath: folder for storing figures
    #out_op: option that controls whether to store the position information as a dataframe. This helps to reproduce the graph
    #ratio: parameter defining the weakest correlation that can be used for visualization. Zero means all.
    dfc1 = {ix: sns.dark_palette(dfc[feat_dim], n_colors=8)[9 - int(float(ix.split('_')[-2]) * 10)] for ix
            in data.index.to_list()}
    data.loc[[pc for pc in data.index.to_list() if pc.split('_')[1]==feat_dim],
                [pc for pc in data.columns.to_list() if pc.split('_')[1]==feat_dim]]
    data.reset_index(inplace=True)
    php210 = pd.melt(data, id_vars=['index'], value_vars=data.columns.to_list()[1:])
    php210 = php210.loc[php210['index'] != php210['variable']]
    php210.columns = ['from', 'to', 'Corr']
    patches = [mpatches.Patch(color=j, label=str((i+2)/10)) for i, j in enumerate(sns.dark_palette(dfc[feat_dim],n_colors=8))]
    php210_1 = php210.loc[php210['Corr'].abs() > ratio]
    php210_1=php210_1.loc[[php210_1.loc[ix,'from'].split('_')[2]!=php210_1.loc[ix,'to'].split('_')[2] for ix in php210_1.index.to_list()]]
    G = nx.from_pandas_edgelist(df=php210_1, source='from', target='to', edge_attr=['Corr'])
    pos = nx.spring_layout(G, seed=2)
    ppos = pd.DataFrame(pos.values(), index=pos, columns=['pos_x', 'pos_y'])
    if out_op:
        ppos.to_csv(savepath + '\\pos_spring_{}.txt'.format(feat_dim), encoding='utf_8_sig', sep='\t')
    plt.rcParams['font.size'] = 28
    f, ax = plt.subplots(figsize=(40, 40))
    nx.draw(G, pos=pos, arrows=False, with_labels=False,
            node_size=[2**(int(n.split('_')[-1])/10) + 100 for n in list(G.nodes.keys())],
            node_color=[sns.dark_palette(dfc[feat_dim], n_colors=8)[9 - int(float(n.split('_')[-2]) * 10)]
                        for n in list(G.nodes.keys())],
            node_shape="o", alpha=0.6,edge_color='gray',
            ax=ax,width=np.abs(php210_1['Corr'].to_numpy().flatten())*10-2)
    plt.legend(handles=patches, bbox_to_anchor=(0.5, 0.5), loc='best', ncol=1)
    ax.text(x=0,y=0,s='Total features: {}'.format(len(list(G.nodes.keys()))),
            fontsize=30,fontweight='bold')
    plt.savefig(savepath + '\\{}_NX_Spring.svg'.format(feat_dim),
                dpi=1200, bbox_inches='tight')
    plt.savefig(savepath + '\\{}_NX_Spring.png'.format(feat_dim),
                dpi=300, bbox_inches='tight')
    plt.close()


#parameters for MIC analysis
mine = MINE(alpha=0.6, c=15)

#Function for MIC value computation
def MIC_matirx0(dataframe, mine=mine,nvar=['OC_OPC'], nf=['PCA_FCR_3_0.4_20']):
    # dataframe refers to the dataframe at least containing features in nvar and nf
    # mine refers to the function used for MIC value computation
    # nvar controls the features involved in MIC analysis in one dimension, here refers to disease information
    # nf controls the features involved in MIC analysis in the other dimension, here refers to PCA-transformed speech omics features
    result = np.zeros([len(nf), len(nvar)])
    for k, i in tqdm(enumerate(nf)):
        for kk, j in enumerate(nvar):
            mine.compute_score(dataframe[i].to_numpy(), dataframe[j].to_numpy())
            result[k, kk] = mine.mic()
        time.sleep
    RT = pd.DataFrame(result, index=nf, columns=nvar)
    return RT

std=StandardScaler()
cca=rcca.CCA(numCC=2,ktype='gaussian',reg=0.05,verbose=False)

#function for cca score computation and feature selection via CCA methods
def CCA_selection(data, bp, savepath, feat_dim):
    #data refers to the dataframe containing all speech omics features
    #bp refers to the dataframe of biological profiles
    #set2 refers to the initial feature set
    #step controls the number of features deleted per iterative down sampling
    #ratio determines the anticipated percent of selected features
    #savepath refers to the absolute path for saving the selection procedure. This will help a lot in case of program crash.
    sor1=pd.DataFrame(std.fit_transform(data),index=data.index.to_list(),columns=data.columns.to_list())
    bp1=pd.DataFrame(std.fit_transform(bp),index=bp.index.to_list(),columns=bp.columns.to_list())
    data1=pd.concat([sor1,bp1],axis=1)
    set1=data.columns.to_list()
    n0=len(set1)
    csc = float('-inf')
    coe = 0.0
    pset = pd.DataFrame(columns=['CCA_coef'])
    pcoef = pd.DataFrame(columns=['CCA_coef'])
    n = 1
    while n < n0 and coe > csc:
        csc = coe
        for j in list(combinations(set1, len(set1) - 1)):
            cca.train([data1[list(j)].to_numpy(), data1[bp.columns.to_list()].to_numpy()])
            pset.loc['+'.join(j)] = cca.cancorrs[0]
        pset.sort_values('CCA_coef', inplace=True, ascending=False)
        set1 = pset.index.to_list()[0].split('+')
        coe = pset.iloc[0, 0]
        if coe > csc and n/50==n//50: #The selection outcomes are temporarily stored while deleting every other 50 features.
            with open(savepath + '\\{}_rfs_dim_tempo.txt'.format(feat_dim), 'a+') as ff:
                ff.write('\n'.join(set1))
        pset = pset.iloc[:1, :1]
        pcoef.loc['+'.join(set1)] = pset.iloc[0, 0]
        n += 1
    return set1,pcoef

class CCA_modularity:
    def __init__(self, fc, data):
        self.fc=fc
        self.data=data
        
    def min_pc(self):
        allf=self.data.columns.to_list()
        min_dic=dict(zip(self.fc,[1]*len(self.fc)))
        for f in self.fc:
            numf=[]
            for k in range(2,10):
                for d in range(10,101,10):
                    numf.append(len([i for i in allf if (i.split('_')[1]==f and i.split('_')[-1]==str(d) and i.split('_')[-2]==str(k/10))]))
            min_dic[f]=np.min(numf)
        return min_dic
                    
    def CCA_m(self, min_dic, f1, f2, name):
        npf1=np.zeros((80,min_dic[f1]))
        npf2=np.zeros((80,min_dic[f2]))
        iter=0
        while iter<80:
            for k in range(2, 10):
                for d in range(10, 101, 10):
                    for x in range(min_dic[f1]):
                        npf1[iter, x] = self.data.loc[
                            name, 'PCA_{}_{}_{}_{}'.format(f1, str(x + 1), str(k / 10), str(d))]
                    for y in range(min_dic[f2]):
                        npf2[iter, y] = self.data.loc[
                            name, 'PCA_{}_{}_{}_{}'.format(f2, str(y + 1), str(k / 10), str(d))]
                    iter += 1
        cca.train([npf1,npf2])
        coe = cca.cancorrs[0]
        return coe