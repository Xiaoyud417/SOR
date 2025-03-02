import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from minepy import MINE  # maximal information coefficient (MIC)
import rcca  # canonical correlation analysis (CCA)
from sklearn.preprocessing import StandardScaler
from itertools import combinations

# For displaying Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # for displaying negative sign

fc = ['ALL', 'FCR', 'VSA', 'F2R'] + [f'{d}{j}{v}' for v in ['a', 'i', 'u'] for d in ['F', 'B'] for j in range(1, 4)]

dfc = dict(zip(fc, ['#00ffff', '#be03fd', '#aaff32', '#0165fc'] +
               sns.light_palette('#047495')[-3:] + sns.light_palette('#0485d1')[-3:] +
               sns.light_palette('#cb416b')[-3:] + sns.light_palette('#ffd1df')[-3:] +
               sns.light_palette('#029386')[-3:] + sns.light_palette('#04d8b2')[-3:]))


def make_wg(data: pd.DataFrame, feat_dim: str, savepath: str, out_op: bool = True, ratio: float = 0.2) -> None:
    """
    Make a weight graph for a given feature dimension.

    Args:
        data (pd.DataFrame): A dataframe of Pearson correlation matrix.
        feat_dim (str): The feature dimension (e.g. F1a, F2a, etc.).
        savepath (str): Path to store weight graphs.
        out_op (bool): Option to store the position information as a dataframe. Defaults to True.
        ratio (float): Parameter to define the weakest correlation that can be used for visualization. Defaults to 0.2.
    """
    data1 = data.loc[[pc for pc in data.index.to_list() if
                      pc.split('_')[1] == feat_dim], [pc for pc in data.columns.to_list() if
                                                      pc.split('_')[1] == feat_dim]]
    data1.reset_index(inplace=True)
    php210 = pd.melt(data1, id_vars=['index'], value_vars=data1.columns.to_list()[1:])
    php210 = php210.loc[php210['index'] != php210['variable']]
    php210.columns = ['from', 'to', 'Corr']
    patches = [mpatches.Patch(color=j, label=str((i + 2) / 10)) for i, j
               in enumerate(sns.dark_palette(dfc[feat_dim], n_colors=8))]
    php210_1 = php210.loc[php210['Corr'].abs() > ratio]
    php210_1 = php210_1.loc[[php210_1.loc[ix, 'from'].split('_')[2] != php210_1.loc[ix, 'to'].split('_')[2] for ix
                             in php210_1.index.to_list()]]
    gg = nx.from_pandas_edgelist(df=php210_1, source='from', target='to', edge_attr=['Corr'])
    pos = nx.spring_layout(gg, seed=2)
    ppos = pd.DataFrame(pos.values(), index=pos, columns=['pos_x', 'pos_y'])
    if out_op:
        ppos.to_csv(savepath + '\\pos_spring_{}.txt'.format(feat_dim), encoding='utf_8_sig', sep='\t')
    plt.rcParams['font.size'] = 28
    f, ax = plt.subplots(figsize=(40, 40))
    nx.draw(gg, pos=pos, arrows=False, with_labels=False,
            node_size=[2**(int(n.split('_')[-1]) / 10) + 100 for n in list(gg.nodes.keys())],
            node_color=[sns.dark_palette(dfc[feat_dim], n_colors=8)[9 - int(float(n.split('_')[-2]) * 10)] for n
                        in list(gg.nodes.keys())],
            node_shape="o", alpha=0.6, edge_color='gray', ax=ax,
            width=np.abs(php210_1['Corr'].to_numpy().flatten()) * 10 - 2)
    plt.legend(handles=patches, bbox_to_anchor=(0.5, 0.5), loc='best', ncol=1)
    ax.text(x=0, y=0, s='Total features: {}'.format(len(list(gg.nodes.keys()))), fontsize=30, fontweight='bold')
    plt.savefig(savepath + '\\{}_NX_Spring.svg'.format(feat_dim), dpi=1200, bbox_inches='tight')
    plt.savefig(savepath + '\\{}_NX_Spring.png'.format(feat_dim), dpi=300, bbox_inches='tight')
    plt.close()


# Parameters for MIC analysis
mine = MINE(alpha=0.6, c=15)

# Function for MIC value computation


def mic_matrix(dataframe: pd.DataFrame, mic_mine: MINE = mine, nvar: list = None, nf: list = None) -> pd.DataFrame:
    """
    Compute MIC values for a given dataframe.

    Args:
        dataframe: The dataframe containing all speech omics features.
        mic_mine: The MINE object for computing MIC values.
        nvar: The features involved in MIC analysis in one dimension.
        nf: The features involved in MIC analysis in the other dimension.

    Returns:
        A dataframe containing MIC values.
    """
    result = np.zeros([len(nf), len(nvar)])
    for k, i in enumerate(nf):
        for kk, j in enumerate(nvar):
            mic_mine.compute_score(dataframe[i].to_numpy(), dataframe[j].to_numpy())
            result[k, kk] = mic_mine.mic()
    pd_mic = pd.DataFrame(result, index=nf, columns=nvar)
    return pd_mic


std = StandardScaler()
cca = rcca.CCA(numCC=2, ktype='gaussian', reg=0.05, verbose=False)

# function for cca score computation and feature selection via CCA methods


def cca_selection(data: pd.DataFrame, bp: pd.DataFrame, savepath: str, feat_dim: str) -> tuple:
    """
    Compute CCA scores and select features using CCA methods.

    Args:
        data (pd.DataFrame): Dataframe containing all speech omics features.
        bp (pd.DataFrame): Dataframe of biological profiles.
        savepath (str): Path to store the selection procedure.
        feat_dim (str): Feature dimension (e.g. F1a, F2a, etc.).

    Returns:
        Tuple: A list of the selected features, and a dataframe containing their corresponding CCA scores.
    """
    sor1 = pd.DataFrame(
        std.fit_transform(data),
        index=data.index.to_list(),
        columns=data.columns.to_list())
    bp1 = pd.DataFrame(
        std.fit_transform(bp),
        index=bp.index.to_list(),
        columns=bp.columns.to_list())
    data1 = pd.concat([sor1, bp1], axis=1)
    set1 = data.columns.to_list()
    n0 = len(set1)
    csc = float('-inf')
    coe = 0.0
    pset = pd.DataFrame(columns=['CCA_coef'])
    pcoef = pd.DataFrame(columns=['CCA_coef'])
    n = 1
    while n < n0 and coe > csc:
        csc = coe
        for j in list(combinations(set1, len(set1) - 1)):
            cca.train([data1[list(j)].to_numpy(),
                       data1[bp.columns.to_list()].to_numpy()])
            pset.loc['+'.join(j)] = cca.cancorrs[0]
        pset.sort_values('CCA_coef', inplace=True, ascending=False)
        set1 = pset.index.to_list()[0].split('+')
        coe = pset.iloc[0, 0]
        # The selection outcomes are temporarily stored while deleting every
        # other 50 features.
        if coe > csc and n / 50 == n // 50:
            with open(savepath + '\\{}_rfs_dim_tempo.txt'.format(feat_dim), 'a+') as ff:
                ff.write('\n'.join(set1))
        pset = pset.iloc[:1, :1]
        pcoef.loc['+'.join(set1)] = pset.iloc[0, 0]
        n += 1
    return set1, pcoef


class cca_modularity:
    def __init__(self, fc_list: list, data: pd.DataFrame) -> None:
        """
        Initialize the modularity analysis using CCA.

        Args：
            fc (list): A list containing all 22 feature dimensions.
            data (pd.DataFrame): Dataframe containing all speech omics features.
        """
        self.fc_list = fc_list
        self.data = data

    def min_pc(self) -> dict:
        """
        Compute the minimum number of features for each dimension.

        Returns:
            A dictionary containing the minimum number of features for each dimension
        """
        allf = self.data.columns.to_list()
        min_dic = dict(zip(self.fc_list, [1] * len(self.fc_list)))
        for f in self.fc_list:
            numf = []
            for k in range(2, 10):
                for d in range(10, 101, 10):
                    numf.append(len([i for i in allf if (i.split('_')[1] == f and i.split(
                        '_')[-1] == str(d) and i.split('_')[-2] == str(k / 10))]))
            min_dic[f] = np.min(numf)
        return min_dic

    def cca_m(self, min_dic: dict, f1: str, f2: str, name: str) -> float:
        """
        Compute the CCA score for two specified feature dimensions.

        Args:
            min_dic (dict): A dictionary containing the minimum number of features for each dimension.
            f1 (str): One feature dimension involved in modularity analysis.
            f2 (str): The other feature dimension involved in modularity analysis.
            name (str): Name of the specified case.

        Returns:
            float: CCA score for two specified feature dimensions.
        """
        npf1 = np.zeros((80, min_dic[f1]))
        npf2 = np.zeros((80, min_dic[f2]))
        iter_kd = 0
        while iter_kd < 80:
            for k in range(2, 10):
                for d in range(10, 101, 10):
                    for x in range(min_dic[f1]):
                        npf1[iter_kd, x] = self.data.loc[
                            name, 'PCA_{}_{}_{}_{}'.format(f1, str(x + 1), str(k / 10), str(d))]
                    for y in range(min_dic[f2]):
                        npf2[iter_kd, y] = self.data.loc[
                            name, 'PCA_{}_{}_{}_{}'.format(f2, str(y + 1), str(k / 10), str(d))]
                    iter_kd += 1
        cca.train([npf1, npf2])
        coe = cca.cancorrs[0]
        return coe
