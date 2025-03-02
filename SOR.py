import numpy as np
import pandas as pd
from scipy import stats, interpolate
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle

dv = dict(zip(['a', 'i', 'u'], ['fXBQhWf', 'fkEMYTo', 'fIHXxem']))


def s_kde(df: pd.DataFrame, top_percent: float = 1, cols: list = None, min_pairs: int = 1,
          col_name: str = 'kde', inplace: bool = False) -> pd.DataFrame:
    """
    Compute kernel density estimate (KDE) for a given DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame
        top_percent (float, optional): Top percentage of data to consider. Defaults to 1.
        cols (list): Columns to consider. Defaults to None.
        min_pairs (int): Minimum number of pairs to consider. Defaults to 1.
        col_name (str): Name of the output column. Default to 'kde'.
        inplace (bool): Whether to modify the original DataFrame. Defaults to False.

    Returns:
        pd.DataFrame: Output DataFrame
    """
    if cols is None:
        cols = ['F1(Hz)', 'F2(Hz)']
    if df.shape[0] >= min_pairs:
        dd = []
        for i in range(len(cols)):
            dd.append(df[cols[i]].to_numpy())
        kde = stats.gaussian_kde(np.vstack(dd))
        density = kde(np.vstack(dd))
        df[col_name] = density
        if inplace:
            df_new = df.sort_values(by=[col_name], ascending=False).iloc[:round(top_percent * df.shape[0]), :]
            return df_new
        else:
            return df
    else:
        return df


def get_fft(name: str, fpath: str) -> tuple:
    """
    Get FFT data for a given name and file path.

    Args:
        name (str): Name of the case
        fpath (str): FiLe path

    Returns：
        tuple: FFT data for /a/, /i/, and /u/
    """
    ppa, ppi, ppu = pd.read_excel(fpath + '\\' + name + '.xlsx', index_col=0, sheet_name='a_ori'),\
        pd.read_excel(fpath + '\\' + name + '.xlsx', index_col=0, sheet_name='i_ori'),\
        pd.read_excel(fpath + '\\' + name + '.xlsx', index_col=0, sheet_name='u_ori')
    return ppa, ppi, ppu


def FCRs(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame,
         name1: list = None, iter_num: int = 1):
    """
    Construct a generator for iterative computing of derived metrics (FCR, VSA, and F2R).

    Args:
        df1 (pd.DataFrame): DataFrame containing the formant measurements of /a/.
        df2 (pd.DataFrame): DataFrame containing the formant measurements of /i/.
        df3 (pd.DataFrame): DataFrame containing the formant measurements of /u/.
        name1 (list): Columns to compute the derived metrics
        iter_num (int): Number of iterations for computing the derived metrics

    Returns:
        Object: Generator for iterative computing
    """
    if name1 is None:
        name1 = ['F1(Hz)', 'F2(Hz)']
    count = 0
    while count < iter_num:
        df11 = df1.sample(n=1)
        df22 = df2.sample(n=1)
        df33 = df3.sample(n=1)
        l1 = list(df11[name1].to_numpy().flatten())
        l2 = list(df22[name1].to_numpy().flatten())
        l3 = list(df33[name1].to_numpy().flatten())
        fcr_value = (l3[1] + l1[1] + l2[0] + l3[0]) / (l2[1] + l1[0])
        f2r_value = l2[1] / l3[1]
        vsa_value = np.abs((l2[0] * (l1[1] - l3[1]) + l1[0] * (l3[1] - l2[1]) + l3[0] * (
                l2[1] - l1[1])) / 2)
        yield fcr_value, f2r_value, vsa_value
        count += 1


class SOR:
    def __init__(self, path: str, outpath: str, case: list, thresholds: float, intervals: int, pdr: pd.DataFrame):
        """
        Initialize Speech Omics Representation (SOR) object.

        Args:
            path (str): Path to the data files
            outpath (str): Path to the output files
            case (list): List of case names
            thresholds (float): threshold values for KDE selection
            intervals (int): Number of intervals for sparse coding
            pdr (pd.DataFrame): DataFrame containing the ranges of each feature dimension for sparse coding.
        """
        self.path = path
        self.outpath = outpath
        self.case = case
        self.thresholds = thresholds
        self.intervals = intervals
        self.pdr = pdr

    def get_kde(self, kpath: str) -> None:
        """
        Compute KDE for each case.

        Args:
            kpath (str): Path to store the output KDE Excel files.
        """
        for i in self.case:
            fta = pd.read_table(self.path + '\\AUV1' + i + dv['a'] + '.Table', index_col=0, sep=',')
            fta = fta[['F1(Hz)', 'B1(Hz)', 'F2(Hz)', 'B2(Hz)', 'F3(Hz)', 'B3(Hz)']]
            fta = fta.loc[fta['F3(Hz)'] != '--undefined--']
            fta = fta.astype('float')
            fta = fta.loc[(fta['F1(Hz)'] > 500) & (fta['F1(Hz)'] < 1200)]
            fta = fta.loc[(fta['F2(Hz)'] > 800) & (fta['F2(Hz)'] < 2000)]
            fti = pd.read_table(self.path + '\\AUV1' + i + dv['i'] + '.Table', index_col=0, sep=',')
            fti = fti[['F1(Hz)', 'B1(Hz)', 'F2(Hz)', 'B2(Hz)', 'F3(Hz)', 'B3(Hz)']]
            fti = fti.loc[fti['F3(Hz)'] != '--undefined--']
            fti = fti.astype('float')
            fti = fti.loc[(fti['F1(Hz)'] > 200) & (fti['F1(Hz)'] < 600)]
            fti = fti.loc[(fti['F2(Hz)'] > 1000) & (fti['F2(Hz)'] < 3800)]
            ftu = pd.read_table(self.path + '\\AUV1' + i + dv['u'] + '.Table', index_col=0, sep=',')
            ftu = ftu[['F1(Hz)', 'B1(Hz)', 'F2(Hz)', 'B2(Hz)', 'F3(Hz)', 'B3(Hz)']]
            ftu = ftu.loc[ftu['F3(Hz)'] != '--undefined--']
            ftu = ftu.astype('float')
            ftu = ftu.loc[(ftu['F1(Hz)'] > 250) & (ftu['F1(Hz)'] < 900)]
            ftu = ftu.loc[(ftu['F2(Hz)'] > 500) & (ftu['F2(Hz)'] < 1800)]
            writer = pd.ExcelWriter(kpath + '\\' + i + '.xlsx')
            for v, t in zip(['a', 'i', 'u'], [fta, fti, ftu]):
                pdl = [t]
                for m, n in zip([['F1(Hz)', 'F2(Hz)'], ['F3(Hz)'], ['B1(Hz)'], ['B2(Hz)'], ['B3(Hz)']],
                                ['kde', 'kde_F3', 'kde_B1', 'kde_B2', 'kde_B3']):
                    nt = s_kde(t, cols=m, col_name=n, inplace=True)
                    pdl.append(nt[n].to_frame())
                ppdl = pd.concat(pdl, axis=1).iloc[:, :11]
                ppdl.sort_values('kde', ascending=False, inplace=True)
                ppdl.to_excel(writer, v + '_ori')
            writer._save()
            writer.close()

    def dense_discrete(self, data: np.ndarray, dsc_range: list = None, index: str = ' ') -> pd.DataFrame:
        """
        Calculate absolute discrete density values for a given 1-D array.

        Args:
            data (np.ndarray): 1-D array to calculate absolute discrete density values.
            dsc_range (list): List containing two floats indicating the range of discretion.
            index (str): Name for a specific case. Defaults to ' '.

        Returns:
            pd.DataFrame: Output DataFrame
        """
        if dsc_range is None:
            dsc_range = [0, 1]
        logarr = sorted([x for x in np.logspace(dsc_range[0], dsc_range[1], num=self.intervals, endpoint=True)])
        ddata = pd.DataFrame(columns=logarr)
        for j in reversed(range(self.intervals)):
            ddata.loc[index, logarr[j]] = data[data > float(logarr[j])].shape[0]
            data = data[data <= float(logarr[j])]
        return ddata

    def get_pdn(self, ipath: str) -> pd.DataFrame:
        """
        Yield 10000 iterations of derived metrics (FCR, VSA, and F2R) for a list of cases.

        Args:
            ipath (str): Path to store the Text files containing derived metrics measurements.

        Return:
            pd.DataFrame: Output DataFrame
        """
        pdn = pd.DataFrame(columns=[i + '_' + str(j) for i in ['FCR', 'VSA', 'F2R'] for j in range(10000)])
        sik = {}
        for name in self.case:
            ppa, ppi, ppu = get_fft(name, fpath='\\'.join(self.path.split('\\')[:-1]) + '\\2.FormantExcel')
            ffta = ppa.sort_values('kde', ascending=False).iloc[:int(np.ceil(self.thresholds * ppa.shape[0])), :]
            ffti = ppi.sort_values('kde', ascending=False).iloc[:int(np.ceil(self.thresholds * ppi.shape[0])), :]
            fftu = ppu.sort_values('kde', ascending=False).iloc[:int(np.ceil(self.thresholds * ppu.shape[0])), :]
            fset = FCRs(df1=ffta, df2=ffti, df3=fftu, name1=['F1(Hz)', 'F2(Hz)'], iter_num=10000)
            iter_num = 0
            nv = np.zeros((10000, 3))
            try:
                while iter_num < 10000:
                    fcr1, vsa1, f2r1 = next(fset)
                    nv[iter_num, 0] = fcr1
                    nv[iter_num, 1] = vsa1
                    nv[iter_num, 2] = f2r1
                    iter_num += 1
            except StopIteration:
                sik[name] = iter_num
            vfcr = np.sort(nv[:, 0])[::-1]
            vvsa = np.sort(nv[:, 1])[::-1]
            vf2r = np.sort(nv[:, 2])[::-1]
            pdn.loc[name] = np.hstack([vfcr, vvsa, vf2r])
        for i in list(sik.keys()):
            for x, y in enumerate(['FCR', 'VSA', 'F2R']):
                sp = pdn.loc[i, pdn.columns.to_list()[10000 * x:10000 * (x + 1)]].to_numpy()
                sp = sp[sp != 0]
                x = range(sp.shape[0])
                x_new = np.linspace(0, sp.shape[0], 10000)
                tck = interpolate.splrep(x, sp)
                y_smooth = interpolate.splev(x_new, tck)
                y_sp = np.sort(y_smooth)[::-1]
                pdn.loc[i, [s for s in pdn.columns.to_list() if s.startswith(y)]] = y_sp
        pdn.to_csv(ipath + '\\data10000_{}.txt'.format(str(self.thresholds)), sep='\t', encoding='utf_8_sig')
        return pdn

    def get_pdl(self, pdn: pd.DataFrame, spath: str) -> tuple:
        """
        Get the sparse-distribute representations (SDR) for a list of cases.

        Args:
            pdn (pd.DataFrame): DataFrame containing absolute discrete density values.
            spath (str): Path to store the SDR data.

        Returns:
            tuple: A list containing 21 feature-level SDR data, and a DataFrame containing case-level SDR data (ALL)
        """
        mms = MinMaxScaler()
        npd0 = []
        for i in range(21):
            npd0.append(mms.fit_transform(pdn.iloc[:, self.intervals * i:self.intervals * (i + 1)]))
        npd = np.hstack(npd0)
        pddata = pd.DataFrame(npd, index=pdn.index.to_list(), columns=pdn.columns.to_list())
        pddata.to_csv(spath + '\\SDR_{}_{}.txt'.format(str(self.thresholds), str(self.intervals)), sep='\t',
                      encoding='utf_8_sig')
        return npd0, pddata

    def sor_pca(self, npd0: list, pddata: pd.DataFrame, ppath: str) -> pd.DataFrame:
        """
        Get the Speech Omics representations (SOR) for a list of cases.

        Args:
            npd0: List containing 21 feature-level SDR data
            pddata: DataFrame containing case-level SDR data (ALL)
            ppath: Path to store the SDR data

        Returns:
            pd.DataFrame: DataFrame containing SOR data.
        """
        pca = PCA()
        pnpr = pd.DataFrame(columns=['Accumulated ratio of ' + i for i in ['ALL', 'FCR', 'VSA', 'F2R'] +
                                     [x + y for x in ['F1', 'F2', 'F3', 'B1', 'B2', 'B3'] for y in ['a', 'i', 'u']]])
        acn = np.ones(22, dtype=int)
        x_transform = []
        for k, (m, n) in enumerate(zip(pnpr.columns.to_list(), [pddata] + npd0)):
            pca.fit(n)
            npr = pca.explained_variance_ratio_
            for i in range(1, npr.shape[0] + 1):
                pnpr.loc[i, m] = np.sum(npr[:i])
            acn[k] = pnpr.loc[pnpr[m] >= 0.95].index.to_list()[0]
            x_pca = pca.fit_transform(n)[:, :acn[k]]
            ppca = pd.DataFrame(x_pca, index=pddata.index.to_list(),
                                columns=['PCA_' + str.split(m, ' ')[-1] + '_' + str(i) for i in range(1, acn[k] + 1)])
            x_transform.append(ppca)

            with open(ppath + '\\PCA_{}_{}_{}.pickle'.format(str(self.thresholds),
                                                             str(self.intervals), str.split(m, ' ')[-1]), 'wb') as f:
                pickle.dump(pca, f)

        pxt = pd.concat(x_transform, axis=1)
        pxt.columns = ['_'.join([x, str(self.thresholds), str(self.intervals)]) for x in pxt.columns.to_list()]
        pxt.to_csv(self.outpath + '\\PCA_transform_{}_{}.txt'.format(str(self.thresholds), str(self.intervals)),
                   sep='\t', encoding='utf_8_sig')
        return pxt
