import numpy as np
import pandas as pd
from scipy import stats, interpolate

from sklearn.decomposition import PCA as PCAA
from sklearn.preprocessing import MinMaxScaler
import pickle

dv = dict(zip(['a', 'i', 'u'], ['fXBQhWf', 'fkEMYTo', 'fIHXxem']))

root_path = 'D:\\医院\\科研\\成果\\in_sub\\npj_digital_med\\upload\\re_upload\\re_re_upload\\code'


def s_kde(df, top_percent=1, cols=None, min_pairs=1, col_name='kde', inplace=False):
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


def get_fft(name, fpath):
    ppa = pd.read_excel(fpath + '\\' + name + '.xlsx', index_col=0, sheet_name='a_ori')
    ppi = pd.read_excel(fpath + '\\' + name + '.xlsx', index_col=0, sheet_name='i_ori')
    ppu = pd.read_excel(fpath + '\\' + name + '.xlsx', index_col=0, sheet_name='u_ori')

    return ppa, ppi, ppu


def FCRs(df1, df2, df3, name1=None, iter_num=1):
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
        FCR = (l3[1] + l1[1] + l2[0] + l3[0]) / (l2[1] + l1[0])
        F2R = l2[1] / l3[1]
        VSA = np.abs((l2[0] * (l1[1] - l3[1]) + l1[0] * (l3[1] - l2[1]) + l3[0] * (
                l2[1] - l1[1])) / 2)
        yield FCR, VSA, F2R
        count += 1


class SOR:
    def __init__(self, path, outpath, case, thresholds, intervals, pdr):
        self.path = path
        self.outpath = outpath
        self.thresholds = thresholds
        self.intervals = intervals
        self.case = case
        self.pdr = pdr

    def get_kde(self, kpath):
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

    def dense_discrete(self, data=np.ones(100), Range=None, index=' '):
        if Range is None:
            Range = [0, 1]
        logarr = sorted([x for x in np.logspace(Range[0], Range[1], num=self.intervals, endpoint=True)])
        ddata = pd.DataFrame(columns=logarr)
        for j in reversed(range(self.intervals)):
            ddata.loc[index, logarr[j]] = data[data > float(logarr[j])].shape[0]
            data = data[data <= float(logarr[j])]
        return ddata

    def get_pdn(self, ipath):
        pdn = pd.DataFrame(columns=[i + '_' + str(j) for i in ['FCR', 'VSA', 'F2R'] for j in range(10000)])
        sik = {}
        for name in self.case:
            ppa, ppi, ppu = get_fft(name, fpath=root_path + '\\2.FormantExcel')
            ffta = ppa.sort_values('kde', ascending=False).iloc[:int(np.ceil(self.thresholds * ppa.shape[0])), :]
            ffti = ppi.sort_values('kde', ascending=False).iloc[:int(np.ceil(self.thresholds * ppi.shape[0])), :]
            fftu = ppu.sort_values('kde', ascending=False).iloc[:int(np.ceil(self.thresholds * ppu.shape[0])), :]

            fset = FCRs(df1=ffta, df2=ffti, df3=fftu, name1=['F1(Hz)', 'F2(Hz)'], iter_num=10000)

            iter = 0
            nv = np.zeros((10000, 3))
            try:
                while iter < 10000:
                    fcr1, vsa1, f2r1 = next(fset)
                    nv[iter, 0] = fcr1
                    nv[iter, 1] = vsa1
                    nv[iter, 2] = f2r1
                    iter += 1
            except:
                sik[name] = iter
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
                    Y = np.sort(y_smooth)[::-1]
                    pdn.loc[i, [s for s in pdn.columns.to_list() if s.startswith(y)]] = Y

        pdn.to_csv(ipath + '\\data10000_{}.txt'.format(str(self.thresholds)), sep='\t', encoding='utf_8_sig')
        return pdn

    def get_pdl(self, pdn, spath):
        pl = []
        for xx, name in enumerate(self.case):
            pl0 = []
            ppa, ppi, ppu = get_fft(name, fpath=root_path + '\\2.FormantExcel')
            vfcr = pdn.iloc[xx, :10000].to_numpy()
            vvsa = pdn.iloc[xx, 10000:20000].to_numpy()
            vf2r = pdn.iloc[xx, 20000:].to_numpy()
            pl0.append(self.dense_discrete(data=vfcr, Range=[np.log10(self.pdr.loc['FCR', 'Minimum']),
                                                             np.log10(self.pdr.loc['FCR', 'Maximum'])], index=name))
            pl0.append(self.dense_discrete(data=vvsa, Range=[np.log10(self.pdr.loc['VSA', 'Maximum']), 1], index=name))
            pl0.append(self.dense_discrete(data=vf2r, Range=[np.log10(self.pdr.loc['F2R', 'Minimum']),
                                                             np.log10(self.pdr.loc['F2R', 'Maximum'])], index=name))
            for p in ['F1(Hz)', 'F2(Hz)', 'F3(Hz)']:
                for vv, uu in zip(['a', 'i', 'u'], [
                    ppa.sort_values('kde', ascending=False).iloc[:int(np.ceil(self.thresholds * ppa.shape[0])), :],
                    ppi.sort_values('kde', ascending=False).iloc[:int(np.ceil(self.thresholds * ppi.shape[0])), :],
                    ppu.sort_values('kde', ascending=False).iloc[:int(np.ceil(self.thresholds * ppu.shape[0])), :]]):
                    pl0.append(self.dense_discrete(data=uu[p].to_numpy(),
                                                   Range=[np.log10(self.pdr.loc[p[:2] + vv, 'Minimum']),
                                                          np.log10(self.pdr.loc[p[:2] + vv, 'Maximum'])], index=name))
            for p in ['B1(Hz)', 'B2(Hz)', 'B3(Hz)']:
                for vv, uu in zip(['a', 'i', 'u'], [
                    ppa.sort_values('kde_' + p[:2], ascending=False).iloc[:int(np.ceil(self.thresholds * ppa.shape[0])),
                    :],
                    ppi.sort_values('kde_' + p[:2], ascending=False).iloc[:int(np.ceil(self.thresholds * ppi.shape[0])),
                    :],
                    ppu.sort_values('kde_' + p[:2], ascending=False).iloc[:int(np.ceil(self.thresholds * ppu.shape[0])),
                    :]]):
                    pl0.append(self.dense_discrete(data=uu[p].to_numpy(),
                                                   Range=[np.log10(self.pdr.loc[p[:2] + vv, 'Maximum']), 0.1],
                                                   index=name))
            pl.append(pd.concat(pl0, axis=1))
        pdl = pd.concat(pl)
        pdl.to_csv(spath + '\\Density_{}_{}.txt'.format(str(self.thresholds), str(self.intervals)), sep='\t',
                   encoding='utf_8_sig')
        return pdl

    def get_ALL(self, pdl, spath):
        mms = MinMaxScaler()
        npd0 = []
        for i in range(21):
            npd0.append(
                mms.fit_transform(pdl.iloc[:, self.intervals * i:self.intervals * (i + 1)]))  # 将各个特征缩放到[0,1]的范围内
        npd = np.hstack(npd0)
        pddata = pd.DataFrame(npd, index=pdl.index.to_list(), columns=pdl.columns.to_list())
        pddata.to_csv(spath + '\\SDR_{}_{}.txt'.format(str(self.thresholds), str(self.intervals)), sep='\t',
                      encoding='utf_8_sig')
        # npd0 为标准化后数据， pddate为横向叠加后的npd0
        return npd0, pddata

    def PCA(self, npd0, pddata, ppath):
        mms = MinMaxScaler()
        # Define a pipeline to search for the best combination of PCA truncation
        # and classifier regularization.
        pca = PCAA()

        pnpr = pd.DataFrame(columns=['Accumulated ratio of ' + i for i in
                                     ['ALL', 'FCR', 'VSA', 'F2R'] + [x + y for x in ['F1', 'F2', 'F3', 'B1', 'B2', 'B3']
                                                                     for
                                                                     y in ['a', 'i', 'u']]])
        acn = np.ones(22, dtype=int)
        x_transform = []
        for k, (m, n) in enumerate(zip(pnpr.columns.to_list(), [pddata] + npd0)):
            pca.fit(n)
            npr = pca.explained_variance_ratio_  # 载荷
            for i in range(1, npr.shape[0] + 1):
                pnpr.loc[i, m] = np.sum(npr[:i])
            acn[k] = pnpr.loc[pnpr[m] >= 0.95].index.to_list()[0]
            X_pca = pca.fit_transform(n)[:, :acn[k]]
            ppca = pd.DataFrame(X_pca, index=pddata.index.to_list(),
                                columns=['PCA_' + str.split(m, ' ')[-1] + '_' + str(i) for i in range(1, acn[k] + 1)])
            x_transform.append(ppca)

            with open(ppath + '\\PCA_{}_{}_{}.pickle'.format(str(self.thresholds), str(self.intervals),
                                                             str.split(m, ' ')[-1]), 'wb') as f:
                pickle.dump(pca, f)

        pxt = pd.concat(x_transform, axis=1)
        pxt.columns=['_'.join([x, str(self.thresholds), str(self.intervals)]) for x in pxt.columns.to_list()]
        pxt.to_csv(self.outpath + '\\PCA_transform_{}_{}.txt'.format(str(self.thresholds), str(self.intervals)),
                   sep='\t', encoding='utf_8_sig')
        return pxt
