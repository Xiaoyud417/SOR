import os, shutil
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import openpyxl

from sor import SOR

cpath = 'D:\\npj_digital_med\\upload\\re_upload\\re_re_upload\\code'  # PLease change accordingly.
case = list(set([fn[4:21] for fn in os.listdir(cpath + '\\1.FormantTable')]))
path = cpath + '\\1.FormantTable'
outpath = cpath + '\\6.OmicsData'
pdr = pd.read_excel(cpath + '\\0.basic\\pdr.xlsx', index_col=0)

epo = SOR.SOR(path=path, outpath=outpath, case=case, thresholds=1, intervals=1, pdr=pdr)
pkde = epo.get_kde(kpath=cpath + '\\2.FormantExcel')
for th in tqdm(range(2, 10)):
    threshold = th / 10
    epo = SOR.SOR(path=path, outpath=outpath, case=case, thresholds=threshold, intervals=1, pdr=pdr)
    pdn = epo.get_pdn(ipath=cpath + '\\3.DerivedMetric')

    for thr in tqdm(range(10, 101, 10)):
        interval = thr
        epo = SOR.SOR(path=path, outpath=outpath, case=case, thresholds=threshold, intervals=interval, pdr=pdr)
        pdl = epo.get_pdl(pdn, spath=cpath + '\\4.SDRdata')
        npd0, pddata = epo.get_ALL(pdl, spath=cpath + '\\4.SDRdata')
        epo.PCA(npd0, pddata, ppath=cpath + '\\5.PCAmodels')
    time.sleep
