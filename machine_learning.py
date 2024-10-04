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
import pickle #packages for ML models save and read

filterwarnings('ignore')

#Hyperparameter settings for 12 ML classifiers
models = [('LR', LR(max_iter=10000, tol=0.1)), ('LDA', LDA(solver='lsqr', shrinkage='auto')),
          ('QDA', QDA(store_covariance=True)),
          ('ET', ETC(n_estimators=10, min_samples_split=2, max_features="sqrt")),
          ('GBC', GBC(n_estimators=1000, learning_rate=1.0, max_depth=4)),
          ('ABC', ABC(n_estimators=400)),
          ('KNN', KN(n_neighbors=6)), ('CART', DTC()), ('NB', GNB()),
          ('RSVM',SVC(C=.7, class_weight='balanced', kernel='rbf', cache_size=300, gamma='auto', probability=True)),
          ('RFC', RFC(min_samples_split=2, n_estimators=10, max_features="sqrt")),
          ('MLP', MLPClassifier(random_state=1, max_iter=3000, hidden_layer_sizes=(100, 100), batch_size=100,
                                solver='adam', warm_start=False, early_stopping=True, n_iter_no_change=100,
                                validation_fraction=0.2))]

# transform to a dictionary
dmodels=dict(zip([i[0] for i in models],[i[-1] for i in models]))
score_set = ('accuracy', 'precision', 'average_precision', 'recall', 'f1', 'roc_auc')

#function for ML training without any data balancing strategy

def MLSS(data, data1,outpath,cols=[], var=['OC_OPC'], models=models, shuffle=True, model_output=True,
         model_name='OC_OPC',prefix=''):
    # data refers to the dataframe contains both feature values and disease information of training cohort
    # data1 refers to the dataframe contains both feature values and disease information of test cohort
    # cols controls the PCA-transformed speech omics feature sets used for machine learning
    # outpath controls where all outputs including models and figures to save
    # var refers to the disease label used for machine learning
    # models refers to 12 ML classifier which have been ready in terms of hyperparameter settings
    # model_output controls whether to save the trained ML classifiers
    # model_name and prefix allow for custom naming for saved model
    data = data.astype({'Gender_no': 'category'})
    data1 = data1.astype({'Gender_no': 'category'})
    pdval = pd.DataFrame(columns=['Test_accuracy'])
    # this dataframe is used to record the test accuracy of each ML model in test cohorts
    pmodel_n = pd.DataFrame(columns=['CV_' + str(s) for s in range(1, 6)])
    # this dataframe is used to record the training metrics of each ML model in training cohorts
    # stratified 5-fold cross-validation is adopt in this study
    aval=data1[var]
    # this dataframe is used to record individual prediction outcome and probability
    for name, model in models:
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
    return pmodel_n, pdval, aval

#package for SMOTE
from imblearn.over_sampling import BorderlineSMOTE
bsmote=BorderlineSMOTE(k_neighbors=5, random_state=2)

#function for ML training with SMOTE data balancing strategy
def MLSS_upo(data, data1, outpath, cols=[], var=['OC_OPC'], models=models, model_output=True,
             model_name='OC_OPC', prefix=''):
    # data refers to the dataframe contains both feature values and disease information of training cohort
    # data1 refers to the dataframe contains both feature values and disease information of test cohort
    # outpath controls where all outputs including models and figures to save
    # cols controls the PCA-transformed speech omics feature sets used for machine learning
    # var refers to the disease label used for machine learning
    # models refers to 12 ML classifier which have been ready in terms of hyperparameter settings
    # model_output controls if save the trained ML classifiers
    # model_name and prefix allow for custom naming for saved model
    X_res, y_res = bsmote.fit_resample(data[cols + ['Age', 'Gender_no']], data[var[0]])
    ocdata1 = pd.concat([X_res, y_res], axis=1)
    pdval = pd.DataFrame(columns=['Test_accuracy'])
    # this dataframe is used to record the test accuracy of each ML model in test cohorts
    pmodel_n = pd.DataFrame(columns=['CV_' + str(s) for s in range(1, 6)])
    # this dataframe is used to record the training metrics of each ML model in training cohorts
    # stratified 5-fold cross-validation is adopt in this study
    aval = data1[var]
    # this dataframe is used to record individual prediction outcome and probability
    for name, model in models:
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
    return pmodel_n, pdval, aval
