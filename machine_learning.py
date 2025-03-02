import pandas as pd
from warnings import filterwarnings
from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pickle
from imblearn.over_sampling import BorderlineSMOTE

bsmote = BorderlineSMOTE(k_neighbors=5, random_state=2)
filterwarnings('ignore')

# Hyperparameter settings for 12 ML classifiers
ML_MODELS = [
    ('LR', LogisticRegression(max_iter=10000, tol=0.1)),
    ('LDA', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
    ('QDA', QuadraticDiscriminantAnalysis(store_covariance=True)),
    ('ET', ExtraTreesClassifier(n_estimators=10, min_samples_split=2, max_features='sqrt')),
    ('GBC', GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0, max_depth=4)),
    ('ABC', AdaBoostClassifier(n_estimators=400)),
    ('KNN', KNeighborsClassifier(n_neighbors=6)),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('RSVM', SVC(C=0.7, class_weight='balanced', kernel='rbf', cache_size=300, gamma='auto', probability=True)),
    ('RFC', RandomForestClassifier(min_samples_split=2, n_estimators=10, max_features='sqrt')),
    ('MLP', MLPClassifier(random_state=1, max_iter=3000, hidden_layer_sizes=(100, 100), batch_size=100,
                          solver='adam', warm_start=False, early_stopping=True, n_iter_no_change=100,
                          validation_fraction=0.2))
]

# Transform to a dictionary
ML_MODEL_DICT = dict(zip([model[0] for model in ML_MODELS], [model[1] for model in ML_MODELS]))
SCORE_SET = ('accuracy', 'precision', 'average_precision', 'recall', 'f1', 'roc_auc')

# Function for ML training without any data balancing strategy


def mlss(data: pd.DataFrame, data1: pd.DataFrame, outpath: str, cols: list = None,
         var: list = None, models: list = ML_MODELS, shuffle: bool = True,
         model_output: bool = True, model_name: str = None, prefix: str = '') -> tuple:
    """
    Perform machine learning training without any data balancing method and evaluation on a given dataset.

    Args:
        data: A Pandas DataFrame containing the feature values and disease information of the training cohort.
        data1: A Pandas DataFrame containing the feature values and disease information of the test cohort.
        outpath: The absolute path where all outputs, including models and figures, will be saved.
        cols: A list of columns (features) to be used for machine learning.
        var: A list of disease labels to be used for machine learning.
        models: A list of 12 ML classifiers with pre-defined hyperparameter settings.
        shuffle: A boolean indicating whether to shuffle the data before splitting it into training and test sets.
        model_output: A boolean indicating whether to save the trained ML models.
        model_name: A string used to customize the naming of saved models.
        prefix: A string used to customize the naming of saved models.

    Returns:
        A tuple containing the training metrics, test accuracy, and predicted outcomes and probabilities.
    """
    data = data.astype({'Gender_no': 'category'})
    data1 = data1.astype({'Gender_no': 'category'})
    pdval = pd.DataFrame(columns=['Test_accuracy'])
    pmodel_n = pd.DataFrame(columns=[f'CV_{i}' for i in range(1, 6)])
    aval = data1[var]

    for name, model in models:
        for va in var:
            cv_results = model_selection.cross_validate(model, data[cols + ['Gender_no', 'Age']], data[va],
                                                        cv=StratifiedKFold(n_splits=5, shuffle=shuffle),
                                                        scoring=SCORE_SET)
            for sc in SCORE_SET:
                pmodel_n.loc[name + '_' + model_name + '_' + sc] = cv_results[f'test_{sc}']
            if model_output:
                model.fit(data[cols + ['Gender_no', 'Age']], data[va])
                with open(outpath + f'\\{prefix}_{name}_{model_name}.pickle', 'wb') as f:
                    pickle.dump(model, f)
            for ix in data1.index:
                aval.loc[ix, f'Prediction_outcome_{name}_{model_name}'] = model.predict(
                    data1.loc[ix][cols + ['Gender_no', 'Age']].to_numpy().reshape(1, -1)).flatten()
                aval.loc[ix, [f'{p}_{name}_{model_name}' for p in ['PPP', 'NPP']]] = model.predict_proba(
                    data1.loc[ix][cols + ['Gender_no', 'Age']].to_numpy().reshape(1, -1)).flatten()
            x_test = data1[cols + ['Gender_no', 'Age']].to_numpy()
            y = data1[va].to_numpy()
            pdval.loc[name + '_' + model_name] = model.score(x_test, y)

    return pmodel_n, pdval, aval


# Function for ML training with SMOTE data balancing strategy


def mlss_upo(data: pd.DataFrame, data1: pd.DataFrame, outpath: str, cols: list = None,
             var: list = None, models: list = ML_MODELS, model_output: bool = True,
             model_name: str = 'OC_OPC', prefix: str = '') -> tuple:
    """
    Perform machine learning training using SMOTE data balancing method and evaluation on a given dataset.

    Args:
        data: A Pandas DataFrame containing the feature values and disease information of the training cohort.
        data1: A Pandas DataFrame containing the feature values and disease information of the test cohort.
        outpath: The absolute path where all outputs, including models and figures, will be saved.
        cols: A list of columns (features) to be used for machine learning.
        var: A list of disease labels to be used for machine learning.
        models: A list of 12 ML classifiers with pre-defined hyperparameter settings.
        model_output: A boolean indicating whether to save the trained ML models.
        model_name: A string used to customize the naming of saved models.
        prefix: A string used to customize the naming of saved models.

    Returns:
        A tuple containing the training metrics, test accuracy, and predicted outcomes and probabilities.
    """
    x_res, y_res = bsmote.fit_resample(data[cols + ['Age', 'Gender_no']], data[var[0]])
    ocdata1 = pd.concat([x_res, y_res], axis=1)
    pdval = pd.DataFrame(columns=['Test_accuracy'])
    pmodel_n = pd.DataFrame(columns=[f'CV_{i}' for i in range(1, 6)])
    aval = data1[var]

    for name, model in models:
        for va in var:
            cv_results = model_selection.cross_validate(model, ocdata1[cols + ['Gender_no', 'Age']],
                                                        ocdata1[va], scoring=SCORE_SET)
            for sc in SCORE_SET:
                pmodel_n.loc[name + '_' + model_name + '_' + sc] = cv_results[f'test_{sc}']
            if model_output:
                model.fit(ocdata1[cols + ['Gender_no', 'Age']], ocdata1[va])
                with open(outpath + f'\\{prefix}_{name}_{model_name}.pickle', 'wb') as f:
                    pickle.dump(model, f)
            for ix in data1.index:
                aval.loc[ix, f'Prediction_outcome_{name}_{model_name}'] = model.predict(
                    data1.loc[ix][cols + ['Gender_no', 'Age']].to_numpy().reshape(1, -1)).flatten()
                aval.loc[ix, [f'{p}_{name}_{model_name}' for p in ['PPP', 'NPP']]] = model.predict_proba(
                    data1.loc[ix][cols + ['Gender_no', 'Age']].to_numpy().reshape(1, -1)).flatten()
            x_test = data1[cols + ['Gender_no', 'Age']].to_numpy()
            y = data1[va].to_numpy()
            pdval.loc[name + '_' + model_name] = model.score(x_test, y)

    return pmodel_n, pdval, aval
