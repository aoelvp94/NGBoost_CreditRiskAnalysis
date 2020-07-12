"""
REPO OF USEFUL FUNCTIONS
"""

import pandas as pd
import numpy as np
from constants import cols

# ngboost and modelling libraries
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils import class_weight
from sklearn.inspection import permutation_importance


def register_imputation(df):
    """
    Register imputations of certain df
    Args:
        - df (DataFrame): Dataframe to be computed
    Return df with filled values and booleans that indicate if each row was changed
    """
    for c in cols[1:]:
        # df[f"{c}_amputado"] = df[f"{c}"].fillna(0)
        df[f"{c}_imputed"] = df[f"{c}"].fillna(0)
        df[f"dummy_{c}"] = (df[f"{c}"] != df[f"{c}_imputed"]).astype(int)
        del df[f"{c}"]
        df.columns = df.columns.str.replace(f"{c}_imputed", f"{c}")
    return df


def preprocess_df(df):
    """
    Preprocess df imputing certain columns such as MonthlyIncome and NumberOfDependencies
    
    Args:
        - df (DataFrame object): df to be computed
 
    for col in [
        "RevolvingUtilizationOfUnsecuredLines",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ]:
        df.loc[(df["age"] < 18) | (df["age"] >= 60), col] = 18
    """

    df["MonthlyIncome"].fillna(0, inplace=True)
    df["NumberOfDependents"].fillna(0, inplace=True)
    # df.loc[(df["age"] > 60), "age"] = 60
    return df


def clean_outliers(df, flag_filter=False):
    """ 
    Register imputations, identify outliers with LOF and clean them. Also it process an extra function to clean some outliers.

    Args:
        - df (DataFrame Object): dataframe to be processed
        - flag_filter (boolean): Flag that indicates if the process requres an extra cleanup of outliers.

    Returns dataframe without outliers
    """
    # df = preprocess_df(df.copy())
    df = register_imputation(df.copy())
    local_outlier_factor = LocalOutlierFactor(contamination=0.1)
    is_outlier = local_outlier_factor.fit_predict(df[cols[1:]]) == -1
    data_outlier_excluded = df.loc[~is_outlier, :]
    if flag_filter:
        preprocess_extra(data_outlier_excluded)
    return data_outlier_excluded


def preprocess_extra(df):
    """
    Extra cleanup process of some outliers.
    """
    df.loc[
        (df["RevolvingUtilizationOfUnsecuredLines"] > 1),
        "RevolvingUtilizationOfUnsecuredLines",
    ] = 0
    df.loc[(df["DebtRatio"] > 10), "DebtRatio"] = 0
    df.loc[(df["age"] < 18), "age"] = 0


def scaling_values_df(df):
    """
    Scale values of df
    Args:
        - df (DataFrame object): Dataframe to be processed
    """
    mms = MinMaxScaler()
    df[cols[1:]] = mms.fit_transform(df[cols[1:]])
    return df


def get_sample_weights(y_train, y_train_resampled):
    """
    Get weights for each class.

    Args:
        - y_train (np.array): target of train df
        - y_train_resampled (np.array): target of resampled train df

    Return array for each target value with their weights.
    """
    class_weights = class_weight.compute_class_weight(
        "balanced", np.unique(y_train), y_train
    )
    return np.asarray(
        [class_weights[0] if x == 0 else class_weights[1] for x in y_train_resampled],
        dtype=np.float32,
    )


# METRICS


def process_unit_cost(x, rate):
    """
    Process the unit cost for each observation, according to our cost matrix.
    
    Args:
        - x (pd.Series): data to be identified with name column (real, predicted or LoanPrincipal)
        - rate (float): interest rate
        
    Returns for each case his cost value.
    """
    cost = (
        x["predicted"] * (1 - x["real"]) * x["LoanPrincipal"] * rate
        + (1 - x["predicted"]) * x["real"] * x["LoanPrincipal"] * (1 + rate)
        - x["predicted"] * (x["real"]) * x["LoanPrincipal"] * rate
        + (1 - x["predicted"]) * (1 - x["real"]) * 0
    )
    return cost


def process_learning_unit_cost(x, alpha):
    """
    Process the learning unit cost for each observation.
    
    Args:
        - x (pd.Series): data to be identified with name column (predicted, proba_predicted)
        - alpha (int): hyperparam to set a weight of learning
        
    Returns for each case his cost value.
    """
    return (
        alpha * (1 - x["predicted"]) * x["proba_predicted"] * (1 - x["proba_predicted"])
    )


def cost_score(loan, y_pred, y_true):
    """
    From input data, generates auxiliar dataframe in order to apply process_unit_cost for each row and then summarize that.
    
    Args:
        - loan (pd.Series): data about the requested amount of money
        - y_pred (pd.Series): list of predictions
        - y_true (pd.Series): list of true values
        
    Returns sum of unit costs
    """
    aux_df = pd.DataFrame(
        data={"LoanPrincipal": loan, "predicted": y_pred, "real": y_true}
    )
    return sum(aux_df.apply(lambda x: process_unit_cost(x, 0.01), axis=1))


def calculate_learning_cost(y_pred, proba_pred, alpha):
    """
    From input data, generates auxiliar dataframe in order to apply process_learning_unit.
    Args:
        - y_pred (pd.Series): list of predictions
        - y_proba (pd.Series): list of proba for each prediction
        - alpha (int): hyperparam to set a weight of learning
        
    Returns sum of learning unit costs
    """
    aux_df = pd.DataFrame(data={"predicted": y_pred, "proba_predicted": proba_pred})
    return sum(aux_df.apply(lambda x: process_learning_unit_cost(x, alpha), axis=1))


def calculate_cost_score_with_learning(cost_score_without_learning, learning_cost):
    """
    Process cost score with learning.
    """
    return cost_score_without_learning + learning_cost


def generate_y_pred_with_custom_threshold(model, x_data, threshold):
    """
    Generates new y_predictions according to a threshold.
    
    Args:
        - model (NGBoost model): NGBoost model that was trained.
        - x_data (np.ndarray): Data on which we predict probabilities
        - threshold (float): Float value to determine 1 or 0 for new predictions
    
    Returns updated y_predictions
    """
    y_predictions = model.predict_proba(x_data)
    y_pred = []
    count_zero = 0
    count_one = 0
    for i in range(len(list(y_predictions))):
        if y_predictions[i][1] > threshold:
            y_pred.append(1)
            count_one += 1
        else:
            y_pred.append(0)
            count_zero += 1
    print("count_zero " + str(count_zero))
    print("count_one " + str(count_one))
    return y_pred


def check_counts(model, x_data, threshold):
    y_predictions = model.predict_proba(x_data)
    y_pred = []
    count_zero = 0
    count_one = 0
    for i in range(len(list(y_predictions))):
        if y_predictions[i][1] > threshold:
            y_pred.append(1)
            count_one += 1
        else:
            y_pred.append(0)
            count_zero += 1
    return count_zero, count_one
