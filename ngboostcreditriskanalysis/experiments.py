from ngboostcreditriskanalysis.constants import cols, cols_with_missing_indicators
import pandas as pd
import numpy as np
from ngboostcreditriskanalysis.utils import (
    register_imputation,
    preprocess_extra,
    cost_score,
    generate_y_pred_with_custom_threshold,
    get_sample_weights,
    check_counts,
)
from sklearn.neighbors import LocalOutlierFactor
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE
import logging
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
from sklearn.metrics import roc_auc_score


class Experiment:
    """
    Base class to process experiments.
    """

    def __init__(self):
        self.seed = 2020
        self.raw_cols = cols
        self.cols_with_missing_indicators = cols_with_missing_indicators
        self.df = pd.read_csv("./data/cs-training.csv", usecols=self.raw_cols)
        self.filename_model = ""
        self.ngb_clf = None

    def clean_outliers(self, flag_filter=False):
        """ 
        Register imputations, identify outliers with LOF and clean them. Also it process an extra function to clean some outliers.

        Args:
            - df (DataFrame Object): dataframe to be processed
            - flag_filter (boolean): Flag that indicates if the process requres an extra cleanup of outliers.

        Returns dataframe without outliers
        """
        # df = preprocess_df(df.copy())
        df = register_imputation(self.df.copy())
        local_outlier_factor = LocalOutlierFactor(contamination=0.1)
        is_outlier = local_outlier_factor.fit_predict(df[cols[1:]]) == -1
        data_outlier_excluded = self.df.loc[~is_outlier, :]
        if flag_filter:
            preprocess_extra(data_outlier_excluded)
        return data_outlier_excluded

    def process_auc_roc(self, ngb_clf, X_train, y_train, X_test, y_test):
        # predict probabilities
        train_probs = ngb_clf.predict_proba(X_train)
        test_probs = ngb_clf.predict_proba(X_test)
        # keep probabilities for the positive outcome only
        train_probs = train_probs[:, 1]
        test_probs = test_probs[:, 1]
        # calculate scores
        train_auc = roc_auc_score(y_train, train_probs)
        test_auc = roc_auc_score(y_test, test_probs)

        return train_auc, test_auc

    def run_custom_grid_search(
        self,
        X_train_resampled,
        y_train_resampled,
        X_train,
        y_train,
        X_test,
        y_test,
        list_estimators,
        list_lr,
        list_base,
        results_filename,
    ):
        df_collector = pd.DataFrame(
            columns=[
                "hyperparams",
                "estimators",
                "learning_rate",
                "max_depth",
                "threshold",
                "cost",
                "count_zero",
                "count_one",
                "train_auc",
                "test_auc",
            ]
        )
        df_collector["hyperparams"] = df_collector["hyperparams"].astype("object")
        k = 0

        for estimator in list_estimators:
            for lr in list_lr:
                for baset in list_base:
                    ngb_clf = NGBClassifier(
                        Dist=Bernoulli,
                        verbose=True,
                        Base=baset,
                        n_estimators=estimator,
                        random_state=2020,
                        learning_rate=lr,
                        verbose_eval=0,
                    )
                    logging.info(f"Training the following model: {ngb_clf}")
                    ngb_clf.fit(
                        X_train_resampled,
                        y_train_resampled,
                        sample_weight=get_sample_weights(y_train, y_train_resampled),
                    )
                    for threshold in list(np.arange(0.2, 0.35, 0.05)):
                        df_collector.ix[str(k), "hyperparams"] = ngb_clf
                        df_collector.ix[str(k), "estimators"] = estimator
                        df_collector.ix[str(k), "learning_rate"] = lr
                        df_collector.ix[str(k), "max_depth"] = baset.max_depth
                        threshold = round(threshold, 2)
                        logging.info("k: " + str(k))
                        cost = self.process_cost_score(
                            ngb_clf, X_test, y_test, threshold
                        )
                        logging.info(threshold)
                        logging.info(cost)
                        logging.info("th " + str(threshold))
                        df_collector.ix[str(k), "threshold"] = threshold
                        df_collector.ix[str(k), "cost"] = cost
                        count_zero, count_one = check_counts(ngb_clf, X_test, threshold)
                        df_collector.ix[str(k), "count_zero"] = count_zero
                        df_collector.ix[str(k), "count_one"] = count_one

                        (
                            df_collector.ix[str(k), "train_auc"],
                            df_collector.ix[str(k), "test_auc"],
                        ) = self.process_auc_roc(X_train, y_train, X_test, y_test)
                        k += 1
                        logging.info("sumando k")
                        logging.info("---------------------------------")
                    del ngb_clf
        df_collector = df_collector.sort_values("cost")
        logging.info(f"Saving results in {results_filename}")
        df_collector.to_csv(f"results_filename", index=False)
        logging.info(f"Return best result")
        logging.info(df_collector.head(1))
        return df_collector.head(1)

    def save_model(self, filename):
        """
        Save ngboost model as pickle file.
        """
        self.filename_model = filename
        with open(self.filename_model, "wb") as file:
            pickle.dump(self.ngb_clf, file)

    def load_model(self, filename):
        """
        Load model from pickle file.
        """
        with open(self.filename_model, "rb") as file:
            ngb_clf = pickle.load(file)
        return ngb_clf

    def plot_roc_curve(self):
        pass

    def process_cost_score(self, ngb_clf, X_test, y_test, threshold):
        """
        Process cost metric without uncertainty. We can set a custom threshold to process it.
        """
        df_aux = pd.DataFrame(X_test, columns=self.cols_with_missing_indicators)
        df_aux["predicted"] = generate_y_pred_with_custom_threshold(
            ngb_clf, X_test, threshold
        )
        df_aux["real"] = list(y_test)
        df_aux["LoanPrincipal"] = df_aux.MonthlyIncome * 2
        return cost_score(df_aux.LoanPrincipal, df_aux.predicted, df_aux.real)


class FirstExperiment(Experiment):
    def __init__(self):
        self.best_model = self.load_model("best_model_first_experiment.pickle")

    def run_ml_pipeline(self, list_estimators, list_lr, list_base):
        """
        Run ML pipeline for First Experiment. You can specify estimators, learning_rate 
        and base estimators to run your NGBoost model.

        Returns best fitted model.
        """
        df, df_test = train_test_split(self.df, test_size=0.2, random_state=42)
        df = self.clean_outliers(df.copy(), True)
        X_train = df.drop(columns=["SeriousDlqin2yrs"]).values
        y_train = df.SeriousDlqin2yrs.values
        X_train_resampled, y_train_resampled = SMOTE(random_state=2019).fit_sample(
            X_train, y_train
        )
        logging.info("Resampled dataset shape {}".format(Counter(y_train_resampled)))
        df_test = register_imputation(df_test.copy())
        X_test = df_test.drop(columns=["SeriousDlqin2yrs"]).values
        y_test = df_test.SeriousDlqin2yrs.values

        best_result = self.run_custom_grid_search(
            X_train_resampled,
            y_train_resampled,
            X_train,
            y_train,
            X_test,
            y_test,
            list_estimators,
            list_lr,
            list_base,
        )
        ngb_clf = best_result.hyperparams
        ngb_clf.fit(
            X_train_resampled,
            y_train_resampled,
            sample_weight=get_sample_weights(y_train, y_train_resampled),
        )
        return ngb_clf

    class SecondExperiment(Experiment):
        def __init__(self):
            self.best_model = self.load_model(
                "best_model_second_experiment_without_uncertainty.pickle"
            )
            self.best_model = self.load_model(
                "best_model_second_experiment_without_uncertainty_with_degradation.pickle"
            )
            self.best_model = self.load_model("best_model_second_experiment.pickle")

        def run_ml_pipeline_without_uncertainty(self):
            pass

        def run_ml_pipeline_without_uncertainty_with_degradation(self):
            pass

        def run_ml_pipeline_with_uncertainty_with_degradation(self):
            pass

