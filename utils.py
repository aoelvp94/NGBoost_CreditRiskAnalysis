"""
REPO OF USEFUL FUNCTIONS
"""

# ngboost and modelling libraries
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils import class_weight
from sklearn.inspection import permutation_importance

# data manipulation libraries
import pandas as pd
import numpy as np

# data viz libraries
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from constants import cols
from sklearn.utils import class_weight


def correlation_heatmap(df):
    """
    Plot a correlation heatmap for the entire dataframe
    
    Args:
        - df (DataFrame object): dataframe to be illustrated
    """
    heatmap = go.Heatmap(
        z=df.corr(method="pearson").as_matrix(),
        x=df.columns,
        y=df.columns,
        colorbar=dict(title="Pearson Coefficient"),
        colorscale="Reds",
    )

    layout = go.Layout(title="Matriz de correlaciones")

    fig = go.Figure(data=[heatmap], layout=layout)
    iplot(fig)


def register_amputation(df):
    """
    Register amputations of df
    Args:
        - df (DataFrame): Dataframe to be computed
    Return df with filled values and booleans that indicate if each row was changed
    """
    for c in df.columns:
        df[f"{c}_amputado"] = df[f"{c}"].fillna(0)
        df[f"dummy_falta_{c}"] = (df[f"{c}"] != df[f"{c}_amputado"]).astype(int)
    return df


def preprocess_df(df):
    """
    Preprocess df imputing certain columns such as MonthlyIncome and NumberOfDependencies
    
    Args:
        - df (DataFrame object): df to be computed
    """

    """
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
    df.loc[(df["RevolvingUtilizationOfUnsecuredLines"]>1), "RevolvingUtilizationOfUnsecuredLines" ] = 0
    df.loc[(df["DebtRatio"]>10), "DebtRatio" ] = 0
    df.loc[(df["age"] < 18), "age"] = 0
    # df.loc[(df["age"] > 60), "age"] = 60


def clean_outliers(df):

    local_outlier_factor = LocalOutlierFactor(contamination=0.1)
    preprocess_df(df)
    is_outlier = local_outlier_factor.fit_predict(df[cols[1:]]) == -1
    data_outlier_excluded = df.loc[~is_outlier, :]
    return data_outlier_excluded


def scaling_values_df(df):
    """
    Scale values of df
    Args:
        - df (DataFrame object): Dataframe to be processed
    """
    mms = MinMaxScaler()
    df[cols[1:]] = mms.fit_transform(df[cols[1:]])
    return df


def plot_target_balance(target_value_counts):
    """
    Plot target balance
    
    Args:
        - target_value_counts(pd.Series object): Serie that contains counts of each instance of target variable.
    """
    data = [
        go.Bar(
            x=["Responsible (target=0)"],
            y=[target_value_counts[0]],
            marker=dict(color="cornflowerblue"),
            name="Responsible (target=0)",
        ),
        go.Bar(
            x=["Delinquent (target=1)"],
            y=[target_value_counts[1]],
            marker=dict(color="darksalmon"),
            name="Delinquent (target=1)",
        ),
    ]
    layout = dict(
        title="Proporción del target",
        xaxis=dict(title="Target value"),
        yaxis=dict(title="Count"),
    )
    fig = dict(data=data, layout=layout)
    iplot(fig)


def plot_trace_line(df_target_zero, df_target_one, column_to_op, op_name):
    """
    Plot lines with 'age' column as x axis and custom column (to compute sum/avg).
    
    Args:
        - df_target_zero (DataFrame Object): rows of df that contain target = 0
        - df_target_ one (DataFrame Object): rows of df that contain target = 1
        - column_to_op (string): name of column to be computed
        - op_name (string): Operation to be computed. For example: "Sum" or "Avg" 
    """
    if op_name == "Sum":
        df_target_zero_op = df_target_zero.groupby("age").sum()
        df_target_one_op = df_target_one.groupby("age").sum()
    if op_name == "Avg":
        df_target_zero_op = df_target_zero.groupby("age").mean()
        df_target_one_op = df_target_one.groupby("age").mean()

    # Create and style traces
    trace0 = go.Scatter(
        x=df_target_zero_op.index,
        y=df_target_zero_op[column_to_op],
        name="Responsible (target=0)",
        line=dict(color="rgb(167, 103, 4)", width=4),
    )
    trace1 = go.Scatter(
        x=df_target_one_op.index,
        y=df_target_one_op[column_to_op],
        name="Delinquent (target=1)",
        line=dict(color="rgb(32, 205, 119)", width=4),
    )
    data = [trace0, trace1]

    # Edit the layout
    layout = dict(
        title=f"{op_name} of {column_to_op} according to the age",
        xaxis=dict(title="Age"),
        yaxis=dict(title=f"{op_name} of {column_to_op}"),
    )

    fig = dict(data=data, layout=layout)
    iplot(fig)


def plot_trace_line2(df_target_zero, df_target_one, column_to_op, op_name):
    """
    Plot lines with 'age' column as x axis and custom column (to compute sum/avg).
    
    Args:
        - df_target_zero (DataFrame Object): rows of df that contain target = 0
        - df_target_ one (DataFrame Object): rows of df that contain target = 1
        - column_to_op (string): name of column to be computed
        - op_name (string): Operation to be computed. For example: "Sum" or "Avg" 
    """
    if op_name == "Sum":
        df_target_zero_op = df_target_zero.groupby("age").sum()
        df_target_one_op = df_target_one.groupby("age").sum()
    if op_name == "Avg":
        df_target_zero_op = df_target_zero.groupby("age").mean()
        df_target_one_op = df_target_one.groupby("age").mean()

    # Create and style traces
    trace0 = go.Scatter(
        x=df_target_zero_op.index,
        y=df_target_zero_op[column_to_op],
        name="Responsible (target=0)",
        line=dict(color="rgb(86,157, 242)", width=4),
    )
    trace1 = go.Scatter(
        x=df_target_one_op.index,
        y=df_target_one_op[column_to_op],
        name="Delinquent (target=1)",
        line=dict(color="rgb(239, 102, 75)", width=4),
    )
    data = [trace0, trace1]

    # Edit the layout
    layout = dict(
        title=f"{op_name} of {column_to_op} according to the age",
        xaxis=dict(title="Age"),
        yaxis=dict(title=f"{op_name} of {column_to_op}"),
    )

    fig = dict(data=data, layout=layout)
    iplot(fig)


def plot_scatter_matrix(df):
    """
    Plot scatter matrix
    
    Args:
        - df (DataFrame object): Dataframe to be shown
    """
    textd = [
        "Responsible (target=0)" if target == 0 else "Delinquent (target=1)"
        for target in df["SeriousDlqin2yrs"]
    ]

    fig = go.Figure(
        data=go.Splom(
            dimensions=[
                dict(
                    label="RevUtilOfUnsecLines",
                    values=df["RevolvingUtilizationOfUnsecuredLines"],
                ),
                dict(label="age", values=df["age"]),
                dict(
                    label="NTime30-59Days",
                    values=df["NumberOfTime30-59DaysPastDueNotWorse"],
                ),
                dict(label="DebtRatio", values=df["DebtRatio"]),
                dict(label="MonthlyIncome", values=df["MonthlyIncome"]),
                dict(
                    label="NOpenCreditLinesLoans",
                    values=df["NumberOfOpenCreditLinesAndLoans"],
                ),
                dict(label="NTimes90DaysLate", values=df["NumberOfTimes90DaysLate"]),
                dict(
                    label="NRealEstateLoansLines",
                    values=df["NumberRealEstateLoansOrLines"],
                ),
                dict(
                    label="NTime60-89Days",
                    values=df["NumberOfTime60-89DaysPastDueNotWorse"],
                ),
                dict(label="NDepend", values=df["NumberOfDependents"]),
            ],
            marker=dict(
                color=df["SeriousDlqin2yrs"],
                size=5,
                colorscale="Bluered",
                line=dict(width=0.5, color="rgb(230,230,230)"),
            ),
            text=textd,
            diagonal=dict(visible=False),
        )
    )

    fig.update_layout(
        title={
            "text": "Scatterplot Matrix of Dataset",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        dragmode="select",
        width=1000,
        height=1000,
        hovermode="closest",
        font=dict(size=7, color="#7f7f7f"),
    )

    fig.show()


def plot_feature_importances(features, clf):
    """
    Show feature importances plot.
    
    Args:
        - features (list of strings): list of name columns
        - clf (XGboost model): XGboost model that was trained
    """
    trace1 = go.Bar(
        y=features,
        x=clf.feature_importances_[0],
        marker=dict(color="cornflowerblue", opacity=1),
        orientation="h",
    )

    data = [trace1]
    layout = go.Layout(
        barmode="group",
        margin=go.layout.Margin(l=120, r=50, b=100, t=100, pad=4),
        title="Feature importances",
        xaxis=dict(title="Importance"),
        yaxis=dict(title="Features"),
    )
    fig = dict(data=data, layout=layout)
    iplot(fig)


def visualize_roc_curve(model, X_train, y_train, X_test, y_test):
    """
    Plot roc curve
    Args
        - model (NGBClassifier object): NGBoost model
        - X_test (numpy.ndarray): Data without target values
        - y_test (numpy.ndarray): Target values
    """
    # generate a no skill prediction (majority class)
    # ns_probs = [0 for _ in range(len(y_test))]
    # predict probabilities
    train_probs = model.predict_proba(X_train)
    test_probs = model.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    train_probs = train_probs[:, 1]
    test_probs = test_probs[:, 1]
    # calculate scores
    # ns_auc = roc_auc_score(y_test, ns_probs)
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    # summarize scores
    # print("No Skill: ROC AUC=%.3f" % (ns_auc))
    print("TRAIN: ROC AUC=%.3f" % (train_auc))
    print("TEST: ROC AUC=%.3f" % (test_auc))
    # calculate roc curves
    # ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    train_fpr, train_tpr, _ = roc_curve(y_train, train_probs)
    test_fpr, test_tpr, _ = roc_curve(y_test, test_probs)

    trace0 = go.Scatter(
        x=train_fpr, y=train_tpr, name="TRAIN ROC", line=dict(color="red", width=2)
    )
    trace1 = go.Scatter(
        x=test_fpr, y=test_tpr, name="TEST ROC", line=dict(color="blue", width=2)
    )
    trace2 = go.Scatter(
        x0=0,
        x=[0, 1],
        y0=0,
        y=[0, 1],
        name="RANDOM ROC",
        line=dict(color="grey", width=2),
    )

    data = [trace0, trace1, trace2]

    # Edit the layout
    layout = dict(title="ROC curve", xaxis=dict(title="FPR"), yaxis=dict(title="TPR"))

    fig = dict(data=data, layout=layout)

    iplot(fig)


def visualize_permutation_feature_importance(model, X_train, y_train):
    # plot feature importance
    results = permutation_importance(model, X_train, y_train)
    importance = results.importances_mean
    for i, v in enumerate(importance):
        print("Feature: %0d, Score: %.5f" % (i, v))
    trace1 = go.Bar(
        x=cols[1:], y=importance, marker=dict(color="cornflowerblue", opacity=1)
    )

    data = [trace1]
    layout = go.Layout(
        barmode="group",
        margin=go.layout.Margin(l=120, r=50, b=100, t=100, pad=4),
        title="Permutation Feature importances",
        yaxis=dict(title="Importance"),
        xaxis=dict(title="Features"),
    )
    fig = dict(data=data, layout=layout)
    iplot(fig)


def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = "#FAA8AB" if val > 0 else None
    return "background-color: %s" % color


def process_unit_cost(x, rate):
    """
    Process the unit cost for each observation, according to our cost matrix.
    
    Args:
        - x: data to be identified with name column (real, predicted or LoanPrincipal)
        - rate: interest rate
        
    Returns for each case his cost value.
    """
    if (x["predicted"] == 1) & (x["real"] == 0):
        return x["LoanPrincipal"] * rate
    elif (x["predicted"] == 0) & (x["real"] == 1):
        return x["LoanPrincipal"]
    else:
        return 0


def cost_score(loan, y_pred, y_true):
    """
    From input data, generates auxiliar dataframe in order to apply process_unit_cost for each row and then summarize that.
    
    Args:
        - loan: data about the requested amount of money
        - y_pred: list of predictions
        - y_true: list of true values
        
    Returns sum of unit costs
    """
    aux_df = pd.DataFrame(
        data={"LoanPrincipal": loan, "predicted": y_pred, "real": y_true}
    )
    return sum(aux_df.apply(lambda x: process_unit_cost(x, 0.0075), axis=1))


def generate_y_pred_with_custom_threshold(model, x_data, threshold):
    """
    Generates new y_predictions according to a threshold.
    
    Args:
        - model: NGBoost model that was trained.
        - x_data: Data on which we predict probabilities
        - threshold: Float value to determine 1 or 0 for new predictions
    
    Returns updated y_predictions
    """
    y_predictions = model.predict_proba(x_data)
    y_pred = []
    count_zero = 0
    count_one = 0
    for i in range(len(list(y_predictions))):
        if y_predictions[i][0] > threshold:
            y_pred.append(0)
            count_zero += 1
        else:
            y_pred.append(1)
            count_one += 1
    print("count_zero " + str(count_zero))
    print("count_one " + str(count_one))
    return y_pred


def get_sample_weights(y_train, y_train_resampled):
    class_weights = class_weight.compute_class_weight(
        "balanced", np.unique(y_train), y_train
    )
    return np.asarray(
        [class_weights[0] if x == 0 else class_weights[1] for x in y_train_resampled],
        dtype=np.float32,
    )


def plot_box_plot(df, col_to_plot):
    list_name = ["Responsible (target=0)", "Delinquent (target=1)"]
    data = [go.Box(
        y=df.fillna(0)[(df.SeriousDlqin2yrs==x)][col_to_plot], name = list_name[x])  for x in sorted(list(df.SeriousDlqin2yrs.unique()))]


    layout = dict(title= f"Boxplot about {col_to_plot}", 
            xaxis= {"title": "Target (SeriousDlqin2yrs)"}, 
            yaxis= {"title": f"{col_to_plot}"})


    fig = go.Figure(data=data, layout=layout)
    iplot(fig)