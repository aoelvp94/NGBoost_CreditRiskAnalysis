"""
REPO OF USEFUL FUNCTIONS
"""

# ngboost and modelling libraries
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import MinMaxScaler

# data manipulation libraries
import pandas as pd
import numpy as np

# data viz libraries
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from constants import cols


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
        -df (DataFrame object): df to be computed
    """
    df["MonthlyIncome"].fillna(0, inplace=True)
    df["NumberOfDependents"].fillna(0, inplace=True)
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
        df.loc[(df["age"] < 18) | (df["age"] >= 60), col] = 0


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
            x=["target=0"],
            y=[target_value_counts[0]],
            marker=dict(color="cornflowerblue"),
            name="target=0",
        ),
        go.Bar(
            x=["target=1"],
            y=[target_value_counts[1]],
            marker=dict(color="darksalmon"),
            name="target=1",
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
        name="TARGET=0",
        line=dict(color="rgb(167, 103, 4)", width=4),
    )
    trace1 = go.Scatter(
        x=df_target_one_op.index,
        y=df_target_one_op[column_to_op],
        name="TARGET=1",
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


def plot_scatter_matrix(df):
    """
    Plot scatter matrix
    
    Args:
        - df (DataFrame object): Dataframe to be shown
    """
    textd = [
        "target=0" if target == 0 else "target=1" for target in df["SeriousDlqin2yrs"]
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


def visualize_roc_curve(model, X_test, y_test):
    """
    Plot roc curve
    Args
        - model (NGBClassifier object): NGBoost model
        - X_test (numpy.ndarray): Data without target values
        - y_test (numpy.ndarray): Target values
    """
    y_pred = model.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_pred)

    print(classification_report(y_test, y_pred))

    trace0 = go.Scatter(
        x=fpr, y=tpr, name="Predictive Model", line=dict(color="blue", width=2)
    )
    trace1 = go.Scatter(
        x0=0,
        x=[0, 1],
        y0=0,
        y=[0, 1],
        name="Random Chance",
        line=dict(color="grey", width=2),
    )

    data = [trace0, trace1]

    # Edit the layout
    layout = dict(title="ROC curve", xaxis=dict(title="FPR"), yaxis=dict(title="TPR"),)

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
