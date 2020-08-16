# data manipulation libraries
import pandas as pd
import numpy as np

# data viz libraries
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from constants import cols, cols_with_missing_indicators

# sklearn utils
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.inspection import permutation_importance


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


def visualize_roc_curve(model, X_train, y_train, X_test, y_test):
    """
    Plot roc curve
    Args
        - model (NGBClassifier object): NGBoost model
        - X_train (numpy.ndarray): Data without target value (train).
        - X_test (numpy.ndarray): Data without target values (test)
        - y_test (numpy.ndarray): Target values (test)
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


def plot_feature_importances(features, clf):
    """
    Show feature importances plot.
    
    Args:
        - features (list of strings): list of name columns
        - clf (NGboost model): NGboost model that was trained
    """
    df_imp = pd.DataFrame()
    df_imp["feature_name"] = features
    df_imp["feature_importance"] = list(clf.feature_importances_[0])
    df_imp = df_imp.copy().sort_values("feature_importance").reset_index(drop=True)
    trace1 = go.Bar(
        y=df_imp.feature_name,
        x=df_imp.feature_importance,
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


def visualize_permutation_feature_importances(model, X_train, y_train):
    """
    Plot Permutation Feature Importances
    Args
        - model (NGBClassifier object): NGBoost model
        - X_train (numpy.ndarray): Data without target value (train).
        - y_train (numpy.ndarray): Target values of train df.
    """
    # plot feature importance
    results = permutation_importance(model, X_train, y_train)
    importance = results.importances_mean
    for i, v in enumerate(importance):
        print("Feature: %0d, Score: %.5f" % (i, v))

    df_imp = pd.DataFrame()
    df_imp["feature_name"] = cols_with_missing_indicators
    df_imp["feature_importance"] = list(importance)
    df_imp = df_imp.copy().sort_values("feature_importance").reset_index(drop=True)

    trace1 = go.Bar(
        y=df_imp.feature_name,
        x=df_imp.feature_importance,
        marker=dict(color="cornflowerblue", opacity=1),
        orientation="h",
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

    Args:
        - val (int): value that indicates the assignment of color.

    Returns color for that case.
    """
    color = "#FAA8AB" if val > 0 else None
    return "background-color: %s" % color


def plot_box_plot(df, col_to_plot):
    """
    Do box plots for certain column according to target value.
    
    Args:
        - df (DataFrame Object): Dataframe that will be taken.
        - col_to_plot (string): name of column to be considered
    """
    list_name = ["Responsible (target=0)", "Delinquent (target=1)"]
    data = [
        go.Box(
            y=df.fillna(0)[(df.SeriousDlqin2yrs == x)][col_to_plot], name=list_name[x]
        )
        for x in sorted(list(df.SeriousDlqin2yrs.unique()))
    ]

    layout = dict(
        title=f"Boxplot about {col_to_plot}",
        xaxis={"title": "Target (SeriousDlqin2yrs)"},
        yaxis={"title": f"{col_to_plot}"},
    )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


def plot_4d(df, col_cost):
    """ 
    Do 4d plot
    
    Args:
        - df (DataFrame Object): df to be represented.
        - col_cost (string): name of column that contains values in order to set colors
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=df.estimators,
            y=df.learning_rate,
            z=df.max_depth,
            name="Cost",
            mode="markers",
            marker=dict(
                size=6,
                color=df[col_cost],  # set color to an array/list of desired values
                colorscale="RdBu",  # choose a colorscale
                opacity=0.8,
            ),
            hovertemplate="estimators: %{x}<br>learning_rate: %{y} <br>max_depth: %{z}<extra></extra>",
        )
    ),

    fig.add_trace(
        go.Scatter3d(
            x=[100],
            y=[0.01],
            z=[8],
            name="Best Model",
            marker_symbol="x",
            mode="markers",
            marker=dict(
                size=1.5,
                color="yellow",  # set color to an array/list of desired values
                colorscale="RdBu",  # choose a colorscale
                opacity=0.8,
            ),
        ),
    )

    # data = [data1]
    # fig = dict(data=data)
    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis_title="Estimators",
            yaxis_title="Learning_rate",
            zaxis_title="max_depth of Base Learner",
            annotations=[
                dict(
                    showarrow=True,
                    font=dict(
                        family="Courier New, monospace", size=12, color="#000000"
                    ),
                    x=100,
                    y=0.01,
                    z=8,
                    text="Best Model",
                    xanchor="left",
                    opacity=0.7,
                    align="center",
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#000000",
                    ax=-50,
                    ay=-20,
                )
            ],
        ),
    )
    iplot(fig)

