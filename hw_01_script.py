import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# make relative path for repo
def load_data(data_dir, file_path):
    abs_data_dir = os.path.join(os.path.dirname(__file__), data_dir)
    abs_file_path = os.path.join(abs_data_dir, file_path)
    return abs_file_path


# make summary statistics for each column
def column_summary_statistics(column):
    if np.issubdtype(
        column.dtype, np.number
    ):  # need to have check for string, otherwise quartile breaks
        mean = np.mean(column)
        min = np.min(column)
        max = np.max(column)
        quartiles = np.percentile(column, [25, 50, 75])

        return pd.Series(
            {
                "mean": mean,
                "minimum": min,
                "maximum": max,
                "25%": quartiles[0],
                "50%": quartiles[1],
                "75%": quartiles[2],
            }
        )
    else:
        return pd.Series(
            {
                "mean": np.nan,
                "minimum": np.nan,
                "maximum": np.nan,
                "25%": np.nan,
                "50%": np.nan,
                "75%": np.nan,
            }
        )


# apply column_summary_statistcs to entire dataframe
def summary_statistics(df):
    return df.apply(column_summary_statistics)


def create_predictors(df, predictor_columns):
    return df[predictor_columns].values


def create_response(df, response_column):
    return df[response_column].values


# make pipelines for Random Forest, K Nearest Neighbors, and Decision Trees. credit your slides
def build_and_fit_model(x, y, model_name):
    if model_name == "RandomForest":
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("RandomForest", RandomForestClassifier(random_state=111)),
            ]
        )
    elif model_name == "KNeighbors":
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("KNeighbors", KNeighborsClassifier())]
        )
    elif model_name == "DecisionTree":
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("DecisionTree", DecisionTreeClassifier(random_state=111)),
            ]
        )
    else:
        raise ValueError("Pick RandomForest, KNeighbors, DecisionTree")
    pipeline.fit(x, y)
    probability = pipeline.predict_proba(x)
    prediction = pipeline.predict(x)
    print(f"Classes: {pipeline.classes_}")
    print(f"Probability: {probability}")
    print(f"Predictions: {prediction}")


# extract list of probabilties for each row from RandomForest
def list_of_probabilities(x, y):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=111)),
        ]
    )
    pipeline.fit(x, y)
    probabilities = pipeline.predict_proba(x)
    list_of_trues = [i[1] for i in probabilities]
    return list_of_trues


# create total dataframe for histogram scatter plot
# creates a dataframe of the binned predictor values, the population mean, the mean of the bin
# and the left side of the bin for the plot point
# credit to stackexchange explaining lambda functions https://stackoverflow.com/a/61031085
def predictor_dataframe(data, column_name, x, y):
    predictor = data[column_name]
    bin_predictor = pd.cut(x=data[column_name], bins=10)
    true_probs = list_of_probabilities(x, y)
    df = pd.concat({"predictor": predictor, "bins": bin_predictor}, axis=1)
    df["true_probs"] = true_probs
    df["pop_mean_prob"] = df["true_probs"].mean()
    grouped = df.groupby("bins", as_index=False)["true_probs"].mean()
    grouped["pop_mean_prob"] = df["pop_mean_prob"].head(10).to_list()
    grouped["point"] = grouped["bins"].apply(lambda x: x.left)
    x_points = grouped["point"].tolist()
    y_mean = grouped["pop_mean_prob"].tolist()
    y_mean_bin = grouped["true_probs"].tolist()
    return x_points, y_mean, y_mean_bin


# make hist and scatter plots. credit to stackexchange and plotly documentation
def plot_hist_and_scatter(iris, predictor, response):
    x = create_predictors(iris, [predictor])
    y = create_response(iris, [response])
    x_points, y_pop_mean, y_bin_mean = predictor_dataframe(iris, predictor, x, y)

    fig = px.histogram(
        iris, x=predictor, nbins=10, title=f"{predictor} probability for {response}"
    )
    fig.add_trace(
        go.Scatter(
            x=x_points, y=y_pop_mean, mode="lines", name="Population Mean", yaxis="y2"
        )
    )
    fig.add_trace(
        go.Scatter(x=x_points, y=y_bin_mean, mode="lines", name="Bin mean", yaxis="y2")
    )
    fig.update_layout(
        yaxis2=dict(
            overlaying="y",
            side="right",
            range=[0, 1.5],
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=["0", "0.25", "0.5", "0.75", "1"],
            showticklabels=True,
        ),
        xaxis_title=f"{predictor}",
        yaxis_title="Frequency",
    )
    fig.show()


def main():
    iris_path = load_data("data", "iris.data")
    iris = pd.read_csv(iris_path, header=None)

    iris.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]
    iris["is_setosa"] = iris["class"] == "Iris-setosa"
    iris["is_versicolor"] = iris["class"] == "Iris-versicolor"
    iris["is_virginica"] = iris["class"] == "Iris-virginica"
    print(iris.head(3))
    print(iris.tail(3))
    print(summary_statistics(iris))

    fig = px.scatter(
        iris,
        x="sepal_length",
        y="sepal_width",
        color="class",
        title="Sepal Length vs Width",
    )
    fig.show()

    fig = px.scatter(
        iris,
        x="petal_length",
        y="petal_width",
        color="class",
        title="Petal Length vs Width",
    )
    fig.show()

    fig = px.histogram(iris, x="sepal_length", y="sepal_width", color="class")
    fig.show()

    fig = px.histogram(iris, x="petal_length", y="petal_width", color="class")
    fig.show()

    fig = px.violin(iris, x="petal_length", y="sepal_length", color="class", box=True)
    fig.show()

    predictor = ["sepal_length"]
    response = ["is_setosa"]
    x = create_predictors(iris, predictor)
    y = create_response(iris, response)

    build_and_fit_model(x, y, "RandomForest")
    build_and_fit_model(x, y, "KNeighbors")
    build_and_fit_model(x, y, "DecisionTree")

    predictors_list = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    responses_list = ["is_setosa", "is_versicolor", "is_virginica"]
    for i in predictors_list:
        for j in responses_list:
            plot_hist_and_scatter(iris, i, j)
    return


if __name__ == "__main__":
    sys.exit(main())
