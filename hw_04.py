import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
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


def make_plots(iris, x, y, color=None):

    fig = px.scatter(iris, x=x, y=y, title=f"{x} vs {y}")
    fig.show()
    pyo.plot(fig, filename=f"{x}_{y}_scatter_plot.html")

    fig = px.histogram(iris, x=x, y=y, color=color)
    fig.show()
    pyo.plot(fig, filename=f"{x}_{y}_histogram_plot.html")

    fig = px.violin(iris, x=x, y=y, color=color, box=True)
    fig.show()
    pyo.plot(fig, filename=f"{x}_{y}_violin_plot.html")


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
    continuous = []
    boolean = []
    categorical = []
    for column in iris:
        dtype = iris[column].dtype
        if dtype == "bool":
            boolean.append(column)
        if dtype == "string" or dtype == "object":
            categorical.append(column)
        if dtype == "int64" or dtype == "float64":
            continuous.append(column)
        print(dtype)
    print(continuous)
    print(boolean)
    print(categorical)

    make_plots(iris, "sepal_length", "sepal_width", "class")
    make_plots(iris, "petal_length", "petal_width", "class")
    make_plots(iris, "petal_length", "sepal_length", "class")

    print(iris.head(3))
    return


if __name__ == "__main__":
    sys.exit(main())
