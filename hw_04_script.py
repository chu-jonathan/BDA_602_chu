import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.offline as pyo
import statsmodels.api as sm
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# make relative path for repo
def load_data(data_dir, file_path):
    abs_data_dir = os.path.join(os.path.dirname(__file__), data_dir)
    abs_file_path = os.path.join(abs_data_dir, file_path)
    return abs_file_path


# assuming that data is a pandas df, and response list is a list of column name strings
def response_dtype(data: pd.DataFrame, response_list: list[str]) -> dict[str, str]:
    response_datatype = {}
    for i in response_list:
        dtype = data[i].dtype
        if dtype in [float, int]:
            response_datatype[i] = "continuous"
        if dtype == bool:
            response_datatype[i] = "boolean"
    return response_datatype


# assuming that data is a pandas df, and predictor list is a list of column name strings
def predictor_dtype(data: pd.DataFrame, predictor_list: list[str]) -> dict[str, str]:
    predictor_datatype = {}
    for i in predictor_list:
        dtype = data[i].dtype
        if dtype in [float, int]:
            predictor_datatype[i] = "continuous"
        if dtype == object:  # what if bool?
            predictor_datatype[i] = "categorical"
    return predictor_datatype


# make plots
def make_plots(data: pd.DataFrame, response_list: list[str], predictor_list: list[str]):
    response_dictionary = response_dtype(data, response_list)
    predictor_dictionary = predictor_dtype(data, predictor_list)
    for i in predictor_list:
        for j in response_list:
            if (
                predictor_dictionary[i] == "continuous"
                and response_dictionary[j] == "continuous"
            ):
                fig = px.scatter(data, x=i, y=j, title=f"{i} vs {j}")
                pyo.plot(fig, filename=f"{i}_{j}_scatter_plot.html")
            elif (
                predictor_dictionary[i] == "categorical"
                and response_dictionary[j] == "continuous"
            ):
                fig = px.violin(data, x=i, y=j, box=True)
                pyo.plot(fig, filename=f"{i}_{j}_violin_plot.html")
    return


# fit linear and logistic regression models
def regression(
    data: pd.DataFrame, response_list: list[str], predictor_list: list[str]
) -> None:
    response_dictionary = response_dtype(data, response_list)
    predictor_dictionary = predictor_dtype(data, predictor_list)

    results = []
    for i in predictor_list:
        for j in response_list:
            if (
                predictor_dictionary[i] == "continuous"
                and response_dictionary[j] == "continuous"
            ):
                x = sm.add_constant(data[i])
                y = data[j]

                linear_model = sm.OLS(y, x).fit()
                pval = linear_model.pvalues[i]
                tval = linear_model.tvalues[i]
                result = {
                    "predictor": i,
                    "response": j,
                    "p_value": pval,
                    "t_value": tval,
                }
                results.append(result)

            elif (
                predictor_dictionary[i] == "continuous"
                and response_dictionary[j] == "boolean"
            ):
                x = sm.add_constant(data[i])
                y = data[j]

                logit_model = sm.Logit(y, x).fit()
                pval = logit_model.pvalues[i]
                tval = logit_model.tvalues[i]
                result = {
                    "predictor": i,
                    "response": j,
                    "p_value": pval,
                    "t_value": tval,
                }
                results.append(result)
    regression_df = pd.DataFrame(results)
    return regression_df


# mean of response
def mean_of_response(
    data: pd.DataFrame, response_list: list[str], predictor_list: list[str]
):
    response_dictionary = response_dtype(data, response_list)
    predictor_dictionary = predictor_dtype(data, predictor_list)

    results = []

    for i in predictor_list:
        for j in response_list:
            if (
                predictor_dictionary[i] == "continuous"
                and response_dictionary[j] == "continuous"
            ):
                data["bin"] = pd.cut(data[i], bins=10)
                bin_response_mean = data.groupby("bin")[j].mean()
                pop_response_mean = data[j].mean()
                sum_sq_diff = (
                    np.sum(np.square(bin_response_mean - pop_response_mean))
                ) / 10

                result = {"predictor": i, "response": j, "sum_sq_diff": sum_sq_diff}
                results.append(result)
    mean_df = pd.DataFrame(results)
    return mean_df


# random forest gini importance for continuous predictors only
def random_forest(
    data: pd.DataFrame, response_list: list[str], predictor_list: list[str]
):
    response_dictionary = response_dtype(data, response_list)
    predictor_dictionary = predictor_dtype(data, predictor_list)
    continuous = []
    for i in predictor_list:
        if predictor_dictionary[i] == "continuous":
            continuous.append(i)
    x = data[continuous].values
    y = data[response_list].values

    results = []
    for j in response_list:
        if response_dictionary[j] == "boolean":
            rfc = RandomForestClassifier()
            rfc.fit(x, y)
            gini_importance = rfc.feature_importances_
            for idx, pred in enumerate(continuous):
                result = {
                    "predictor": pred,
                    "response": j,
                    "gini importance": gini_importance[idx],
                }
                results.append(result)
        elif response_dictionary[j] == "continuous":
            rfr = RandomForestRegressor()
            rfr.fit(x, y)
            gini_importance = rfr.feature_importances_
            for idx, pred in enumerate(continuous):
                result = {
                    "predictor": pred,
                    "response": j,
                    "gini importance": gini_importance[idx],
                }
                results.append(result)
    rf_df = pd.DataFrame(results)
    return rf_df


# merge and write to html
def output(data: pd.DataFrame, response_list: list[str], predictor_list: list[str]):
    regression_model = regression(data, response_list, predictor_list)
    mean_of_response_model = mean_of_response(data, response_list, predictor_list)
    random_forest_model = random_forest(data, response_list, predictor_list)

    merged = pd.merge(
        regression_model,
        mean_of_response_model,
        on=["predictor", "response"],
        how="outer",
    )
    merged = pd.merge(
        merged, random_forest_model, on=["predictor", "response"], how="outer"
    )
    html_merged = merged.to_html()
    with open("output.html", "w") as f:
        f.write(html_merged)
    return merged


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
    print(
        output(iris, ["sepal_length", "sepal_width"], ["petal_width", "petal_length"])
    )

    wine_data = load_wine()
    wine_df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
    wine_df["target"] = wine_data.target
    print(
        output(
            wine_df, ["alcohol", "color_intensity"], ["ash", "magnesium", "malic_acid"]
        )
    )
    make_plots(
        wine_df, ["alcohol", "color_intensity"], ["ash", "magnesium", "malic_acid"]
    )
    return


if __name__ == "__main__":
    sys.exit(main())
