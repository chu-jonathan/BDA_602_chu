import os
import sys

import numpy
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.offline as pyo
import pyarrow.parquet as pq
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier

# from sklearn.model_selection import train_test_split


def predictor_dtype(data, predictor_list):
    predictor_datatype = {}
    for i in predictor_list:
        dtype = data[i].dtype
        if dtype in [float, int]:
            predictor_datatype[i] = "continuous"
        if dtype == object:  # what if bool?
            predictor_datatype[i] = "categorical"
    return predictor_datatype


def violin_plots(data, response, predictor_list):
    predictor_dictionary = predictor_dtype(data, predictor_list)
    folder = "plots"
    for i in predictor_list:
        if predictor_dictionary[i] == "continuous":
            fig = px.violin(data, x=i, y=response, box=True)
            plot_path = os.path.join(folder, f"{i}_{response}_violin_plot.html")
            pyo.plot(fig, filename=plot_path)
    return


def logistic(data, response, predictor_list):
    predictor_dictionary = predictor_dtype(data, predictor_list)
    valid = []
    invalid = []
    results = []
    for i in predictor_list:
        if predictor_dictionary[i] == "continuous":
            try:
                x = sm.add_constant(data[i])
                y = data[response]
                logit_model = sm.Logit(y, x).fit()
                pval = logit_model.pvalues[i]
                tval = logit_model.tvalues[i]
                result = {
                    "predictor": i,
                    "response": response,
                    "p_value": pval,
                    "t_value": tval,
                }
                results.append(result)
                valid.append(i)
            except Exception:
                invalid.append(i)
    regression_df = pd.DataFrame(results)
    return valid, invalid, regression_df


def mean_of_response(data, response, predictor_list):
    predictor_dictionary = predictor_dtype(data, predictor_list)
    results = []
    data["float_response"] = data[response].astype(float)
    for i in predictor_list:
        if predictor_dictionary[i] == "continuous":
            data["bin"] = pd.cut(data[i], bins=10)
            bin_response_mean = data.groupby("bin")["float_response"].mean()
            pop_response_mean = data["float_response"].mean()
            sum_sq_diff = (
                np.sum(np.square(bin_response_mean - pop_response_mean))
            ) / 10

            result = {
                "predictor": i,
                "response": response,
                "sum_sq_diff": sum_sq_diff,
            }
            results.append(result)
    mean_df = pd.DataFrame(results)
    return mean_df


def random_forest(data, response, predictor_list):
    predictor_dictionary = predictor_dtype(data, predictor_list)
    continuous = []
    data = data.fillna(data.mean())
    for i in predictor_list:
        if predictor_dictionary[i] == "continuous":
            continuous.append(i)
    x = data[continuous].values
    y = data[response].values

    results = []
    rfc = RandomForestClassifier()
    rfc.fit(x, y)
    gini_importance = rfc.feature_importances_
    for idx, pred in enumerate(continuous):
        result = {
            "predictor": pred,
            "response": response,
            "gini importance": gini_importance[idx],
        }
        results.append(result)
    rf_df = pd.DataFrame(results)
    return rf_df


def main():

    table = pq.read_table("export.parquet")
    df = table.to_pandas()
    response = "win_lose"
    df[response] = df[response].astype("category")
    df[response] = df[response].map({"W": 1, "L": 0}).astype("category")
    predictor = [
        "streak",
        "first_home_line",
        "go_ao",
        "bb_9",
        "k_9",
        "k_pitch_load",
        "k_rest",
        "month_column",
        "days_since_last_game",
    ]
    violin_plots(df, response, predictor)
    print(logistic(df, response, predictor))
    print(mean_of_response(df, response, predictor))
    print(random_forest(df, response, predictor))


if __name__ == "__main__":
    sys.exit(main())
