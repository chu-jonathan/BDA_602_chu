import sys
import warnings

import numpy as numpy
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import pearsonr
from sklearn.datasets import load_breast_cancer, load_wine


# assuming that data is a pandas df, and response is a string
def response_dtype(data, response):
    response_datatype = {}
    dtype = data[response].dtype
    if dtype in [float, int]:
        response_datatype[response] = "continuous"
    elif dtype == bool:
        response_datatype[response] = "boolean"
    return response_datatype


# assuming that data is a pandas df, and predictor list is a list of column name strings
def predictor_dtype(data: pd.DataFrame, predictor_list):
    predictor_datatype = {}
    for i in predictor_list:
        dtype = data[i].dtype
        if dtype in [float, int]:
            predictor_datatype[i] = "continuous"
        if dtype == object:  # what if bool?
            predictor_datatype[i] = "categorical"
    return predictor_datatype


# if response continuosu run pearsonr
def pearsonr_table(data, predictor_list):
    predictor_dictionary = predictor_dtype(data, predictor_list)
    continuous = []
    result = pd.DataFrame(columns=["cont_1", "cont_2", "corr"])

    for i in predictor_list:
        if predictor_dictionary[i] == "continuous":
            continuous.append(i)
    corr_matrix = pd.DataFrame(columns=continuous, index=continuous)
    for i in range(len(continuous)):
        for j in range(i + 1, len(continuous)):
            cont_1 = continuous[i]
            cont_2 = continuous[j]

            pearsonr_result = pearsonr(data[cont_1], data[cont_2])
            pearsonr_value = pearsonr_result[0]

            result = result.append(
                {"cont_1": cont_1, "cont_2": cont_2, "corr": pearsonr_value},
                ignore_index=True,
            )
            corr_matrix.loc[cont_1, cont_2] = pearsonr_value
    result_sorted = result.sort_values("corr", ascending=False)

    # https://stackoverflow.com/questions/66572672/correlation-heatmap-in-plotly
    # heatmap code + annotations from stackoverflow
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu",
        )
    )
    for i, row in enumerate(corr_matrix.values):
        for j, val in enumerate(row):
            fig.add_annotation(
                text="{:.2f}".format(val),
                x=corr_matrix.columns[j],
                y=corr_matrix.index[i],
                font=dict(color="black"),
                showarrow=False,
            )

    fig.update_xaxes(title_text="Category 1")
    fig.update_yaxes(title_text="Category 2")
    fig.update_layout(title="Correlation Pearson's Matrix")

    return result_sorted, fig


# correlation taken from class code
def cat_cont_correlation_ratio(
    categories: numpy.ndarray, values: numpy.ndarray
) -> float:
    f_cat, _ = pd.factorize(categories)
    cat_num = numpy.max(f_cat) + 1
    y_avg_array = numpy.zeros(cat_num)
    n_array = numpy.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[numpy.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = numpy.average(cat_measures)
    y_total_avg = numpy.sum(numpy.multiply(y_avg_array, n_array)) / numpy.sum(n_array)
    numerator = numpy.sum(
        numpy.multiply(
            n_array, numpy.power(numpy.subtract(y_avg_array, y_total_avg), 2)
        )
    )
    denominator = numpy.sum(numpy.power(numpy.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numpy.sqrt(numerator / denominator)
    return eta


def cat_correlation_table(data, predictor_list):
    predictor_dictionary = predictor_dtype(data, predictor_list)
    continuous = []
    categorical = []
    result = pd.DataFrame(columns=["Categorical", "Continuous", "Correlation Ratio"])
    for i in predictor_list:
        if predictor_dictionary[i] == "continuous":
            continuous.append(i)
        elif predictor_dictionary[i] == "categorical":
            categorical.append(i)

    for i in categorical:
        for j in continuous:
            corr_ratio = cat_cont_correlation_ratio(data[i], data[j])
            result = result.append(
                {"Categorical": i, "Continuous": j, "Correlation Ratio": corr_ratio},
                ignore_index=True,
            )
    corr_matrix = result.pivot(
        index="Categorical", columns="Continuous", values="Correlation Ratio"
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu",
            colorbar=dict(title="Correlation Ratio Matrix"),
        )
    )

    for i, row in enumerate(corr_matrix.values):
        for j, val in enumerate(row):
            fig.add_annotation(
                text="{:.2f}".format(val),
                x=corr_matrix.columns[j],
                y=corr_matrix.index[i],
                font=dict(color="black"),
                showarrow=False,
            )

    fig.update_layout(
        title="Correlation Ratio Matrix",
        xaxis_title="Continuous Variable",
        yaxis_title="Categorical Variable",
    )

    return result, fig


# taken from class slides
def fill_na(data):
    if isinstance(data, pd.Series):
        return data.fillna(0)
    else:
        return numpy.array([value if value is not None else 0 for value in data])


def cat_correlation(x, y, bias_correction=True, tschuprow=False):

    corr_coeff = numpy.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pd.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = numpy.sqrt(
                    phi2_corrected / numpy.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = numpy.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = numpy.sqrt(phi2 / numpy.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = numpy.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


def cat_cat_table(data, predictor_list):
    predictor_dictionary = predictor_dtype(data, predictor_list)
    categorical = []
    for i in predictor_list:
        if predictor_dictionary[i] == "categorical":
            categorical.append(i)

    tschuprow_results = []
    cramer_results = []
    for i in range(len(categorical)):
        for j in range(i + 1, len(categorical)):
            p1 = categorical[i]
            p2 = categorical[j]
            tschuprow_corr = cat_correlation(
                data[p1], data[p2], bias_correction=True, tschuprow=True
            )
            cramer_corr = cat_correlation(
                data[p1], data[p2], bias_correction=True, tschuprow=False
            )
            tschuprow_results.append((p1, p2, tschuprow_corr))
            cramer_results.append((p1, p2, cramer_corr))

    tschuprow_df = pd.DataFrame(
        tschuprow_results, columns=["cat_1", "cat_2", "tschuprow"]
    )
    cramer_df = pd.DataFrame(cramer_results, columns=["cat_1", "cat_2", "cramer"])
    tschuprow_sorted = tschuprow_df.sort_values("tschuprow", ascending=False)
    cramer_sorted = cramer_df.sort_values("cramer", ascending=False)
    return tschuprow_sorted, cramer_sorted


def cat_cat_heatmap(data, predictor_list):

    tschuprow_sorted, cramer_sorted = cat_cat_table(data, predictor_list)

    tschuprow_heatmap = go.Figure(
        data=go.Heatmap(
            z=tschuprow_sorted["tschuprow"],
            x=tschuprow_sorted["cat_2"],
            y=tschuprow_sorted["cat_1"],
            colorscale="RdBu",
            colorbar=dict(title="Tschuprow's T"),
        )
    )

    for i, row in enumerate(tschuprow_heatmap.data[0].z):
        for j, val in enumerate(row):
            tschuprow_heatmap.add_annotation(
                text="{:.2f}".format(val),
                x=tschuprow_sorted["cat_2"][j],
                y=tschuprow_sorted["cat_1"][i],
                font=dict(color="black"),
                showarrow=False,
            )

    tschuprow_heatmap.update_layout(
        title="Tschuprow's Correlation Matrix",
        xaxis_title="Cat 2",
        yaxis_title="Cat 1",
    )

    cramer_heatmap = go.Figure(
        data=go.Heatmap(
            z=cramer_sorted["cramer"],
            x=cramer_sorted["cat_2"],
            y=cramer_sorted["cat_1"],
            colorscale="RdBu",
            colorbar=dict(title="Cramer's V"),
        )
    )

    for i, row in enumerate(cramer_heatmap.data[0].z):
        for j, val in enumerate(row):
            cramer_heatmap.add_annotation(
                text="{:.2f}".format(val),
                x=cramer_sorted["cat_2"][j],
                y=cramer_sorted["cat_1"][i],
                font=dict(color="black"),
                showarrow=False,
            )

    cramer_heatmap.update_layout(
        title="Cramer's  Correlation Matrix",
        xaxis_title="Cat 2",
        yaxis_title="Cat 1",
    )

    return tschuprow_heatmap, cramer_heatmap


def to_html(data, predictor_list):
    html = ""
    pearsonr_result, pearsonr_fig = pearsonr_table(data, predictor_list)
    html += pearsonr_result.to_html() + pearsonr_fig.to_html()

    cat_corr_result, cat_corr_fig = cat_correlation_table(data, predictor_list)
    html += cat_corr_result.to_html() + cat_corr_fig.to_html()

    tschuprow_result, cramer_result = cat_cat_table(data, predictor_list)
    html += tschuprow_result.to_html() + cramer_result.to_html()

    tschuprow_fig, cramer_fig = cat_cat_heatmap(data, predictor_list)
    html += tschuprow_fig.to_html() + cramer_fig.to_html()

    with open("midterm_report.html", "w") as f:
        f.write(html)
    return html


def main():
    breast_cancer_data = load_breast_cancer()
    breast_cancer_df = pd.DataFrame(
        data=breast_cancer_data.data, columns=breast_cancer_data.feature_names
    )
    breast_cancer_df["target"] = breast_cancer_data.target
    wine_data = load_wine()
    wine_df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
    wine_df["target"] = wine_data.target
    print(wine_df.head())
    predictors = ["alcohol", "malic_acid", "ash"]
    to_html(wine_df, predictors)
    print(breast_cancer_df.head())
    predictors = ["target", "mean radius", "worst concave points", "worst symmetry"]
    to_html(breast_cancer_df, predictors)

    return


if __name__ == "__main__":
    sys.exit(main())
