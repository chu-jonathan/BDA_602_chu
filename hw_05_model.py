import sys
import warnings

import numpy
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import statsmodels.api as sm
from scipy import stats
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text


# assuming that data is a pandas df, and response is a string
def response_dtype(data, response):
    response_datatype = {}
    dtype = data[response].dtype
    if dtype in [float, int]:
        response_datatype[response] = "continuous"
    elif dtype == object:
        response_datatype[response] = "boolean"
    return response_datatype


# assuming that data is a pandas df, and predictor list is a list of column name strings
def predictor_dtype(data, predictor_list):
    predictor_datatype = {}
    for i in predictor_list:
        dtype = data[i].dtype
        if dtype in [float, int]:
            predictor_datatype[i] = "continuous"
        if dtype == object:  # what if bool?
            predictor_datatype[i] = "categorical"
    return predictor_datatype


def make_plots(data, response_list, predictor_list):
    response_dictionary = response_dtype(data, response_list)
    predictor_dictionary = predictor_dtype(data, predictor_list)
    for i in predictor_list:
        if (
            predictor_dictionary[i] == "continuous"
            and response_dictionary[response_list] == "continuous"
        ):
            fig = px.scatter(
                data, x=i, y=response_list, title=f"{i} vs {response_list}"
            )
            pyo.plot(fig, filename=f"{i}_{response_list}_scatter_plot.html")
        elif (
            predictor_dictionary[i] == "categorical"
            and response_dictionary[response_list] == "continuous"
        ):
            fig = px.violin(data, x=i, y=response_list, box=True)
            pyo.plot(fig, filename=f"{i}_{response_list}_violin_plot.html")

    return


def regression(data, response_list, predictor_list):
    response_dictionary = response_dtype(data, response_list)
    predictor_dictionary = predictor_dtype(data, predictor_list)

    results = []
    for i in predictor_list:
        if (
            predictor_dictionary[i] == "continuous"
            and response_dictionary[response_list] == "continuous"
        ):
            x = sm.add_constant(data[i])
            y = data[response_list]

            linear_model = sm.OLS(y, x).fit()
            pval = linear_model.pvalues[i]
            tval = linear_model.tvalues[i]
            result = {
                "predictor": i,
                "response": response_list,
                "p_value": pval,
                "t_value": tval,
            }
            results.append(result)

        elif (
            predictor_dictionary[i] == "continuous"
            and response_dictionary[response_list] == "boolean"
        ):
            x = sm.add_constant(data[i])

            data["home_team_win"] = data["winner_home_or_away"].apply(
                lambda x: 1 if x == "H" else 0
            )

            y = data["home_team_win"]

            logit_model = sm.Logit(y, x).fit()
            pval = logit_model.pvalues[i]
            tval = logit_model.tvalues[i]
            result = {
                "predictor": i,
                "response": response_list,
                "p_value": pval,
                "t_value": tval,
            }
            results.append(result)

    regression_df = pd.DataFrame(results)
    return regression_df


def mean_of_response(data, response_list, predictor_list):
    response_dictionary = response_dtype(data, response_list)
    predictor_dictionary = predictor_dtype(data, predictor_list)

    results = []

    for i in predictor_list:
        if (
            predictor_dictionary[i] == "continuous"
            and response_dictionary[response_list] == "continuous"
        ):
            data["bin"] = pd.cut(data[i], bins=10)
            bin_response_mean = data.groupby("bin")[response_list].mean()
            pop_response_mean = data[response_list].mean()
            sum_sq_diff = (
                np.sum(np.square(bin_response_mean - pop_response_mean))
            ) / 10

            result = {
                "predictor": i,
                "response": response_list,
                "sum_sq_diff": sum_sq_diff,
            }
            results.append(result)
    mean_df = pd.DataFrame(results)
    return mean_df


# merge and write to html
def output(data, response_list, predictor_list):
    merged = regression(data, response_list, predictor_list)
    html_merged = merged.to_html()
    with open("output.html", "w") as f:
        f.write(html_merged)
    return merged


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


def to_html(data, predictor_list):
    html = ""
    pearsonr_result, pearsonr_fig = pearsonr_table(data, predictor_list)
    html += pearsonr_result.to_html() + pearsonr_fig.to_html()

    cat_corr_result, cat_corr_fig = cat_correlation_table(data, predictor_list)
    html += cat_corr_result.to_html() + cat_corr_fig.to_html()

    with open("midterm_report.html", "w") as f:
        f.write(html)
    return html


def main():

    user = "jchu"
    password = "bda"  # pragma: allowlist secret
    host = "mariadb"
    port = "3306"
    database = "baseball"

    engine = create_engine(
        f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}",
        future=True,
    )

    # sqlalchemy 2.0 rewrite
    with engine.begin() as connection:
        features = connection.execute(text("SELECT * FROM features"))

    df = pd.DataFrame(features)
    df = df.fillna(0)
    response = "winner_home_or_away"
    print(df[response].unique())
    predictor = [
        "hr_h",
        "obp",
        "pa_so",
        "bb_9",
        "h_9",
        "hr_9",
        "k_9",
        "go_ao",
        "k_pitch_load",
        "k_rest",
    ]
    output(df, response, predictor)
    to_html(df, predictor)

    df["home_team_win"] = df["winner_home_or_away"].apply(
        lambda x: 1 if x == "H" else 0
    )
    print(df.dtypes)

    # something breaks in the switch from sqlalchemy 1.4 to sqlalchemy 2.0 that takes
    # obp and h_9 as object and not floats
    df["obp"] = df["obp"].astype(float)
    df["h_9"] = df["h_9"].astype(float)
    print(df.dtypes)
    # trouble reconciling response dtype object with mean of response and random forest
    # model restricted to logistic regression
    # looking at t vals, obp, bb/9, pa/so, h/9, k/9 have strongest relationships predicting home team win

    strong_predictors = ["obp", "bb_9", "pa_so", "h_9", "k_9"]
    X = df[strong_predictors]
    Y = df["home_team_win"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    logit_model = sm.Logit(y_train, X_train)
    result = logit_model.fit()
    y_pred = result.predict(X_test)
    accuracy = sum((y_pred >= 0.5) == y_test) / len(y_test)
    print("Accuracy:", accuracy)

    # we have a subset of predictors that are still significant, but not as strongly correlated
    # as the strong predictors from the t-values
    # interestingly, created features k/pitchload (measuring load on a pitcher) and k/rest (measuring
    # how rested a pitcher is) are still predictive
    weak_predictors = ["hr_h", "go_ao", "k_pitch_load", "k_rest"]
    X = df[weak_predictors]
    Y = df["home_team_win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    logit_model = sm.Logit(y_train, X_train)
    result = logit_model.fit()
    y_pred = result.predict(X_test)
    accuracy = sum((y_pred >= 0.5) == y_test) / len(y_test)
    print("Accuracy:", accuracy)

    # The strong predictors have an accuracy of 51.4%, the weak have an accuracy of 51.09%. Given this
    # is a logistic regression model, an accuracy of about 50% is almost no better than random.
    # better predictors and a better model need to be found.


if __name__ == "__main__":
    sys.exit(main())
