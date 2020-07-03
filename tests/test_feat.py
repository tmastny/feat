import os
import pytest
import pandas as pd
import numpy as np
from feat import feat, column_names
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import FunctionTransformer

from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import make_pipeline

dir = os.path.dirname(os.path.abspath(__file__))

hotels = pd.read_csv(os.path.join(dir, "hotels.csv"))
y = hotels["children"]
X = hotels.drop("children", axis=1)

nominal = ["hotel", "meal"]
numeric = ["lead_time", "average_daily_rate"]


def test_default():

    assert feat(None, nominal).equals(
        pd.DataFrame({"name": nominal, "feature": nominal})
    )


def test_string():
    expected = pd.DataFrame({"name": numeric, "feature": numeric})

    string = "passthrough"
    columns = [1, 3]
    all_columns = X.columns

    assert feat(string, column_names(columns, all_columns)).equals(expected)


def test_string_not_supported():
    string = "other_string"
    columns = [1, 3]

    with pytest.raises(ValueError, match=r".* string `passthrough` .*"):
        feat(string, columns)


def test_OneHotEncoder():

    expected = pd.DataFrame(
        {
            "name": ["hotel", "hotel", "meal", "meal", "meal"],
            "feature": ["x0_City_Hotel", "x0_Resort_Hotel", "x1_BB", "x1_HB", "x1_SC"],
        }
    )

    ohe = OneHotEncoder(sparse=False)

    ohe.fit(X[nominal])

    assert feat(ohe, nominal).equals(expected)


def test_OneHotEncoder_drop():

    expected = pd.DataFrame(
        {
            "name": ["hotel", "meal", "meal"],
            "feature": ["x0_Resort_Hotel", "x1_HB", "x1_SC"],
        }
    )

    ohe = OneHotEncoder(sparse=False, drop="first")

    ohe.fit(X[nominal])

    assert feat(ohe, nominal).equals(expected)


def test_OrdinalEncoder():
    expected = pd.DataFrame({"name": nominal, "feature": nominal})

    oe = OrdinalEncoder()
    oe.fit(X[nominal])

    assert feat(oe, nominal).equals(expected)


kbin_expected = pd.DataFrame(
    {
        "name": {
            0: "lead_time",
            1: "lead_time",
            2: "lead_time",
            3: "lead_time",
            4: "lead_time",
            5: "average_daily_rate",
            6: "average_daily_rate",
            7: "average_daily_rate",
            8: "average_daily_rate",
            9: "average_daily_rate",
        },
        "feature": {
            0: "lead_time-4",
            1: "lead_time-36",
            2: "lead_time-87",
            3: "lead_time-171",
            4: "lead_time-457",
            5: "average_daily_rate-69",
            6: "average_daily_rate-94",
            7: "average_daily_rate-123",
            8: "average_daily_rate-162",
            9: "average_daily_rate-335",
        },
    }
)


def test_KBinsDiscretizer():
    kbin = KBinsDiscretizer(n_bins=5, encode="onehot")
    kbin.fit(X[numeric])

    assert feat(kbin, numeric).equals(kbin_expected)


def test_KBinsDiscretizer_dense():

    kbin = KBinsDiscretizer(n_bins=5, encode="onehot-dense")
    kbin.fit(X[numeric])

    assert feat(kbin, numeric).equals(kbin_expected)


def test_KBinsDiscretizer_ordinal():
    expected = pd.DataFrame({"name": numeric, "feature": numeric})

    kbin = KBinsDiscretizer(n_bins=5, encode="ordinal")
    kbin.fit(X[numeric])

    assert feat(kbin, numeric).equals(expected)


@pytest.mark.filterwarnings("ignore:Bins whose width")
def test_KBinsDiscretizer_different_n_bins():

    expected = pd.DataFrame(
        {
            "name": {
                0: "lead_time",
                1: "lead_time",
                2: "lead_time",
                3: "lead_time",
                4: "average_daily_rate",
                5: "average_daily_rate",
                6: "average_daily_rate",
                7: "average_daily_rate",
                8: "low_rate",
                9: "low_rate",
                10: "low_rate",
            },
            "feature": {
                0: "lead_time-8",
                1: "lead_time-58",
                2: "lead_time-151",
                3: "lead_time-457",
                4: "average_daily_rate-79",
                5: "average_daily_rate-108",
                6: "average_daily_rate-151",
                7: "average_daily_rate-335",
                8: "low_rate-78",
                9: "low_rate-107",
                10: "low_rate-151",
            },
        }
    )

    X_diff = X.assign(
        low_rate=np.select(
            [
                X["average_daily_rate"] < 78,
                X["average_daily_rate"] < 107,
                X["average_daily_rate"] < 151,
            ],
            [0, 78, 107,],
            default=151,
        )
    )

    columns = numeric + ["low_rate"]
    kbin = KBinsDiscretizer(n_bins=4, encode="onehot-dense")

    kbin.fit(X_diff[columns])

    assert feat(kbin, columns).equals(expected)


def group_meals(array):
    return array.applymap(lambda x: "HB" if x == "SC" else x)


def test_Pipeline():

    expected = pd.DataFrame(
        {"name": ["meal", "meal",], "feature": ["x0_BB", "x0_HB",],}
    )

    xfer_pipeline = make_pipeline(
        FunctionTransformer(group_meals), OneHotEncoder(sparse=False)
    )

    xfer_pipeline.fit(X[["meal"]])

    assert feat(xfer_pipeline, ["meal"]).equals(expected)


def test_Pipeline_with_passthrough():

    expected = pd.DataFrame({"name": ["meal"], "feature": ["meal"],})

    xfer_pipeline = make_pipeline(FunctionTransformer(group_meals), "passthrough")

    xfer_pipeline.fit(X[["meal"]])

    assert feat(xfer_pipeline, ["meal"]).equals(expected)


def test_ColumnTransformer1():
    expected = pd.DataFrame(
        {
            "name": [
                "hotel",
                "hotel",
                "meal",
                "meal",
                "meal",
                "lead_time",
                "average_daily_rate",
            ],
            "feature": [
                "x0_City_Hotel",
                "x0_Resort_Hotel",
                "x1_BB",
                "x1_HB",
                "x1_SC",
                "lead_time",
                "average_daily_rate",
            ],
        }
    )

    preprocess = make_column_transformer(
        (OneHotEncoder(sparse=False), nominal), (StandardScaler(), numeric)
    )

    preprocess.fit(X)

    assert feat(preprocess, X.columns).equals(expected)


def test_ColumnTransformer1_rev():
    expected = pd.DataFrame(
        {
            "name": [
                "lead_time",
                "average_daily_rate",
                "hotel",
                "hotel",
                "meal",
                "meal",
                "meal",
            ],
            "feature": [
                "lead_time",
                "average_daily_rate",
                "x0_City_Hotel",
                "x0_Resort_Hotel",
                "x1_BB",
                "x1_HB",
                "x1_SC",
            ],
        }
    )

    preprocess = make_column_transformer(
        (StandardScaler(), numeric), (OneHotEncoder(sparse=False), nominal)
    )

    preprocess.fit(X)

    assert feat(preprocess, X.columns).equals(expected)


def test_ColumnTransformer_with_Pipeline():
    expected = pd.DataFrame(
        {
            "name": [
                "meal",
                "meal",
                "lead_time",
                "average_daily_rate",
                "hotel",
                "hotel",
            ],
            "feature": [
                "x0_BB",
                "x0_HB",
                "lead_time",
                "average_daily_rate",
                "x0_City_Hotel",
                "x0_Resort_Hotel",
            ],
        }
    )

    xfer_pipeline = make_pipeline(
        FunctionTransformer(group_meals), OneHotEncoder(sparse=False)
    )

    preprocess = make_column_transformer(
        (xfer_pipeline, ["meal"]),
        (StandardScaler(), numeric),
        (OneHotEncoder(sparse=False), ["hotel"]),
    )

    preprocess.fit(X)

    assert feat(preprocess, X.columns).equals(expected)


def test_ColumnTransformer_with_passthrough():
    expected = pd.DataFrame(
        {
            "name": [
                "meal",
                "meal",
                "hotel",
                "hotel",
                "lead_time",
                "average_daily_rate",
            ],
            "feature": [
                "x0_BB",
                "x0_HB",
                "x0_City_Hotel",
                "x0_Resort_Hotel",
                "lead_time",
                "average_daily_rate",
            ],
        }
    )

    xfer_pipeline = make_pipeline(
        FunctionTransformer(group_meals), OneHotEncoder(sparse=False)
    )

    preprocess = make_column_transformer(
        (xfer_pipeline, ["meal"]),
        (OneHotEncoder(sparse=False), ["hotel"]),
        remainder="passthrough",
    )

    preprocess.fit(X)

    assert feat(preprocess, X.columns).equals(expected)


def test_ColumnTransformer_with_indices():
    expected = pd.DataFrame(
        {
            "name": [
                "meal",
                "meal",
                "lead_time",
                "average_daily_rate",
                "hotel",
                "hotel",
            ],
            "feature": [
                "x0_BB",
                "x0_HB",
                "lead_time",
                "average_daily_rate",
                "x0_City_Hotel",
                "x0_Resort_Hotel",
            ],
        }
    )

    xfer_pipeline = make_pipeline(
        FunctionTransformer(group_meals), OneHotEncoder(sparse=False)
    )

    preprocess = make_column_transformer(
        (xfer_pipeline, ["meal"]),
        (StandardScaler(), [1, 3]),
        (OneHotEncoder(sparse=False), [0]),
    )

    preprocess.fit(X)

    assert feat(preprocess, X.columns).equals(expected)


def test_ColumnTransformer_with_selector():
    expected = pd.DataFrame(
        {
            "name": [
                "hotel",
                "hotel",
                "meal",
                "meal",
                "meal",
                "lead_time",
                "average_daily_rate",
            ],
            "feature": [
                "x0_City_Hotel",
                "x0_Resort_Hotel",
                "x1_BB",
                "x1_HB",
                "x1_SC",
                "lead_time",
                "average_daily_rate",
            ],
        }
    )

    preprocess = make_column_transformer(
        (OneHotEncoder(sparse=False), make_column_selector(dtype_include=object)),
        (StandardScaler(), numeric, make_column_selector(dtype_exclude=object)),
    )

    preprocess.fit(X)

    assert feat(preprocess, X.columns).equals(expected)


def test_VarianceThreshold():
    expected = pd.DataFrame(
        {
            "name": ["lead_time", "average_daily_rate",],
            "feature": ["lead_time", "average_daily_rate",],
        }
    )

    X_var = X[numeric].assign(no_var=150)

    var = VarianceThreshold()
    var.fit(X_var)

    assert feat(var, X_var.columns).equals(expected)
