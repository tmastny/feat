import os
import pytest
import pandas as pd
from feat import feat
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import make_column_transformer

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

    assert feat(string, columns, all_columns).equals(expected)


def test_string_not_supported():
    string = "other_string"
    columns = [1, 3]

    with pytest.raises(ValueError, match=r".* string `passthrough` .*"):
        feat(string, columns)


def test_string_all_columns():
    string = "passthrough"
    columns = [1, 3]

    with pytest.raises(ValueError, match=r".* index.*"):
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

    assert feat(preprocess).equals(expected)


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

    assert feat(preprocess).equals(expected)
