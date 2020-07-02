import os
import pandas as pd
from feat import feat
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import make_column_transformer

dir = os.path.dirname(os.path.abspath(__file__))

hotels = pd.read_csv(os.path.join(dir, "hotels.csv"))
y = hotels['children']
X = hotels.drop('children', axis=1)

nominal = ['hotel', 'meal']
numeric = ['lead_time', 'average_daily_rate']


def test_OneHotEncoder():

    expected = pd.DataFrame({
        "name": ['hotel', 'hotel', 'meal', 'meal', 'meal'],
        "feature": ["x0_City_Hotel", "x0_Resort_Hotel", "x1_BB", "x1_HB", "x1_SC"]
    })

    ohe = OneHotEncoder(sparse=False)

    ohe.fit(hotels[nominal])

    assert feat(ohe, nominal).equals(expected)

def test_OneHotEncoder_drop():

    expected = pd.DataFrame({
        "name": ['hotel', 'meal', 'meal'],
        "feature": ["x0_Resort_Hotel", "x1_HB", "x1_SC"]
    })

    ohe = OneHotEncoder(sparse=False, drop='first')

    ohe.fit(hotels[nominal])

    assert feat(ohe, nominal).equals(expected)


def test_default():

    expected = "default"

    assert feat(1, 1) == expected
