from functools import singledispatch
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

@singledispatch
def feat(transformer, columns):
    return "default"

@feat.register(OneHotEncoder)
def _(transformer, columns):
    n_columns = np.array([len(category) for category in transformer.categories_])
    if transformer.drop:
        n_columns = n_columns - 1

    name = pd.Series(columns).repeat(n_columns).reset_index(drop=True)

    return pd.DataFrame({'name': name, 'feature': transformer.get_feature_names()})

@feat.register(StandardScaler)
def _(transformer, columns)
    return pd.DataFrame({'name': columns, 'feature': columns})
