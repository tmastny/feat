from functools import singledispatch
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer

from typing import Union

import pandas as pd
import numpy as np

@singledispatch
def feat(transformer, columns):
    return "default"

@feat.register(OneHotEncoder)
def _(transformer : OneHotEncoder, columns):
    n_columns = np.array([len(category) for category in transformer.categories_])
    if transformer.drop:
        n_columns = n_columns - 1

    name = pd.Series(columns).repeat(n_columns).reset_index(drop=True)

    return pd.DataFrame({'name': name, 'feature': transformer.get_feature_names()})

@feat.register(OrdinalEncoder)
@feat.register(StandardScaler)
def _(transformer : Union[OrdinalEncoder, StandardScaler], columns):
    return pd.DataFrame({'name': columns, 'feature': columns})

@feat.register(KBinsDiscretizer)
def _(transformer : KBinsDiscretizer, columns):
    if transformer.encode == 'ordinal':
        return pd.DataFrame({'name': columns, 'feature': columns})

    name = pd.Series(columns).repeat(transformer.n_bins_).reset_index(drop=True)

    edge_labels = [label[1:] for label in transformer.bin_edges_]
    feature = pd.concat(list(map(pd.Series, edge_labels))).reset_index(drop=True)

    df = pd.DataFrame({'name': name, 'feature': feature})
    df['feature'] = df['name'] + "-" + df['feature'].apply(lambda x: str(round(x)))

    return df
