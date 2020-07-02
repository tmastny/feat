from functools import singledispatch
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from typing import Union

import pandas as pd
import numpy as np


@singledispatch
def feat(transformer, columns, all_columns=None):
    return pd.DataFrame({"name": columns, "feature": columns})


@feat.register(str)
def _(transformer: str, columns, all_columns=None):
    if isinstance(columns[0], str):
        return feat(None, columns)

    if transformer != "passthrough":
        raise ValueError(
            "Transformer passed as the string '"
            + transformer
            + "'. Only the string `passthrough` and `sklearn` transformers are supported."
        )

    if all_columns is None:
        raise ValueError(
            "Columns were passed by index. Please pass all the original columns to `all_columns`."
        )

    return feat(None, all_columns[columns])


@feat.register(OneHotEncoder)
def _(transformer: OneHotEncoder, columns, all_columns=None):
    n_columns = np.array([len(category) for category in transformer.categories_])
    if transformer.drop:
        n_columns = n_columns - 1

    name = pd.Series(columns).repeat(n_columns).reset_index(drop=True)

    return pd.DataFrame({"name": name, "feature": transformer.get_feature_names()})


@feat.register(KBinsDiscretizer)
def _(transformer: KBinsDiscretizer, columns, all_columns=None):
    if transformer.encode == "ordinal":
        return feat(None, columns)

    name = pd.Series(columns).repeat(transformer.n_bins_).reset_index(drop=True)

    edge_labels = [label[1:] for label in transformer.bin_edges_]
    feature = pd.concat(list(map(pd.Series, edge_labels))).reset_index(drop=True)

    df = pd.DataFrame({"name": name, "feature": feature})
    df["feature"] = df["name"] + "-" + df["feature"].apply(lambda x: str(round(x)))

    return df


@feat.register(ColumnTransformer)
def _(transformer: ColumnTransformer, columns=None, all_columns=None):
    xf_columns = [
        feat(xfer[1], xfer[2], all_columns=all_columns)
        for xfer in transformer.transformers_
    ]

    return pd.concat(xf_columns).reset_index(drop=True)


@feat.register(Pipeline)
def _(transformer: Pipeline, columns, all_columns=None):

    last_xfer = transformer.steps[-1][1]

    return feat(last_xfer, columns)
