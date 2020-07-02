from functools import singledispatch
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from typing import Union

import pandas as pd
import numpy as np


def column_names(columns, all_columns):
    if isinstance(columns[0], str):
        return columns

    return all_columns[columns]


@singledispatch
def feat(transformer, names):
    return pd.DataFrame({"name": names, "feature": names})


@feat.register(str)
def _(transformer: str, names):
    if transformer != "passthrough":
        raise ValueError(
            "Transformer passed as the string '"
            + transformer
            + "'. Only the string `passthrough` and `sklearn` transformers are supported."
        )
    return feat(None, names)


@feat.register(OneHotEncoder)
def _(transformer: OneHotEncoder, names):
    n_columns = np.array([len(category) for category in transformer.categories_])
    if transformer.drop:
        n_columns = n_columns - 1

    name = pd.Series(names).repeat(n_columns).reset_index(drop=True)

    return pd.DataFrame({"name": name, "feature": transformer.get_feature_names()})


@feat.register(KBinsDiscretizer)
def _(transformer: KBinsDiscretizer, names, all_columns=None):
    if transformer.encode == "ordinal":
        return feat(None, names)

    name = pd.Series(names).repeat(transformer.n_bins_).reset_index(drop=True)

    edge_labels = [label[1:] for label in transformer.bin_edges_]
    feature = pd.concat(list(map(pd.Series, edge_labels))).reset_index(drop=True)

    df = pd.DataFrame({"name": name, "feature": feature})
    df["feature"] = df["name"] + "-" + df["feature"].apply(lambda x: str(round(x)))

    return df


@feat.register(Pipeline)
def _(transformer: Pipeline, names):

    last_xfer = transformer.steps[-1][1]

    return feat(last_xfer, names)


@feat.register(ColumnTransformer)
def _(transformer: ColumnTransformer, names):
    xf_columns = [
        feat(xfer[1], column_names(xfer[2], names))
        for xfer in transformer.transformers_
    ]

    return pd.concat(xf_columns).reset_index(drop=True)
