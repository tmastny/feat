from functools import singledispatch, reduce
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import SelectorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from typing import Union

import pandas as pd
import numpy as np


def column_names(columns, all_columns):
    if isinstance(columns[0], str):
        return np.array(columns)

    return np.array(all_columns)[columns]


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


def merge_feat(x, y):
    return x.merge(y, left_on='feature', right_on='name') \
        [["name_x", "feature_y"]] \
        .rename(columns={"name_x": "name", "feature_y": "feature"})


@feat.register(Pipeline)
def _(transformer: Pipeline, names):

    feats = []
    input_names = names
    for _, xfer in transformer.steps:
        feats.append(feat(xfer, input_names))
        input_names = feats[-1]['feature']


    return reduce(merge_feat, feats)


@feat.register(ColumnTransformer)
def _(transformer: ColumnTransformer, names):
    xf_columns = [
        feat(xfer[1], column_names(xfer[2], names))
        for xfer in transformer.transformers_
    ]

    return pd.concat(xf_columns).reset_index(drop=True)


@feat.register(SelectorMixin)
def _(transformer: SelectorMixin, names):
    mask = transformer.get_support()
    selected = names[mask]

    return feat(None, selected)
