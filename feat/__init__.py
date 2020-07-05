from functools import singledispatch, reduce

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import SelectorMixin
from sklearn.decomposition import PCA

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd
from pandas import DataFrame


@singledispatch
def feat(transformer, names) -> DataFrame:
    return DataFrame({"name": names, "feature": names})


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

    return DataFrame({"name": name, "feature": transformer.get_feature_names()})


@feat.register(KBinsDiscretizer)
def _(transformer: KBinsDiscretizer, names, all_columns=None):
    if transformer.encode == "ordinal":
        return feat(None, names)

    name = pd.Series(names).repeat(transformer.n_bins_).reset_index(drop=True)

    edge_labels = [label[1:] for label in transformer.bin_edges_]
    feature = pd.concat(list(map(pd.Series, edge_labels))).reset_index(drop=True)

    df = DataFrame({"name": name, "feature": feature})
    df["feature"] = df["name"] + "-" + df["feature"].apply(lambda x: str(round(x)))

    return df


def nest_feature(df: DataFrame):
    n_rows = df.shape[0]
    n_features = df["feature"].unique().shape[0]
    if n_rows == n_features:
        return df

    return (
        df.groupby(["feature"])
        .apply(lambda x: list(x["name"]))
        .reset_index()
        .rename(columns={0: "name"})[["name", "feature"]]
    )


def merge_feat(x: DataFrame, y: DataFrame):
    unnest_y = y.explode("name")

    name_to_feat = x.merge(unnest_y, left_on="feature", right_on="name")[
        ["name_x", "feature_y"]
    ].rename(columns={"name_x": "name", "feature_y": "feature"})

    return nest_feature(name_to_feat)


@feat.register(Pipeline)
def _(transformer: Pipeline, names):

    feats = []
    input_names = names
    for _, xfer in transformer.steps:
        feats.append(feat(xfer, input_names))
        input_names = feats[-1]["feature"].values

    return reduce(merge_feat, feats)


def column_names(columns, all_columns):
    if isinstance(columns[0], str):
        return np.array(columns)

    return np.array(all_columns)[columns]


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


@feat.register(PCA)
def _(transformer: PCA, names):
    feature_name = transformer.__class__.__name__
    n_features = transformer.n_components_

    df = DataFrame(
        {
            "name": np.repeat(feature_name, n_features),
            "feature": np.arange(0, n_features),
        }
    )

    df["feature"] = df["name"] + "-" + df["feature"].astype(str)
    df["name"] = df["name"].apply(lambda x: list(names))

    return df
