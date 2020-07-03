
<!-- README.md is generated from README.Rmd. Please edit that file -->

# feat

feat makes it easy to pull feature importance and variable names from
sklearn pipelines and models.

## Installation

Coming soon to PyPi.

``` bash
git clone https://github.com/tmastny/feat.git
pip install feat/
```

## Examples

sklearn [column
transformers](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html?highlight=columntransformer#sklearn.compose.ColumnTransformer)
gather all preprocessing and feature engineering into a single unit that
can be reused during training, evaluation, and analysis. Transformations
could be

  - single column transformations like `log` or scaling.
  - single to multi-column conversion like one-hot or dummy encoding
  - column removal, like low variance filters
  - multi-column transformation like PCA

<!-- end list -->

``` python
import pandas as pd
from sklearn.compose import make_column_transformer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold


hotels = pd.read_csv("tests/hotels.csv")
X = hotels.drop('children', axis=1)
y = hotels['children']

preprocess = make_column_transformer(
    (OneHotEncoder(), ['hotel', 'meal']),
    (VarianceThreshold(), ['lead_time']),
    (StandardScaler(), ['average_daily_rate'])
)
```

You can also combine column transformations with an sklearn estimator
using
[pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
Pipelines combine data transformations and machine learning algorithms
into a single object that ingests raw data and outputs predictions.

Unfortunately, the final output is stripped of all metadata, like column
names, leaving attributes like *feature importance* and *coefficients*
difficult to interpret.

``` python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

log_pipeline = make_pipeline(
    preprocess,
    LogisticRegression()
)

log_model = log_pipeline.fit(X, y)

coef = log_model.named_steps['logisticregression'].coef_[0]
coef
#> array([ 7.13309064e-02,  2.18560209e-01, -7.65540765e-01,  2.92076193e-01,
#>         7.63355688e-01, -7.29692233e-04, -1.06387812e+00])
```

How can we determine which categories are mapped to the `onehot`
columns? Was `lead_time` removed with the low-variance filter?

`feat` maps the transformed columns back to the original column name in
a pandas dataframe so you can make sense of coefficients and feature
importances:

``` python
from feat import feat

feats = feat(log_model.named_steps['columntransformer'], X.columns)
feats.assign(coef=coef)
#>                  name             feature      coef
#> 0               hotel       x0_City_Hotel  0.071331
#> 1               hotel     x0_Resort_Hotel  0.218560
#> 2                meal               x1_BB -0.765541
#> 3                meal               x1_HB  0.292076
#> 4                meal               x1_SC  0.763356
#> 5           lead_time           lead_time -0.000730
#> 6  average_daily_rate  average_daily_rate -1.063878
```
