# AI4I model → app handoff (for Tab 1)

The app's Tab 1 loads **`app/models/ai4i_pipeline.pkl`**. Until that file exists, the tab shows a "model not added yet" notice; nothing else breaks.

## What the file must contain

A `joblib` dump of a dict:

| key | type | required | used for |
|---|---|---|---|
| `pipeline` | fitted scikit-learn `Pipeline` with `predict_proba` | yes | the prediction |
| `model_name` | str, e.g. `"Random Forest (tuned)"` | no | caption under the result |
| `threshold` | float in (0, 1), default 0.5 | no | probability above which "failure" is shown |
| `metrics` | dict, e.g. `{"F1": 0.81, "Recall": 0.78, "ROC-AUC": 0.97}` | no | caption under the result |

(A bare fitted pipeline instead of a dict also works.)

## What the pipeline must accept

A pandas DataFrame with **exactly these six raw columns**, `Type` as the letters `L` / `M` / `H`:

```
Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]
```

Feature engineering (ΔT, temperature ratio, power, cumulative strain, Type → 0/1/2) and scaling must happen **inside** the pipeline. The engineering step must come from `app/features.py` (`ai4i_engineer_features`), not from a function defined in the notebook. A function defined in a notebook can't be loaded back by the app.

## Export cell (paste at the end of notebook 02, adapt the model line)

```python
import sys, joblib
sys.path.append("../app")
from features import ai4i_engineer_features, AI4I_INPUT_COLUMNS

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

X_raw = df[AI4I_INPUT_COLUMNS]           # raw dataframe loaded in A.1
y = df["Machine failure"]
X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.20,
                                          random_state=RANDOM_STATE, stratify=y)

final = Pipeline([
    ("features", FunctionTransformer(ai4i_engineer_features)),
    ("scaler", StandardScaler()),
    ("model", clone(best_model)),        # <- your final tuned model from Part B
])
final.fit(X_tr, y_tr)

proba = final.predict_proba(X_te)[:, 1]
THRESHOLD = 0.5                          # change if you tuned a decision threshold
pred = (proba >= THRESHOLD).astype(int)
metrics = {"F1": f1_score(y_te, pred), "Recall": recall_score(y_te, pred),
           "Precision": precision_score(y_te, pred), "ROC-AUC": roc_auc_score(y_te, proba)}
print(metrics)

joblib.dump({"pipeline": final, "model_name": "<name of best model>",
             "threshold": THRESHOLD, "metrics": metrics},
            "../app/models/ai4i_pipeline.pkl")
```

## Things that will break it

- **scikit-learn version.** The app pins `scikit-learn==1.7.2`. Export with that same version (`pip install scikit-learn==1.7.2`), or the pickle may fail to load on the deployed app.
- **SVC** needs `probability=True` to have `predict_proba`.
- **SMOTE / imbalanced-learn.** If you resample, use `imblearn.pipeline.Pipeline` instead of the sklearn one, and tell B so `imbalanced-learn` is added (with its exact version) to `app/requirements.txt`.
- **XGBoost / LightGBM.** Same as above: tell B so the package goes into `app/requirements.txt`.
