import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score as f1
import os
import joblib
import click
import mlflow
from pathlib import Path
from .data import get_data
from .data import get_test_data
from .preprocess import Preprocessing
from .tune import eval_metric


class IntList(click.Option):
    def type_cast_value(self, ctx, value) -> list[int]:
        if value:
            value = str(value)
            list_as_str = value.lstrip("[").rstrip("]")
            list_of_items = [int(item) for item in list_as_str.split(",")]
            return list_of_items
        else:
            return []


class FloatList(click.Option):
    def type_cast_value(self, ctx, value) -> list[float]:
        if value:
            value = str(value)
            list_as_str = value.lstrip("[").rstrip("]")
            list_of_items = [float(item) for item in list_as_str.split(",")]
            return list_of_items
        else:
            return []


class StringList(click.Option):
    def type_cast_value(self, ctx, value) -> list[str]:
        if value:
            value = str(value)
            list_as_str = value.lstrip("[").rstrip("]")
            list_of_items = [item for item in list_as_str.split(",")]
            return list_of_items
        else:
            return []


class BoolList(click.Option):
    def type_cast_value(self, ctx, value) -> list[float]:
        if value:
            value = str(value)
            list_as_str = value.lstrip("[").rstrip("]")
            list_of_items = [eval(item) for item in list_as_str.split(",")]
            return list_of_items
        else:
            return []


@click.command()
@click.option(
    "--n_estimators", cls=IntList, default=[], help="Number of trees in random forest"
)
@click.option(
    "--learning_rate", cls=FloatList, default=[], help="Learning rate for AdaBoost"
)
@click.option(
    "--cat_encoding",
    default="no",
    help="Type of encoding for categorical features",
)
@click.option(
    "--num_scaling", default="standard", help="Type of scaling for numerical features"
)
@click.option("--remove_outliers", default=True, help="Remove outliers from dataset")
@click.option("--data_valid", default=True, help="Remove negative values")
@click.option("--dataset_path", default=os.path.join(Path.cwd(), "data", "train.csv"))
@click.option("--random_state", default=42, help="Random state")
@click.option("--average", default="macro")
def boosting(
    dataset_path: Path,
    average: str,
    n_estimators: list,
    learning_rate: list,
    cat_encoding: str,
    num_scaling: str,
    data_valid: bool,
    remove_outliers: bool,
    random_state: int,
) -> None:

    X_train, y = get_data(dataset_path)
    X_test, index = get_test_data(os.path.join(Path.cwd(), "data", "test.csv"))

    transformer = Preprocessing(
        cat_encoding=cat_encoding,
        num_scaling=num_scaling,
        remove_outliers=remove_outliers,
        data_valid=data_valid,
    )

    transformer.fit(X_train)
    X_train_prep = transformer.transform(X_train)
    X_test_prep = transformer.transform(X_test)

    prep_params = {
        "cat_encoding": cat_encoding,
        "num_scaling": num_scaling,
        "remove_outliers": remove_outliers,
    }

    model_params = {"n_estimators": n_estimators, "learning_rate": learning_rate}

    with mlflow.start_run(experiment_id=3, run_name="Boosting Classifier"):

        ada_clf = AdaBoostClassifier(
            DecisionTreeClassifier(),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        gs_metric = make_scorer(f1, average=average)
        rnd_search = RandomizedSearchCV(
            ada_clf, model_params, scoring=gs_metric, n_iter=3, cv=5
        )
        result = rnd_search.fit(X_train_prep, y)

        best_model = result.best_estimator_

        f1_score = result.best_score_

        best_params = result.best_params_

        params = {**prep_params, **best_params}

        for param in params.items():
            mlflow.log_param(param[0], param[1])

        mlflow.log_metric("f1_score", f1_score)

        mlflow.sklearn.log_model(best_model, "Boosting Classifier")

    y_pred = rnd_search.predict(X_test_prep)

    df = pd.DataFrame({"Id": index, "Cover_Type": y_pred})
    df.to_csv(os.path.join(Path.cwd(), "data", "submission.csv"), index=False)
