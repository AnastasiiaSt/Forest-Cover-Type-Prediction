from operator import index
import os
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import click
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.metrics import f1_score as f1
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from .data import get_data
from .data import get_test_data
from typing import Tuple
from .preprocess import Preprocessing


def eval_metric(true, pred, average: str = "macro") -> Tuple[float, float, float]:
    precision = precision_score(true, pred, average=average)
    recall = recall_score(true, pred, average=average)
    f1_score = f1(true, pred, average=average)
    return precision, recall, f1_score


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
    "--model",
    type=click.Choice(
        ["Decision Tree", "Logistic Regression", "Random Forest", "SVM", "Extra Trees"],
        case_sensitive=False,
    ),
    help="The model to be trained",
)
@click.option(
    "--kf_n_outer", default=5, help="Number of folds for outer loop of cross-validation"
)
@click.option(
    "--kf_n_inner", default=5, help="Number of folds for inner loop of cross-validation"
)
@click.option(
    "--max_depth", cls=IntList, default=[], help="Tree maximum depth for decision tree"
)
@click.option(
    "--max_iter",
    cls=IntList,
    default=[],
    help="Maximum number of iterations for logistic regression",
)
@click.option(
    "--regularization",
    cls=FloatList,
    default=[],
    help="Inverse of regularization strength for logistic regression or SVM",
)
@click.option(
    "--penalty",
    cls=StringList,
    default=[],
    help="Penalty for Logistic Regression: 'l2', 'none'",
)
@click.option(
    "--tol",
    cls=FloatList,
    default=[],
    help="Tolerance for stopping criteria for logistic regression and SVM",
)
@click.option(
    "--solver",
    cls=StringList,
    default=[],
    help="Solver for Logistic Regression: 'newton-cg', 'lbfgs', 'sag'",
)
@click.option(
    "--n_estimators", cls=IntList, default=[], help="Number of trees in random forest"
)
@click.option(
    "--criterion",
    cls=StringList,
    default=[],
    help="Criterion for RandomForestClassifier or ExtraTreeClassifier",
)
@click.option(
    "--min_samples_split",
    cls=IntList,
    default=[],
    help="Minimum number of samples to split the node",
)
@click.option(
    "--bootstrap", cls=BoolList, default=[], help="Whether use or not bootstrap"
)
@click.option(
    "--loss",
    cls=StringList,
    default=[],
    help="Criterion for SVM",
)
@click.option("--save_model_path", default=os.path.join(Path.cwd(), "data"))
@click.option(
    "--cat_encoding", default="no", help="Type of encoding for categorical features"
)
@click.option(
    "--num_scaling", default="standard", help="Type of scaling for numerical features"
)
@click.option("--remove_outliers", default=True, help="Remove outliers from dataset")
@click.option("--data_valid", default=True, help="Remove negative values")
@click.option("--dataset_path", default=os.path.join(Path.cwd(), "data", "train.csv"))
@click.option("--average", default="macro")
@click.option("--random_state", default=42, help="Random state")
def tune(
    model: str,
    save_model_path: Path,
    max_depth: list,
    max_iter: list,
    regularization: list,
    penalty: list,
    tol: list,
    solver: list,
    n_estimators: list,
    criterion: list,
    min_samples_split: list,
    bootstrap: list,
    loss: list,
    dataset_path: Path,
    kf_n_inner: int,
    kf_n_outer: int,
    cat_encoding: str,
    num_scaling: str,
    remove_outliers: bool,
    data_valid: bool,
    average: str,
    random_state: int,
) -> None:

    X, y = get_data(dataset_path)

    X_test, index = get_test_data(os.path.join(Path.cwd(), "data", "test.csv"))

    transformer = Preprocessing(
        cat_encoding=cat_encoding,
        num_scaling=num_scaling,
        remove_outliers=remove_outliers,
        data_valid=data_valid,
    )

    transformer.fit(X)
    X_prep = transformer.transform(X)
    X_test_prep = transformer.transform(X_test)

    X_prep = np.array(X_prep)
    y = np.array(y)

    all_params = {
        "n_estimators": n_estimators,  # ExtraTree, RandomForest
        "max_depth": max_depth,  # ExtraTree, RandomForest, DecisionTree
        "max_iter": max_iter,  # LogisticRegression, SVM
        "C": regularization,  # LogisticRegression, SVM
        "penalty": penalty,  # LogisticRegression, SVM
        "tol": tol,  # LogisticRegression, SVM
        "solver": solver,  # LogisticRegression
        "criterion": criterion,  # ExtraTree, RandomForest, DecisionTree
        "bootstrap": bootstrap,  # ExtraTree, RandomForest
        "min_samples_split": min_samples_split,  # ExtraTree, RandomForest, DecisionTree
        "loss": loss,  # SMV
    }

    prep_params = {
        "cat_encoding": cat_encoding,
        "num_scaling": num_scaling,
        "remove_outliers": remove_outliers,
    }

    model_params = {}

    click.echo("Model input lists of parameters:")
    for param in all_params.items():
        if param[1]:
            model_params[param[0]] = param[1]
            click.echo("Parameter: {0} - {1}".format(param[0], param[1]))

    if model == "Decision Tree":
        train_model = DecisionTreeClassifier(random_state=random_state)
        model_file = "tree_model.joblib"
    elif model == "Logistic Regression":
        train_model = LogisticRegression(random_state=random_state)
        model_file = "log_model.joblib"
    elif model == "Random Forest":
        train_model = RandomForestClassifier(random_state=random_state)
        model_file = "rnd_model.joblib"
    elif model == "SVM":
        train_model = LinearSVC(random_state=random_state)
        model_file = "svm_model.joblib"
    elif model == "Extra Trees":
        train_model = ExtraTreesClassifier(random_state=random_state)
        model_file = "extra_model.joblib"

    cv_outer = StratifiedKFold(
        n_splits=kf_n_outer, random_state=random_state, shuffle=True
    )

    outer_scores = []
    outer_models = []
    outer_params = []

    for train_index, test_index in cv_outer.split(X_prep, y):
        with mlflow.start_run(experiment_id=2, run_name=model + "_inner result"):
            X_train, X_test = X_prep[train_index, :], X_prep[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            cv_inner = StratifiedKFold(
                n_splits=kf_n_inner, random_state=random_state, shuffle=True
            )

            gs_metric = make_scorer(f1, average=average)

            gs = RandomizedSearchCV(
                train_model,
                model_params,
                n_iter=2,
                scoring=gs_metric,
                cv=cv_inner,
                refit=True,
            )
            result = gs.fit(X_train, y_train)

            best_model = result.best_estimator_
            outer_models.append(best_model)

            y_pred = best_model.predict(X_test)

            precision, recall, f1_score = eval_metric(y_test, y_pred, average=average)
            outer_scores.append((precision, recall, f1_score))

            mlflow.sklearn.log_model(best_model, model)

            best_params = result.best_params_
            outer_params.append(best_params)

            params = {**prep_params, **best_params}
            params["kf_n_outer"] = kf_n_outer
            params["kf_n_inner"] = kf_n_inner

            for param in params.items():
                mlflow.log_param(param[0], param[1])

            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1_score)

    with mlflow.start_run(experiment_id=2, run_name=model + "_outer result"):

        score_sums = {"avg_precision": 0.0, "avg_recall": 0.0, "avg_f1_score": 0.0}
        max_score = []
        for scores in outer_scores:
            score_sums["avg_precision"] += scores[0]
            score_sums["avg_recall"] += scores[1]
            score_sums["avg_f1_score"] += scores[2]
            max_score.append(scores[2])

        click.echo("Final metrics are:")
        for score_sum in score_sums.items():
            mlflow.log_metric(score_sum[0], score_sum[1] / kf_n_outer)
            click.echo("{0} is {1}.".format(score_sum[0], score_sum[1] / kf_n_outer))

        best_model = outer_models[np.argmax(max_score)]
        mlflow.sklearn.log_model(best_model, model)
        joblib.dump(best_model, os.path.join(save_model_path, model_file))

        best_params = outer_params[np.argmax(max_score)]

        click.echo("Best model parameters are:")
        for param in params.items():
            click.echo("parameter {0} is {1}.".format(param[0], param[1]))
            mlflow.log_param(param[0], param[1])

    y_pred = best_model.predict(X_test_prep)

    df = pd.DataFrame({"Id": index, "Cover_Type": y_pred})
    df.to_csv(os.path.join(Path.cwd(), "data", "submission.csv"), index=False)
