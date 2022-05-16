import numpy as np
import pandas as pd
import os
from sklearn.ensemble import VotingClassifier
import joblib
import click
import mlflow
from pathlib import Path
from .data import get_data
from .data import get_test_data
from .preprocess import Preprocessing
from .tune import eval_metric

experiment_id = mlflow.create_experiment("Voting Classifier")


@click.command()
@click.option(
    "--cat_encoding", default="no", help="Type of encoding for categorical features"
)
@click.option(
    "--num_scaling", default="standard", help="Type of scaling for numerical features"
)
@click.option("--remove_outliers", default=True, help="Remove outliers from dataset")
@click.option("--data_valid", default=True, help="Remove negative values")
@click.option("--dataset_path", default=os.path.join(Path.cwd(), "data", "train.csv"))
@click.option("--save_model_path", default=os.path.join(Path.cwd(), "data"))
def voting(
    dataset_path: Path,
    save_model_path: Path,
    cat_encoding: str,
    num_scaling: str,
    remove_outliers: bool,
    data_valid: bool,
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


    with mlflow.start_run(experiment_id=experiment_id, run_name="Voting Classifier"):
        log_clf = joblib.load(os.path.join(save_model_path, "log_model.joblib"))
        rnd_clf = joblib.load(os.path.join(save_model_path, "rnd_model.joblib"))
        svm_clf = joblib.load(os.path.join(save_model_path, "svm_model.joblib"))
        extra_clf = joblib.load(os.path.join(save_model_path, "extra_model.joblib"))

        voting_clf = VotingClassifier(
            estimators=[
                ("Logistic regression", log_clf),
                ("Random Forest", rnd_clf),
                ("Support Vector Machine", svm_clf),
                ("Extra Trees Classifier", extra_clf),
            ],
            voting="hard",
        )
        voting_clf.fit(X_train_prep, y)

        precision, recall, f1_score = eval_metric(
            y, voting_clf.predict(X_train_prep), average="macro"
        )

        model_params = {
            "Logistic regression": "used",
            "Random Forest": "used",
            "Support Vector Machine": "used",
            "Extra Trees Classifier": "used"
        }

        params = {**prep_params, **model_params}

        for param in params.items():
            mlflow.log_param(param[0], param[1])

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1_score)
        mlflow.sklearn.log_model(voting_clf, "Voting Classifier")
        joblib.dump(
            voting_clf, os.path.join(save_model_path, "voting_classifier.joblib")
        )

    y_pred = voting_clf.predict(X_test_prep)

    df = pd.DataFrame({"Id": index, "Cover_Type": y_pred})
    df.to_csv(os.path.join(Path.cwd(), "data", "submission.csv"), index=False)
