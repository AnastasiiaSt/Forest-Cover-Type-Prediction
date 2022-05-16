import tensorflow as tf
from tensorflow import keras
import click
import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path
from .tune import StringList
from .data import get_data
from .data import get_test_data
from .preprocess import Preprocessing


@click.command()
@click.option(
    "--activation",
    default="relu",
    help="Activation function for inner layers of neural network",
)
@click.option(
    "--nodes_list",
    cls=StringList,
    default=[128, 64, 32, 7],
    help="Number of nodes for each layer",
)
@click.option(
    "--learning_rate", default=0.0005, help="Learning rate for gradient descent"
)
@click.option("--epoch", default=4000, help="Number of epochs for neural network")
@click.option(
    "--cat_encoding", default="no", help="Type of encoding for categorical features"
)
@click.option(
    "--num_scaling", default="standard", help="Type of scaling for numerical features"
)
@click.option(
    "--valid_prop", default=0.2, help="Proportion of dataset used for validation"
)
@click.option("--remove_outliers", default=True, help="Remove outliers from dataset")
@click.option("--data_valid", default=True, help="Remove negative values")
@click.option("--average", default="macro")
@click.option("--random_state", default=42, help="Random state")
@click.option("--dataset_path", default=os.path.join(Path.cwd(), "data", "train.csv"))
def train_n_n(
    nodes_list: list,
    activation: str,
    cat_encoding: str,
    num_scaling: str,
    learning_rate: float,
    epoch: int,
    valid_prop: float,
    remove_outliers: bool,
    data_valid: bool,
    dataset_path: Path,
    average: str,
    random_state: int,
) -> None:

    X, y = get_data(dataset_path)

    ord_enc = OrdinalEncoder()
    y_prep = ord_enc.fit_transform(np.array(y).reshape((-1, 1)))

    X_test, index = get_test_data(os.path.join(Path.cwd(), "data", "test.csv"))

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y_prep, test_size=valid_prop, random_state=random_state
    )

    transformer = Preprocessing(
        cat_encoding=cat_encoding,
        num_scaling=num_scaling,
        remove_outliers=remove_outliers,
        data_valid=data_valid,
    )

    transformer.fit(X_train)
    X_train_prep = transformer.transform(X_train)
    X_valid_prep = transformer.transform(X_valid)
    X_test_prep = transformer.transform(X_test)

    layers = len(nodes_list)

    params = {
        "cat_encoding": cat_encoding,
        "num_scaling": num_scaling,
        "n_layers": layers,
        "nodes_list": nodes_list,
        "epochs": epoch,
        "learning_rate": learning_rate,
        "activation": activation,
        "remove_outliers": remove_outliers,
    }

    with mlflow.start_run(experiment_id=1, run_name="Neural network"):

        model = keras.models.Sequential()
        for i in range(layers - 1):
            model.add(keras.layers.Dense(nodes_list[i], activation=activation))
        model.add(keras.layers.Dense(nodes_list[-1], activation="softmax"))

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
            metrics=["accuracy"],
        )

        history = model.fit(
            X_train_prep, y_train, epochs=epoch, validation_data=(X_valid_prep, y_valid)
        )

        for param in params.items():
            mlflow.log_param(param[0], param[1])

        mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])
        mlflow.log_metric("accuracy", history.history["accuracy"][-1])
        mlflow.keras.log_model(model, "Neural Network")

    y_pred = model.predict(X_test_prep)

    prediction = np.argmax(y_pred, axis=1)
    prediction = ord_enc.inverse_transform(prediction.reshape((-1, 1)))

    df = pd.DataFrame({"Id": index, "Cover_Type": prediction.reshape((-1,))})
    df.to_csv(os.path.join(Path.cwd(), "data", "submission.csv"), index=False)
