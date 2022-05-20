from click.testing import CliRunner
import pytest
import pandas as pd
import os
import joblib
from pathlib import Path
from my_project.neural_network import train_nn


@pytest.fixture
def input_labels():
    test_inputs = pd.read_csv(
        os.path.join(Path.cwd(), "tests", "fixtures", "fixture.csv")
    )
    return test_inputs.iloc[:, -1]


@pytest.fixture
def input_features():
    test_inputs = pd.read_csv(
        os.path.join(Path.cwd(), "tests", "fixtures", "fixture.csv")
    )
    return test_inputs.iloc[:, :-1]


@pytest.fixture
def input_path():
    path = os.path.join(Path.cwd(), "tests", "fixtures", "fixture.csv")
    return path


@pytest.fixture
def output_path():
    path = os.path.join(Path.cwd(), "tests", "nn_model.joblib")
    return path


def test_train(input_path, output_path, input_features, input_labels):
    runner = CliRunner()
    with runner.isolated_filesystem():

        n_network = runner.invoke(
            train_nn,
            [
                "--dataset_path",
                input_path,
                "--save_model_path",
                output_path,
                "--nodes_list",
                "[64,32,16,7]",
            ],
        )
        assert n_network.exit_code == 0
        loaded_n_network = joblib.load(output_path)
        history = loaded_n_network.fit(input_features, input_labels)
        assert history.history["sparse_categorical_loss"][0] > history.history["sparse_categorical_loss"][-1], "Loss of the model does not reduce with number of epochs"
        assert history.history["accuracy"][0] < history.history["accuracy"][-1], "Accuracy of the model does not increase with number of epochs"
        assert history.history["accuracy"][-1] >= 0.5, "Model shows accuracy worse than random guess"
