import pytest
from click.testing import CliRunner
import os
from pathlib import Path
from my_project.neural_network import train_nn
from my_project.data import get_data


@pytest.fixture
def input_path() -> str:
    path = os.path.join(Path.cwd(), "tests", "fixtures", "fixture.csv")
    return path


def test_get_data(input_path: Path) -> None:
    X, y = get_data(path=input_path)
    assert X.shape[0] > 1, "Size of the dataset is insufficient."
    assert (
        X.shape[0] == y.shape[0]
    ), "Number of features is not equal to number of labels."


def test_number_of_nodes_in_nn(input_path: str) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            train_nn,
            ["--nodes_list", [32, 16, 5]],
        )
        assert (
            result.exit_code == 1
        ), "Number of nodes in the output layer should be equal to \
            the number of predicted classes."