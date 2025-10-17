import pytest
import pandas as pd
from pipeline import prepare_data

df = pd.read_csv("./data/dataset.csv")

print(df.head())


@pytest.fixture
def prepared_data():
    return prepare_data(df)


def test_length_of_data(prepared_data):
    X_train, X_test, y_train, y_test, data = prepared_data

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(df)

    assert data.shape == (1000, 5)


def test_type_of_columns(prepared_data):
    _, _, _, _, data = prepared_data

    assert data["Distance_km"].dtype == "float64"
    assert data["Preparation_Time_min"].dtype == "int64"
    assert data["Weather"].dtype == "object"
    assert data["Traffic_Level"].dtype == "object"
    assert data["Delivery_Time_min"].dtype == "int64"
