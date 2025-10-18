from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd

# from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV


def prepare_data(
    df, numerical_features=None, categorical_features=None, target="Delivery_Time_min"
):
    if numerical_features is None:
        numerical_features = ["Distance_km", "Preparation_Time_min"]
    if categorical_features is None:
        categorical_features = ["Weather", "Traffic_Level"]

    df_copy = df.copy()
    categorical_features_df = df_copy[categorical_features]
    numerical_features_df = df_copy[numerical_features]
    target_df = df_copy[target]

    data_prepared = pd.concat(
        [categorical_features_df, numerical_features_df, target_df], axis=1
    )

    data_prepared = data_prepared.apply(
        lambda x: x.fillna(x.mode()[0]) if x.dtype == "object" else x.fillna(x.mean()),
        axis=0,
    )

    X = data_prepared.drop(columns=[target])
    y = data_prepared[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, data_prepared


def model_pipeline(num_list, cat_list, model):

    numeric_transformer = Pipeline(
        steps=[
            # ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            # ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_list),
            ("cat", categorical_transformer, cat_list),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_regression)),
            ("model", model),
        ]
    )


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    return mae, mse, r2


def train_and_evaluate(
    model_type,
    X_train,
    y_train,
    X_test,
    y_test,
    numerical_features,
    categorical_features,
):
    if model_type == "regressor":
        model = RandomForestRegressor(random_state=42)
    elif model_type == "svr":
        model = SVR()
    else:
        raise ValueError("Invalid model type")

    pipeline = model_pipeline(numerical_features, categorical_features, model)
    pipeline.fit(X_train, y_train)

    return evaluate_model(pipeline, X_train, y_train, X_test, y_test)


def train_with_grid_search(
    model_type,
    X_train,
    y_train,
    X_test,
    y_test,
    numerical_features,
    categorical_features,
):
    param_grid_rfr = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "feature_selection__k": [5, 10, 8, "all"],
        # 'preprocessor__num__imputer__strategy': ['mean', 'median'],
        # 'preprocessor__cat__imputer__strategy': ['most_frequent']
    }

    param_grid_svr = {
        "model__C": [0.1, 1, 10, 100],
        "model__epsilon": [0.01, 0.1, 0.2],
        "model__kernel": ["rbf", "poly", "linear"],
        "model__gamma": ["scale", "auto"],
        "model__degree": [3, 4, 5],
    }

    if model_type == "regressor":
        param_grid = param_grid_rfr
        model = RandomForestRegressor(random_state=42)
    elif model_type == "svr":
        param_grid = param_grid_svr
        model = SVR()
    else:
        raise ValueError("Invalid model type")

    pipeline = model_pipeline(numerical_features, categorical_features, model)

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, n_jobs=-1, scoring="neg_mean_absolute_error"
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_

    return evaluate_model(best_model, X_train, y_train, X_test, y_test), best_score
