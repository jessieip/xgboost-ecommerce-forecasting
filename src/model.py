from typing import Any, tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_model(x_train: pd.DataFrame,
            y_train: pd.Series,
            x_test: pd.DataFrame,
            y_test: pd.Series,
            best_params: dict[str, Any]) -> tuple[xgb.XGBRegressor, np.ndarray, pd.DataFrame]:
    # 2) use the best params to train the model
    """
    Input X_train, y_train to xgboost model and output y_test, predictions, feature importance
    Args:
        x_train: training data columns
        y_train: target variable
        x_test: testing data to validate model performance
        y_test: target testing data to validate model performance
        best_params: best hyperparameters to minimise mean squared error
        early_stopping_rounds: it will stop when the next params aren't better than the previous one.
    Returns:
        model, preds, importance
    """

    model = xgb.XGBRegressor(
        **best_params,
        tree_method="hist",
        enable_categorical=True,
        objective="reg:tweedie",
        tweedie_variance_power=1.7,
        early_stopping_rounds = 50
    )

    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)

    # predict
    preds = model.predict(x_test)
    # evaluate performance
    mse = mean_squared_error(y_test, preds)
    rmse = mse**0.5
    mae = mean_absolute_error(y_test, preds)
    # r2 = r2_score(y_test, preds)
    print(f"RMSE is: {rmse:.3f}")
    # print(f'R_squared: {r2:.3f}')
    print(f"Mean Absolute Error: {mae:.3f}")

    # return the feature's importance

    importance = pd.DataFrame(
        {"feature": x_train.columns, "importance": model.feature_importances_}
    ).sort_values(by="importance", ascending=False)

    # plt.scatter(y_test, preds)
    # plt.show()
    # plt.savefig("prediction_plt.png")

    return model, preds, importance
