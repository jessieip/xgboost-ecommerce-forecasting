import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from typing import Tuple, Dict, Any


def prepare_var(df: pd.DataFrame,
                target: str,
                test_size: float,
                random_state: int) -> Tuple[pd.DataFrame,pd.Series,pd.DataFrame,pd.Series,pd.DataFrame,pd.Series]:
    """
    split DataFrame into train, val and test sets.
    Args:
        df: DataFrame contains feature and target
        target: the name of target variable
        test_size: proportion of dataset to include in the test split (e.g. 0.2)
        random_state: random state for reproducibility
    Returns:
         X_train, y_train, X_val, y_val, X_test, y_test
    """
    y = df[target]
    X = df.drop([target], axis=1)

    cat_col = X.select_dtypes(exclude=np.number).columns.tolist()

    for col in cat_col:
        X[col] = X[col].astype("category")

    # Split to Train, Val
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # split X_train_val, y_train_val for Optuna
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=test_size, random_state=random_state
    )

    return X_train, y_train, X_val, y_val, X_test, y_test



def optimise_xgboost(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series) -> Dict[str, Any]:
    """
    Using Optuna to find the best hyperparameters for xgboost model
    Args:
        X_train: training variables
        y_train: training target
        X_val: validation variables
        y_val: validation target

    Returns: return best parameters include max_depth, learning_rate, n_estimators"
    """

    def objective(trial: optuna.Trial) -> float:
        param = {
            "n_estimators": 1000,  # total number fo trees. if there's too many trees, it can cause overfitting
            "max_depth": trial.suggest_int(
                "max_depth", 4, 8
            ),  # the max depth(how many layers) for each trees. model can learn more complex rules for purchase/not purchase. e.g. age <25, is_weekend = No is not purchase..
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.1, log=True
            ),  # the speed of learning, the current tree corrects previous tree mistake
            "objective": "reg:tweedie",
            "tree_method": "hist",  # Histogram Algorithm: faster way to calculate
            "enable_categorical": True,
            "device": "cpu",
            "n_jobs": -1,
            "eval_metric": "mae",
        }

        reg = xgb.XGBRegressor(**param, early_stopping_rounds= 50)

        # reg.fit(X_train, y_train)

        reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        preds = reg.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        return mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    return study.best_params
    



