import pandas as pd
import numpy as np

def prepare_data(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering: transform text to numeric data, calculate average spend per session

    Args:
        df_input: raw DataFrame from database.py
    Returns:
        Transformed DataFrame with the columns that are used in the model and target (label_session_spend)
    """
    df = df_input.copy()

    df["day_num"] = df["session_start"].dt.weekday
    df["is_weekend"] = (df["day_num"] >= 5).astype(int)
    df["hour"] = df["session_start"].dt.hour

    df["country"] = df["country"].replace(
        {"España": "Spain", "Deutschland": "Germany"}, regex=False
    )

    df["avg_spend_per_session"] = np.where(
        df["number_of_prior_session_count"] > 0,
        df["past_total_spend_before_session"] / df["number_of_prior_session_count"],
        0.0,
    )
    df["has_past_spend"] = (df["past_total_spend_before_session"] > 0).astype(int)

    # gamma:add 0.000001 to replace 0 value in label_session_spend. it will be used at gamma regression later
    # df_input['label_session_spend'] = np.where(df_input['label_session_spend'] == 0, 0.000001,df_input['label_session_spend'])


    df_proc = df.drop(
        [
            "session_id",
            "user_id",
            "session_start",
            "day_num",
            "number_of_prior_session_count",
            "past_total_spend_before_session",
        ],
        axis=1,
    )

    # reg:gamma - filter label_session_spend >0. otherwise, it will return "label must be positive for gamma regression"
    # df_proc = df_proc[df_proc['label_session_spend'] > 0]

    return df_proc

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        "session_start": [pd.Timestamp("2024-01-01 10:00:00")],
        "country": ["España"],
        "number_of_prior_session_count": [2],
        "past_total_spend_before_session": [100.0],
        "session_id": ["s1"],
        "user_id": ["u1"]
    })

    df_prepare = prepare_data(sample_data)
    print("Test Result Columns:", df_prepare.columns.tolist())
    print("Extractor test passed")