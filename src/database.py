"""Import libraries that is used for the model"""
from google.cloud import bigquery
import pandas as pd

def extract_data(project_id:str, query:str) -> pd.DataFrame:
    """
        Extracting data process via Bigquery.
        Args:
            project_id: Bigquery project ID
        Returns:
            Dataframe with session_id, user_id, demographic columns,
            past_total_spend_before_session, etc
    """
    client = bigquery.Client(project=project_id)
    query = """ 
    WITH session_windows AS (
        -- Step 1: Define the start and end of every session
        SELECT
            session_id,
            user_id,
            MIN(created_at) AS session_start,
            MAX(created_at) AS session_end
        FROM `bigquery-public-data.thelook_ecommerce.events`
        WHERE created_at >= '2022-01-01'
        GROUP BY 1, 2
    ),
    order_totals AS (
        -- Step 2: Get total value for every order
        SELECT
            user_id,
            order_id,
            created_at,
            SUM(sale_price) AS total_value
        FROM `bigquery-public-data.thelook_ecommerce.order_items`
        GROUP BY 1, 2, 3
    )
    
    SELECT
        sw.session_id,
        sw.user_id,
        u.age,
        u.gender,
        u.country,
        u.traffic_source,
        sw.session_start,
    
        -- FEATURE: Total spend BEFORE this session started
        COALESCE((
            SELECT SUM(ot.total_value)
            FROM order_totals ot
            WHERE ot.user_id = sw.user_id
              AND ot.created_at < sw.session_start
        ), 0) AS past_total_spend_before_session,
    
        -- FEATURE: Session count prior to this one
        COUNT(*) OVER (
            PARTITION BY sw.user_id
            ORDER BY sw.session_start
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS number_of_prior_session_count,
    
        -- TARGET: Total spend WITHIN this session
        -- We look for orders by this user that happened between the session start and end
        COALESCE((
            SELECT SUM(ot.total_value)
            FROM order_totals ot
            WHERE ot.user_id = sw.user_id
              AND ot.created_at BETWEEN sw.session_start AND sw.session_end
        ), 0) AS label_session_spend
    
    FROM session_windows sw
    JOIN `bigquery-public-data.thelook_ecommerce.users` u ON sw.user_id = u.id
    ORDER BY sw.user_id, sw.session_start
    """

    return client.query(query).to_dataframe()

if __name__ == "__main__":
    df = extract_data(project_id = "first-project-321219")
    print(df.head())