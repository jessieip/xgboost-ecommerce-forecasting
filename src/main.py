import logging

from src.database import extract_data
from src.extractor import prepare_data
from src.model import train_model
from src.output import plot_residuals, save_result, shap_analysis
from src.processing import optimise_xgboost, prepare_var

import os
import shutil

# check output folder exists or not
if os.path.exists('outputs'):
    shutil.rmtree('outputs') #delete old folder
os.makedirs('outputs')

# setting up logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main() -> None:

    # 1. getting data from database
    logger.info('Extracting data from BigQuery...')
    data = extract_data(project_id = "first-project-321219")

    if data is None or data.empty:
        logger.error('No data extracted.')


    # 2. preparing data
    logger.info('Preparing and cleaning data...')
    prepare = prepare_data(data)

    # 3. processing data and find best params
    logger.info('Splitting data into Train/Val/Test sets...')
    x_train, y_train, x_val, y_val, x_test, y_test = prepare_var(
                    df = prepare,
                    target = "label_session_spend",
                    test_size = 0.2,
                    random_state = 16
                    )

    logger.info('Finding Best Hyperparameters via Optuna...')
    best_params = optimise_xgboost(x_train, y_train, x_val, y_val)
    logger.info(f'Best Hyperparameters: {best_params}')

    # 4. train the XG Boost model
    logger.info('Training XGBoost model...')
    model, preds, importance = train_model(x_train, y_train, x_test, y_test, best_params)

    # 5. export result to csv file
    logger.info('Saving output...')
    save_result(y_test, preds)
    plot_residuals(y_test,  preds)
    shap_analysis(model, x_test)

if __name__ == "__main__":
    main()
