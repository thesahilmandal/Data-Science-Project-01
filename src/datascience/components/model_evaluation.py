import tempfile
import pandas as pd
import numpy as np
import joblib
import mlflow
from pathlib import Path
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.datascience.config.configuration import ModelEvaluationConfig
from src.datascience.utils.common import save_json

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def log_into_mlflow(self):
        # Load test data and model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop(columns=[self.config.target_column])
        test_y = test_data[self.config.target_column]

        mlflow.set_registry_uri(self.config.mlflow_uri)

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)

            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

            # Save metrics locally
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(Path(self.config.metric_file_name), data=scores)

            # Log parameters and metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # Save model locally and log as artifact (no registry)
            with tempfile.TemporaryDirectory() as tmp_dir:
                model_path = Path(tmp_dir) / "model.pkl"
                joblib.dump(model, model_path)
                mlflow.log_artifact(str(model_path), artifact_path="model")
