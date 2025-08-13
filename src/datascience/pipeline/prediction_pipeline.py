import joblib
import numpy as np
import pandas as pd
from pathlib import Path


class PredictionPipeline:
    def __init__(self):
        # Load the trained model once during initialization
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
    
    def predict(self, data: pd.DataFrame):
        # Make predictions
        return self.model.predict(data)


if __name__ == "__main__":
    # Load test data
    data = pd.read_csv("/workspaces/Data-Science-Project-01/artifacts/data_transformation/test.csv")
    
    # Remove target column
    features = data.drop(columns=['quality'])
    
    # Create pipeline and make predictions
    prediction_pipeline = PredictionPipeline()
    predictions = prediction_pipeline.predict(features)
    
    print(predictions)
