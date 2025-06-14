import pandas as pd
import joblib
from src.logger import get_logger

logger = get_logger(__name__)

def run_batch_prediction(model_path, input_path, output_path):
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    logger.info(f"Reading batch input data from {input_path}")
    df = pd.read_csv(input_path)
    original_ids = df.get("id")

    df.drop(columns=["Unnamed: 0", "id"], inplace=True, errors='ignore')
    df["Arrival Delay in Minutes"].fillna(df["Arrival Delay in Minutes"].median(), inplace=True)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes

    predictions = model.predict(df)
    result_df = pd.DataFrame({"id": original_ids, "prediction": predictions})
    result_df.to_csv(output_path, index=False)
    logger.info(f"Predictions written to {output_path}")