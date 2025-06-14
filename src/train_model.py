from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib
import os
import mlflow
from src.data_preprocessing import load_and_preprocess
from src.logger import get_logger

logger = get_logger(__name__)
mlflow.set_experiment("customer-satisfaction")

def train_and_save_model(data_path, model_path):
    logger.info("Starting training pipeline")
    X, y, encoders = load_and_preprocess(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(random_state=42)

    with mlflow.start_run():
        model.fit(X_train_resampled, y_train_resampled)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_params(model.get_params())
        logger.info("Model training completed and logged with MLflow")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    return model, X_test, y_test