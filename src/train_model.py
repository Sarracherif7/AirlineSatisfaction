from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib
import os
import mlflow
import mlflow.sklearn
from src.data_preprocessing import load_and_preprocess
from src.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Set the MLflow tracking server URI (local server)
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Set or create experiment
mlflow.set_experiment("airline-satisfaction")

def train_and_save_model(data_path, model_path):
    logger.info("Starting training pipeline")

    # Load and preprocess data
    X, y, encoders = load_and_preprocess(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Initialize model
    model = RandomForestClassifier(random_state=42)

    # Start MLflow run
    with mlflow.start_run(run_name="RandomForest_SMOTE_Run") as run:
        # Fit model
        model.fit(X_train_resampled, y_train_resampled)

        # Log model
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_params(model.get_params())
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("test_accuracy", accuracy)

        logger.info(f"Model training completed. Test accuracy: {accuracy:.4f}")
        logger.info("Model logged with MLflow")

        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(model_uri, "airline-satisfaction-model")
        logger.info(f"Registered model: {result.name}, version: {result.version}")

    # Save model locally
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    return model, X_test, y_test

if __name__ == "__main__":
    train_and_save_model("datasets/train.csv", "models/random_forest_model.pkl")
