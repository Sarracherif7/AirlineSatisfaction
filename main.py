from src.train_model import train_and_save_model
from src.evaluate_model import evaluate
from src.predict_batch import run_batch_prediction
from src.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    model, X_test, y_test = train_and_save_model("datasets/train.csv", "models/model.pkl")
    report = evaluate(model, X_test, y_test)
    print("\nClassification Report:\n", report)
    logger.info("Pipeline execution completed")

    # Optional: run batch prediction
    run_batch_prediction("models/model.pkl", "batch_prediction_dataset/dataset.csv", "batch_prediction_dataset/results.csv")