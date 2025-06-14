from sklearn.metrics import classification_report
from src.logger import get_logger

logger = get_logger(__name__)

def evaluate(model, X_test, y_test):
    logger.info("Evaluating model")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info("Evaluation complete")
    return report