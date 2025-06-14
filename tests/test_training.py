def test_training_pipeline():
    from src.train_model import train_and_save_model
    model, X_test, y_test = train_and_save_model("datasets/train.csv", "models/test_model.pkl")
    assert hasattr(model, "predict"), "Trained model lacks predict method"
    assert X_test.shape[0] == y_test.shape[0], "Test features and labels row count mismatch"