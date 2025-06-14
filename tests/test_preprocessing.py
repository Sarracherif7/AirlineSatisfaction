import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_and_preprocess


def test_preprocessing():
    from src.data_preprocessing import load_and_preprocess
    X, y, encoders = load_and_preprocess("datasets/train.csv")
    assert X.shape[0] == y.shape[0], "Mismatch in feature and label row counts"
    assert len(encoders) > 0, "No encoders returned"