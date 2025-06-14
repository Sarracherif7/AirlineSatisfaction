import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.logger import get_logger

logger = get_logger(__name__)

def load_and_preprocess(filepath):
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    df.drop(columns=["Unnamed: 0", "id"], inplace=True, errors='ignore')
    df["Arrival Delay in Minutes"].fillna(df["Arrival Delay in Minutes"].median(), inplace=True)

    label_encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        logger.info(f"Encoded column: {col}")

    X = df.drop(columns=["satisfaction"])
    y = df["satisfaction"]
    return X, y, label_encoders