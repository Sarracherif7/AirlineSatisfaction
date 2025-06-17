#!/bin/bash

echo "➡ Resetting MLflow experiment..."
.venv/bin/python -m src.reset_mlflow_experiment

echo "➡ Training model..."
.venv/bin/python -m src.train_model
