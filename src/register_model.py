import mlflow

# Set the tracking URI (your local MLflow server)
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Initialize the client
client = mlflow.tracking.MlflowClient()

# Config: adjust these as needed
model_name = "airline-satisfaction-model"
run_id = "796d794d7fb340c094c8705d6e2c927e"  # Your latest run ID
model_uri = f"runs:/{run_id}/model"

# Step 1: Create the registered model if it doesn't exist
try:
    client.create_registered_model(model_name)
    print(f"Registered model created: {model_name}")
except mlflow.exceptions.RestException:
    print(f"Model {model_name} already exists in the registry")

# Step 2: Register a new model version
model_version = client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=run_id
)
print(f"Model version {model_version.version} registered successfully.")

# Step 3: Transition to Staging (optional)
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Staging"
)
print(f"Model version {model_version.version} transitioned to Staging.")
