import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fastapi import FastAPI, HTTPException
import numpy as np

app = FastAPI()

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model hyperparameters for Random Forest
params = {
    "n_estimators": 100,  # Number of trees in the forest
    "random_state": 8888,  # Seed for random number generation
    "max_depth": 4
}

# Train the Random Forest model
rf = RandomForestClassifier(**params)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

# Set the tracking server URI for logging
mlflow.set_tracking_uri("http://127.0.0.1:8001")

# Define MLflow experiment name
experiment_name = "Iris_Classification"
mlflow.set_experiment(experiment_name)

# Start an MLflow run
with mlflow.start_run() as run:
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the accuracy metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag to describe the training information
    mlflow.set_tag("Training Info", "Basic Random Forest model for Iris data")

    # Infer the model signature
    signature = infer_signature(X_train, rf.predict(X_train))

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=rf,
        artifact_path="iris_model",
        signature=signature,
        input_example=pd.DataFrame(X_train).to_json(orient="split"),
        registered_model_name="tracking-quickstart"
    )

# Define a route for model inference
@app.post("/predict")
def predict(features: dict):
    try:
        feature_values = features.get("features")
        if feature_values is None:
            raise HTTPException(status_code=422, detail="Missing 'features' field in request body")
        # Convert features to numpy array
        features_array = np.array(feature_values).reshape(1, -1)
        # Make prediction using the loaded model
        prediction = rf.predict(features_array)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8004)

