import mlflow
from mlflow.models import infer_signature
from pydantic import BaseModel
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fastapi import FastAPI, HTTPException
import numpy as np
from sklearn.datasets import load_iris

app = FastAPI()

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

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

# Define a request model
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define the /predict endpoint
@app.post("/predict")
def predict(iris_request: IrisRequest):
    data = np.array([[iris_request.sepal_length, iris_request.sepal_width, iris_request.petal_length, iris_request.petal_width]])
    prediction = rf.predict(data)
    species = iris.target_names[prediction[0]]
    return {"species": species}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8004)
