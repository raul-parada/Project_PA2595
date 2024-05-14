from hyperopt import tpe, STATUS_OK, Trials, hp, fmin
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import ray
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Initialize Ray
ray.init()

# Set up FastAPI
app = FastAPI()

N_FOLDS = 4

# Data loading and preprocessing
df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', header=None)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9, stratify=y)
mx = MinMaxScaler()
X_train = mx.fit_transform(X_train)
X_test = mx.transform(X_test)


mlflow.set_tracking_uri("http://127.0.0.1:8001")

mlflow.set_experiment('Hyperopt_Optimization')

# Hyperparameter space
space ={
    'warm_start': hp.choice('warm_start', [True, False]),
    'fit_intercept': hp.choice('fit_intercept',[True, False]),
    'tol': hp.uniform('tol', 0.00001, 0.001),
    'C': hp.uniform('C', 0.05, 3),
    'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
    'max_iter': hp.choice('max_iter', range(5, 1000))
}

# Function to evaluate
def evaluate(params):
    with mlflow.start_run(nested=True):
        clf = LogisticRegression(**params, random_state=0)
        scores = cross_val_score(clf, X_train, y_train, cv=N_FOLDS, scoring='f1_macro')
        best_score = max(scores)
        if 'best_score' not in evaluate.__dict__ or best_score > evaluate.best_score:
            evaluate.best_score = best_score
            evaluate.best_params = params
        loss = 1 - best_score
        return {'loss': loss, 'params': params, 'status': STATUS_OK}

# Run hyperparameter optimization
with mlflow.start_run(run_name='hyper_opt_logistic') as run:
    # Convert any non-serializable values to serializable types
    best = fmin(fn=evaluate, space=space, algo=tpe.suggest, max_evals=2, trials=Trials())
    best_serializable = {k: float(v) if isinstance(v, np.int64) else v for k, v in best.items()}
    mlflow.log_dict(best_serializable, "best_params.json")
    
    # Train the best model found on the entire training set
    best_model = LogisticRegression(**evaluate.best_params, random_state=0)
    best_model.fit(X_train, y_train)
    
    # Log test score
    test_score = best_model.score(X_test, y_test)
    mlflow.log_metric("test_score", test_score)

    # Log best model
    mlflow.sklearn.log_model(best_model, "best_model", registered_model_name="best_model")

# Define input data model for FastAPI
class InputData(BaseModel):
    preg: int
    plas: int
    pres: int
    skin: int
    insu: int
    mass: float
    pedi: float
    age: int

# Define FastAPI predict endpoint
@app.post("/predict/")
async def predict(data: InputData):
    input_data = np.array([[
        data.preg, data.plas, data.pres, data.skin, data.insu, data.mass, data.pedi, data.age
    ]])
    prediction = best_model.predict(input_data)
    return {"prediction": prediction.tolist()}

# Define a route for model inference
@app.post("/inference/")
def inference(data: dict):
    try:
        feature_values = data.get("features")
        if feature_values is None:
            raise HTTPException(status_code=422, detail="Missing 'features' field in request body")
        features_array = np.array(feature_values).reshape(1, -1)
        prediction = best_model.predict(features_array)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)

