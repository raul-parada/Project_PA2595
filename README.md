# Project_PA2595
This repository contains a basic example of MLOps

This project aims to run a classification application using:
- MLFLow
- Hyperparametrization
- Ray
- FastApi

The application is running locally. How to run it?
1. Check the requirements.txt to install the required libraries to run the app (not all libraries are required).
2. The version of Python is Python 3.10.12
3. Start mlflow ´mlflow server --host 127.0.0.1 --port 8001´
   ![Screenshot from 2024-05-14 10-33-59](https://github.com/raul-parada/Project_PA2595/assets/8438920/5932e48a-e340-4a72-b44c-45423a2409bd)

4. In scenario 1 folder, execute 'python3 train_serve_iris.py'

5. uvicorn will be ready to receive requests to predict iris specieis

6. Send a request using curl

`curl --location 'http://127.0.0.1:8004/predict' \
--header 'Content-Type: application/json' \
--data '{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}'`

Response will be
`{"species":"setosa"}`

7. In scenario 2 folder, execute 'python3 ray_hypB_FA.py'
![Screenshot from 2024-05-14 10-37-02](https://github.com/raul-parada/Project_PA2595/assets/8438920/bdf5696b-18be-4c85-8cc7-3b962a99bcc0)

8. Uvicorn will be ready to receive requests to predict patient's diagnosis
   ![Screenshot from 2024-05-14 10-37-19](https://github.com/raul-parada/Project_PA2595/assets/8438920/3476ad32-e302-49a3-a792-668d2c3b0c5e)

9. Open another terminal and execute the curl (from curl.txt)
   ![Screenshot from 2024-05-14 10-38-30](https://github.com/raul-parada/Project_PA2595/assets/8438920/8d45e0b0-af48-4b0f-a267-ad628e1bf361)

   
