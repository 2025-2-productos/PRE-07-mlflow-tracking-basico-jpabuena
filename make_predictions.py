import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"

df = pd.read_csv(FILE_PATH)
y = df["quality"]
x = df.drop(columns=["quality"])

logged_model = "runs:/3d9e8fc5d7f440e187f4863a7f068c04/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
y = loaded_model.predict(x)

print(y)
