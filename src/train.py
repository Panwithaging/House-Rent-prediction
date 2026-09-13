from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np
import pandas as pd

from preprocessing import preprocessed
from model import HouseRentPrice

import torch
from torch import nn

import joblib
import pathlib as path

BASE_DIR=path.Path(__file__).resolve().parent.parent
pune=pd.read_csv(BASE_DIR/"data"/"Indian_housing_Pune_data.csv")
delhi=pd.read_csv(BASE_DIR/"data"/"Indian_housing_Delhi_data.csv")
mumbai=pd.read_csv(BASE_DIR/"data"/"Indian_housing_Mumbai_data.csv")

df=pd.concat([delhi,pune,mumbai],ignore_index=True)
df=preprocessed(df)

x=df.drop(columns="price")
y=np.log1p(df["price"])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

cat_cols = [
    "location",
    "city",
    "TYPE"
]

num_cols = [
    "house_size",
    "latitude",
    "longitude",
    "numBathrooms",
    "numBalconies",
    "bedrooms"
]

Preprocessor=ColumnTransformer(transformers=[("cat",OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat_cols),
                                            ("num",StandardScaler(),num_cols)])

x_train = Preprocessor.fit_transform(x_train)
x_test = Preprocessor.transform(x_test)

x_train=torch.tensor(x_train,dtype=torch.float)
x_test=torch.tensor(x_test,dtype=torch.float)

y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

print(x_train.shape)
modelv0=HouseRentPrice(input_features=x_train.shape[1])

device="cuda" if torch.cuda.is_available() else "cpu"
device

modelv0.to(device)

loss_fn=nn.MSELoss()
opt=torch.optim.Adam(modelv0.parameters(),lr=0.01)

x_train,y_train=x_train.to(device),y_train.to(device)
x_test,y_test=x_test.to(device),y_test.to(device)

torch.manual_seed(42)
epochs=1200
for epoch in range(epochs):
    modelv0.train()
    pred=modelv0(x_train)
    loss=loss_fn(y_train,pred)
    opt.zero_grad()
    loss.backward()
    opt.step()
    modelv0.eval()
    with torch.inference_mode():
        test_pred=modelv0(x_test)
        test_loss=loss_fn(y_test,test_pred)
    if epoch%10==0:
        print(f"Epoch :{epoch} | loss : {loss:.4f} | test_loss : {test_loss:.4f}")


with torch.inference_mode():
    final_pred=modelv0(x_test)
final_pred

pred = final_pred.cpu().numpy().flatten()
true = y_test.cpu().numpy().flatten()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
pred_price = np.expm1(pred)
true_price = np.expm1(true)

print("MAE :", mean_absolute_error(true_price, pred_price))
print("RMSE:", np.sqrt(mean_squared_error(true_price, pred_price)))
print("R²  :", r2_score(true_price, pred_price))

torch.save(modelv0.state_dict(),BASE_DIR/"model"/"house_rent_ann.pth")

joblib.dump(Preprocessor,BASE_DIR/"model"/"preprocessor.pkl")

print("done")