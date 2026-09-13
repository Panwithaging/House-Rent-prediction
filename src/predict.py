import torch
import joblib
import pathlib as Path
import numpy as np


from preprocessing import preprocessed
from model import HouseRentPrice

BASE_Path=Path.Path(__file__).resolve().parent.parent
#PREP_path=PREP_BASE_Path/"model"/"preproce"
device="cuda" if torch.cuda.is_available() else "cpu"

Ann_model=HouseRentPrice(input_features=659).to(device)

prep=joblib.load(BASE_Path/"model"/"preprocessor.pkl")

Ann_model.load_state_dict(torch.load(BASE_Path/"model"/"house_rent_ann.pth",map_location=device))

Ann_model.eval()

def prediction(data):
    data=preprocessed(data)
    data=prep.transform(data)

    data=torch.tensor(data,dtype=torch.float32).to(device)

    with torch.inference_mode():
        pred=Ann_model(data)

    pred=torch.expm1(pred)

    return pred.cpu().numpy()[0][0]
