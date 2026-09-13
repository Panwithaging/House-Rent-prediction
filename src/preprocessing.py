import pandas as pd
import numpy as np

def preprocessed(df):

    df[["BHK","TYPE"]]=df["house_type"].str.extract(r"(\d+\s*(?:BHK|RK))\s+(.+)")

    df["bedrooms"] = df["house_type"].str.extract(r"(\d+)").astype(float)

    df["house_size"] = (df["house_size"].astype(str).str.replace(",", "", regex=False).str.extract(r"(\d+)").astype(float))

    df["numBathrooms"]=df["numBathrooms"].fillna(df.groupby("bedrooms")["numBathrooms"].transform("median"))

    df["numBalconies"]=df["numBalconies"].fillna(df.groupby("bedrooms")["numBalconies"].transform("median"))

    df["numBalconies"] = df["numBalconies"].fillna(
    df["numBalconies"].median())

    df["isNegotiable"] = df["isNegotiable"].notna().astype(int)

    df.drop(columns=["house_type","currency","priceSqFt","verificationDate","description","SecurityDeposit","BHK"],inplace=True)

    df["Status"]=df["Status"].map({"Unfurnished":0,
                              "Semi-Furnished":1,
                              "Furnished":2})

    return df


