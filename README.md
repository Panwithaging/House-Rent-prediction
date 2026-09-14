# House Rent Prediction using Artificial Neural Network (ANN)

A deep learning project that predicts monthly house rent in Indian cities using an Artificial Neural Network (PyTorch). The application includes an interactive Streamlit dashboard with map-based location selection.

webpage:-https://house-rent-predictiongit-yccdq9mwuickqoru3cbgne.streamlit.app/

---

## Features

- Predict monthly house rent using an ANN model
- Interactive Streamlit web application
- Supports multiple cities:
  - Delhi
  - Mumbai
  - Pune
  - Hisar (demo)
- Clickable interactive map (Folium)
- Custom house details input
- Automatic preprocessing pipeline using scikit-learn
- Trained PyTorch neural network

---

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Folium
- Joblib

---

## Project Structure

```
House Rent Prediction/
│
├── data/
│   ├── Indian_housing_Delhi_data.csv
│   ├── Indian_housing_Mumbai_data.csv
│   └── Indian_housing_Pune_data.csv
│
├── model/
│   ├── house_rent_ann.pth
│   └── preprocessor.pkl
│
├── src/
│   ├── app.py
│   ├── preprocessing.py
│   ├── predict.py
│   └── model.py
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Panwithaging/House-Rent-Prediction.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run src/app.py
```

---

## Model

The model is an Artificial Neural Network (ANN) built using PyTorch.

### Input Features

- House Type
- Bedrooms
- House Size
- Bathrooms
- Balconies
- Furnishing Status
- Negotiable
- Location
- City
- Latitude
- Longitude

The preprocessing pipeline handles:

- Feature engineering
- Missing values
- Label encoding
- One-hot encoding
- Numeric preprocessing

---

## Streamlit Dashboard

Users can

- Select city
- Select location
- Choose house type
- Enter house size
- Select bathrooms and balconies
- Choose furnishing status
- Select negotiable option
- Click on the interactive map to adjust coordinates
- Predict monthly house rent instantly

---

## Future Improvements

- Add more Indian cities
- Deploy on Streamlit Cloud
- Improve ANN accuracy with hyperparameter tuning
- Display confidence intervals
- Add rental price trend visualizations

---

## Author

**Tushar Panging**

GitHub: https://github.com/Panwithaging

LinkedIn: www.linkedin.com/in/pangingtushar54
