# python -m venv myenv # activate environment # myenv\Scripts\activate # pip install streamlit scikit-learn pandas seaborn numpy import pickle
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# load model
model = pickle.load(open(r'gb_model.pkl','rb'))

# scaler
scaler = StandardScaler()


# title
st.title("House Price Prediction App")

# input variables
sqft = st.number_input('Square Footage', min_value=300, max_value=10000, value=1500)
bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)
year_built = st.number_input('Year Built', min_value=1900, max_value=2025, value=2000)

# create dataframe
input_features = pd.DataFrame({
    'Square_Footage': [sqft],
    'Num_Bedrooms': [bedrooms],
    'Num_Bathrooms': [bathrooms],
    'Year_Built': [year_built],
    
})

# apply scaling (only if your model was trained with scaling)
input_scaled = scaler.fit_transform(input_features)

# prediction
if st.button('Predict'):
    prediction = model.predict(input_scaled)
    output = round(prediction[0], 2)
    st.success(f'Estimated House Price: ${output}')