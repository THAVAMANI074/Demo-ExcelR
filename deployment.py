import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Titanic Survival Prediction')

pclass = st.selectbox('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])
sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
age = st.slider('Age', 0, 100, 25)
sibsp = st.slider('Number of Siblings/Spouses Aboard', 0, 10, 0)
parch = st.slider('Number of Parents/Children Aboard', 0, 10, 0)
fare = st.number_input('Fare', 0.0, 600.0, 50.0)
embarked = st.selectbox('Port of Embarkation (0 = C, 1 = Q, 2 = S)', [0, 1, 2])

features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)
probability = model.predict_proba(features_scaled)[0][1]

st.write('Prediction:', 'Survived' if prediction[0] == 1 else 'Did Not Survive')
st.write('Survival Probability:', f'{probability * 100:.2f}%')
