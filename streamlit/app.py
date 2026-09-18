import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
# 1. Load the pre-trained model
model = joblib.load('mymodel.pkl')

# 2. Set up the web page title
st.title("Machine Learning Prediction App for Iris Flower type")

# 3. Create input elements for features
sepal_length = st.number_input("Enter Iris flower sepal length", min_value=0.0, max_value=100.0, value=5.1)
sepal_width = st.slider("Select Iris flower sepal width", 0.0, 50.0, 3.5)
petal_length = st.number_input(
    "Enter Iris flower petal length", min_value=0.0, max_value=100.0, value=1.4
)
petal_width = st.slider("Select  Iris flower petal width", 0.0, 100.0, 0.2)

# 4. Make prediction when user clicks the button
if st.button("Predict"):
    # Format inputs into the array shape expected by the model
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    
    # 5. Display the result
    st.success(f"The model predicted class: {prediction[0]}")
    st.success(f"The model predicted flower:: {iris.target_names[prediction[0]]}")
