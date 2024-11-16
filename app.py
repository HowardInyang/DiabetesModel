import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('diabetes_prediction_model.pkl')

# Function to predict diabetes
def predict_diabetes(data):
    prediction = model.predict(data)
    return prediction

def main():
    st.title("Diabetes Prediction Application")

    # Introduction
    st.write("Welcome to the Diabetes Prediction App. Please enter the patient data below to predict the presence of diabetes.")
    
    # Set default values in an expander
    with st.expander("Click here to view or edit default values"):
        default_values = {
            'Number of times pregnant': 0,
            'Plasma glucose concentration': 148,
            'Diastolic blood pressure': 72,
            'Triceps skin fold thickness': 35,
            '2-Hour serum insulin': 0,
            'Body mass index': 33.6,
            'Diabetes pedigree function': 0.627,
            'Age': 50
        }

    # Collect user input with columns
    col1, col2 = st.columns(2)
    with col1:
        preg = st.number_input("Number of times pregnant:", min_value=0, max_value=17, step=1, value=default_values['Number of times pregnant'])
        glucose = st.number_input("Plasma glucose concentration:", min_value=0, max_value=200, step=1, value=default_values['Plasma glucose concentration'])
        bp = st.number_input("Diastolic blood pressure:", min_value=0, max_value=122, step=1, value=default_values['Diastolic blood pressure'])
        skin_thickness = st.number_input("Triceps skin fold thickness:", min_value=0, max_value=99, step=1, value=default_values['Triceps skin fold thickness'])
    
    with col2:
        insulin = st.number_input("2-Hour serum insulin:", min_value=0, max_value=846, step=1, value=default_values['2-Hour serum insulin'])
        bmi = st.number_input("Body mass index:", min_value=0.0, max_value=67.1, step=0.1, value=default_values['Body mass index'])
        dpf = st.number_input("Diabetes pedigree function:", min_value=0.078, max_value=2.42, step=0.001, value=default_values['Diabetes pedigree function'])
        age = st.number_input("Age:", min_value=21, max_value=81, step=1, value=default_values['Age'])

    # Prepare input data for prediction
    input_data = np.array([[preg, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])

    # Make prediction with a visually distinct button
    if st.button("Predict", help="Click to predict diabetes based on the input data"):
        prediction = predict_diabetes(input_data)
        if prediction[0] == 1:
            st.metric(label="Prediction", value="Positive", delta="Diabetes", delta_color="inverse")
        else:
            st.metric(label="Prediction", value="Negative", delta="No Diabetes")

if __name__ == "__main__":
    main()
