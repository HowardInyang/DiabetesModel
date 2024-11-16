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
    # App title with style
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>Diabetes Prediction Application</h1>",
        unsafe_allow_html=True,
    )

    # Introduction with subheader
    st.subheader("Welcome!")
    st.write(
        "This application uses a trained machine learning model to predict the likelihood of diabetes based on patient data. "
        "Please fill in the fields below, and click **Predict** to get your result."
    )
    st.write("---")

    # Default values in an expander for guidance
    with st.expander("Default Values Guide (Click to View)"):
        st.info(
            """
            **Default Values:**\n
            - **Number of times pregnant:** 0\n
            - **Plasma glucose concentration:** 148\n
            - **Diastolic blood pressure:** 72\n
            - **Triceps skin fold thickness:** 35\n
            - **2-Hour serum insulin:** 0\n
            - **Body mass index:** 33.6\n
            - **Diabetes pedigree function:** 0.627\n
            - **Age:** 50
            """
        )

    # Input fields with styling
    st.markdown("<h3 style='color: #4CAF50;'>Patient Information</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        preg = st.number_input(
            "Number of times pregnant",
            min_value=0,
            max_value=17,
            step=1,
            value=0,
            help="Number of pregnancies the patient has had.",
        )
        glucose = st.number_input(
            "Plasma glucose concentration",
            min_value=0,
            max_value=200,
            step=1,
            value=148,
            help="Plasma glucose concentration measured 2 hours after a meal.",
        )
        bp = st.number_input(
            "Diastolic blood pressure (mm Hg)",
            min_value=0,
            max_value=122,
            step=1,
            value=72,
            help="Blood pressure reading.",
        )
        skin_thickness = st.number_input(
            "Triceps skin fold thickness (mm)",
            min_value=0,
            max_value=99,
            step=1,
            value=35,
            help="Thickness of the skin on the triceps.",
        )

    with col2:
        insulin = st.number_input(
            "2-Hour serum insulin (µU/ml)",
            min_value=0,
            max_value=846,
            step=1,
            value=0,
            help="Insulin levels measured after 2 hours.",
        )
        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=0.0,
            max_value=67.1,
            step=0.1,
            value=33.6,
            help="Body Mass Index, calculated as weight in kg/(height in m)^2.",
        )
        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.078,
            max_value=2.42,
            step=0.001,
            value=0.627,
            help="A score indicating the genetic likelihood of diabetes.",
        )
        age = st.number_input(
            "Age (years)",
            min_value=21,
            max_value=81,
            step=1,
            value=50,
            help="Age of the patient.",
        )

    # Prepare input data for prediction
    input_data = np.array([[preg, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])

    # Call-to-action button
    st.markdown("<h3 style='color: #4CAF50;'>Predict Diabetes</h3>", unsafe_allow_html=True)
    if st.button("Predict", help="Click to predict diabetes based on the input data"):
        with st.spinner("Predicting..."):
            prediction = predict_diabetes(input_data)
        if prediction[0] == 1:
            st.success(
                "The prediction is **Positive**: The patient is likely to have diabetes.",
                icon="✅",
            )
            st.balloons()
        else:
            st.info(
                "The prediction is **Negative**: The patient is unlikely to have diabetes.",
                icon="❎",
            )

    # Footer
    st.markdown(
        "<footer style='text-align: center; margin-top: 20px; color: gray;'>"
        "Developed with ❤️ using Streamlit."
        "</footer>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
