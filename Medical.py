import numpy as np
import joblib
import streamlit as st

# Loading the saved model
loaded_model = joblib.load(open("C:/Users/Mandar/Downloads/Project File/My project/PROJECT/joblib_model.sav", 'rb'))




# Function for Prediction
def medical_insurance_cost_prediction(input_data):
    # Changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    return prediction


def main():
    # Setting the page title with emoji and color
    st.markdown('<h1 style="color:blue;">Medical Insurance Cost Prediction 🩺</h1>', unsafe_allow_html=True)

    # Getting input from the user
    st.sidebar.header('Input Features:')
    age = st.sidebar.text_input('Age')
    sex = st.sidebar.text_input('Sex: 0 -> Female, 1 -> Male')
    bmi = st.sidebar.text_input('Body Mass Index')
    children = st.sidebar.text_input('Number of Children')
    smoker = st.sidebar.text_input('Smoker: 0 -> No, 1 -> Yes')
    region = st.sidebar.text_input('Region of Living: 0 -> NorthEast, 1-> NorthWest, 2-> SouthEast, 3-> SouthWest')

    # Code for prediction
    diagnosis = ''

    # Getting the input data from the user
    if st.sidebar.button('Predicted Medical Insurance Cost'):
        diagnosis = medical_insurance_cost_prediction([age, sex, bmi, children, smoker, region])
        st.write('\n')  # Adding some space for better alignment
        st.header('Predicted Medical Insurance Cost')
        st.write(f'The predicted amount is: ₹ {diagnosis[0]:.2f}', unsafe_allow_html=True)
        st.write('\n')


if __name__ == '__main__':
    main()
