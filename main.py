import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np

# Set page title
st.title('USA College Admission Rate Prediction')

# Load and display an image
image = Image.open('college_admission.jpeg')
st.image(image, caption='College Admission', use_column_width=True)

# Load the pre-trained model
input = open('lr_admit.pkl', 'rb')
model = pkl.load(input)

# Display header for input admission information
st.header('Input Admission Information')

# Input form for user
gre = st.number_input('Insert GRE Score')
toefl = st.number_input('Insert TOEFL Score')
uni_rate = st.number_input('Insert University Rating')
sop = st.number_input('Insert SOP')
lor = st.number_input('Insert LOR')
cgpa = st.number_input('Insert CGPA')
research = st.radio('Choose Research Experience', [0, 1], index=None)

# Check if all input fields are filled
if all([gre, toefl, uni_rate, sop, lor, cgpa, research]):
    # Make prediction when the 'Predict' button is clicked
    if st.button('Predict'):
        feature_vector = np.array([gre, toefl, uni_rate, sop, lor, cgpa, research]).reshape(1, -1)
        result = model.predict(feature_vector)[0]

        # Display the result
        st.header('Result')
        st.markdown(f'The predicted admission rate is: **{result:.2%}**')
