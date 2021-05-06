import streamlit as st
import base64
import numpy as np
import pickle

#Design of application UI
LOGO_IMAGE = "stingrai-header.jpg"

st.markdown(
    """
    <style>
    .container {
        display: flex;
    }
    
    .logo-img {
        float:right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
    """,
    unsafe_allow_html=True
)


#Load Logistic Regression Model and define variable
pickle_in = open('Final_logreg_model.pk1','rb') #Load Logistic Regression model
rf_Model = pickle.load(pickle_in)


def predict_diabetes(number_times_pregnant,plasma_glucose_concentration,diastolic_bp, triceps_skin_fold,
                                two_hr_serum_insulin, BMI,diabetes_pedigree_function, age):
    input = np.array([[number_times_pregnant,plasma_glucose_concentration,diastolic_bp, triceps_skin_fold,
                                two_hr_serum_insulin, BMI,diabetes_pedigree_function, age]]).astype(np.float64)
    prediction = rf_Model.predict_proba(input)
    pred = '{0:.{1}f}'.format(prediction[0][0], 3)
    result = 1.0-float(pred)
    final_result = '{:.3f}'.format(result)
    print(pred)
    return float(final_result)


def main():
    html_temp = """
    <div style="background-color:#08475b;padding-top:5px;padding-bottom:5px;">
    <h2 style="color:white;text-align:center;">Stingr.ai Diabetes Prediction </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    number_times_pregnant = st.text_input("Number of times pregnant","0")
    plasma_glucose_concentration = st.text_input("Plasma Glucose Concentration","120")
    diastolic_bp = st.text_input("Diastolic Blood Pressure (mm Hg)","70")
    triceps_skin_fold = st.text_input("Triceps Skin Fold Thickness (mm)","10")
    two_hr_serum_insulin = st.text_input("2-Hour Serum Insulin (mu U/ml)","60")
    BMI = st.text_input("Body Mass Index","30")
    diabetes_pedigree_function = st.text_input("Diabetes Pedigree Function","0.320")
    age = st.text_input("Age","25")
    result = ""
    safe_html="""
      <div style="background-color:#3ff454;padding:10px >
       <h2 style="color:white;text-align:center;"> You're healthy and not at risk of developing diabetes</h2>
       </div>
    """
    danger_html="""
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> You're at risk of developing diabetes</h2>
       </div>
    """

    if st.button("Predict"):
        result=predict_diabetes(number_times_pregnant,plasma_glucose_concentration,diastolic_bp, triceps_skin_fold,
                                two_hr_serum_insulin, BMI,diabetes_pedigree_function, age)
        st.success('The probability of developing Type 2 Diabetes is {}'.format(result))

        if result > 0.5:
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()
