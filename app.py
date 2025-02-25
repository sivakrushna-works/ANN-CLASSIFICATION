import joblib
import streamlit as st
import pandas as pd

def main():
    st.title("Bank Customer Churn")

    credit_score = st.number_input("Credit Score")
    country = st.selectbox("Country", options=['France', 'Spain', 'Germany'])

    if country:
        st.success(country)

    gender = st.radio("Select Gender: ", ('Male', 'Female'))
    if (gender == 'Male'):
        st.success("Male")
    else:
        st.success("Female")
    
    age = st.number_input("Age")
    tenure = st.number_input("Tenure")
    balance = st.number_input("Balance")
    products_number = st.number_input("Products Number")
    credit_card = st.number_input("Credit Card")
    active_member = st.number_input("Active Member")
    estimated_salary = st.number_input("Estimated Salary")
    submit_button = st.button("Submit")

    if submit_button:
        # Process the form data
        process_form_data(credit_score, country, gender, age, tenure,
                          balance, products_number, credit_card, active_member, estimated_salary)

def process_form_data(credit_score, country, gender, age, tenure,
                          balance, products_number, credit_card, active_member, estimated_salary):


    encoders = joblib.load('encoders .joblib')
    model = joblib.load('rf_model.joblib')
    scaler = joblib.load('StandardScaler.joblib')


    dataDict = {'credit_score': credit_score, 'country': country, 'gender': gender, 'age': age, 'tenure':tenure,
                          'balance':balance, 'products_number': products_number, 'credit_card': credit_card,
                'active_member': active_member, 'estimated_salary': estimated_salary}

    df = pd.DataFrame([dataDict])

    decodedData = df.copy()
    for col in ['country', 'gender']:
        encoder = encoders[col]
        decodedData[col] = encoder.transform(decodedData[col])
    #
    decodedData = scaler.transform(decodedData)
    result = model.predict(decodedData)

    st.write(df)

    if result == 1:
        st.warning("Churn")
    if result == 0:
        st.success("Stay")



if __name__ == "__main__":
    main()
