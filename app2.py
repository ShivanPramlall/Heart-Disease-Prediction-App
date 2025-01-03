import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv(r"https://raw.githubusercontent.com/ShivanPramlall/ITDAA-Project/main/heart%20(1).csv",sep=";")

# creates Random Forest Classification model
def create_model(dataset):

    # Remove outliers
    for i in [i for i in dataset.columns]:
        if dataset[i].nunique()>=12:
            Q1 = dataset[i].quantile(0.25)
            Q3 = dataset[i].quantile(0.75)
            IQR = Q3 - Q1
            dataset = dataset[dataset[i] <= (Q3+(1.5*IQR))]
            dataset = dataset[dataset[i] >= (Q1-(1.5*IQR))]

    X = dataset.iloc[:,:-1] # Using all column except for the last column as X
    y = dataset.iloc[:,-1] # Selecting the last column as Y

    # train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

    # Train the Logistic Regression model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

# create model
def predict(model,input_features):
    # Make predictions
    prediction = model.predict(input_features)

# calculating prediction prediction_probablity of model
def prediction_probablity(model,input_features):
    st.subheader('Prediction Probability')
    st.write('**0 : Patient is healthy \t 1 : Patient has heart disease**')
    prediction_probability = model.predict_proba(input_features)
    st.write(prediction_probability)

# Set the page configuration
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="🫀",
    initial_sidebar_state="expanded"
)

st.write("""
# Heart Disease Prediction App 
# """)
st.sidebar.header('Patient Input Parameters')

# dictionaries for categorical variables
sex_mapping = {"Male" : 0 , "Female" : 1}
cp_mapping = {"Typical angina": 0, 'Atypical angina': 1, 'Non-anginal pain': 2, 'Asymptomatic': 3}
fbs_mapping = {"False" : 0 , "True" : 1}
restecg_mapping = {'Normal': 0, 'Abnormal': 1, 'Ventricular hypertrophy': 2}
exang_mapping = {"No" : 0 , "Yes" : 1}
slope_mapping = {"Upsloping" : 0, "Flat" : 1, "Downsloping" : 2}
thal_mapping = {'Unknown': 0, 'Normal': 1, 'Fixed defect': 2, 'Reversible defect': 3}  

form = st.form(key="Information")

with form:

    st.write('**Information selected**:')

    age = st.sidebar.number_input('**Age**', 0)

    selected_sex_name = st.sidebar.radio('**Sex**', ['Male', 'Female'], index=0, format_func=lambda x: x ,horizontal=True)
    selected_sex_value = sex_mapping[selected_sex_name]
        
    selected_cp = st.sidebar.radio('**Chest Pain Type**', ['Typical angina', 'Atypical angina','Non-anginal pain', 'Asymptomatic'], index=0, format_func=lambda x: x )
    selected_cp_value = cp_mapping[selected_cp]

    selected_trestbps = st.sidebar.number_input('**Resting bood pressure (mm Hg)**',0)

    chol= st.sidebar.number_input('**Cholestrol (mg/dl)**',  0)

    selected_fbs= st.sidebar.radio('**Fasting blood sugar > 120 mg/dl**', ['True', 'False'], index=0, format_func=lambda x: x )
    selected_fbs_value = fbs_mapping[selected_fbs]

    selected_restecg = st.sidebar.radio('**Resting electrocardiographic results**', ['Normal', 'Abnormal','Ventricular hypertrophy'], index=0, format_func=lambda x: x)
    selected_restecg_value  = restecg_mapping[selected_restecg]

    thalach= st.sidebar.number_input('**Max heart rate achieved**',  0)

    selected_exang= st.sidebar.radio('**Exercise included angina**', ['Yes', 'No'], index=0, format_func=lambda x: x ,horizontal=True)
    selected_exang_value = exang_mapping[selected_exang]

    oldpeak= st.sidebar.number_input('**ST depression included by exercise relative to rest**', 0)

    selected_slope= st.sidebar.radio('**Slope of the peak exercise ST segment**', ['Upsloping', 'Flat', 'Downsloping'], index=0, format_func=lambda x: x)
    selected_slope_value = slope_mapping[selected_slope]

    ca= st.sidebar.slider('**Number of major vessls coloured by fluoroscopy**', 0,4)

    selected_thal= st.sidebar.radio('**Status of the heart**', ['Unknown', 'Normal', 'Fixed defect','Reversible defect'], index=0, format_func=lambda x: x)
    selected_thal_values = thal_mapping[selected_thal]

    # features displyed in a dataframe
    data = {
        "Age" : age,
        'Sex' : selected_sex_value,
        'Chest Pain Type' : selected_cp_value,
        'Resting blood pressure (mm Hg)' : selected_trestbps,
        'Cholestoral (mg/dl)' : chol,
        'Fasting blood sugar > 120 mg/dl':selected_fbs_value,
        'Resting electrocardiographic results' : selected_restecg_value,
        'Max heart rate achieved' : thalach,
        'Exercise included angina': selected_exang_value,
        'ST depression included by exercise relative to rest' : oldpeak,
        'Slope of the peak exercise ST segment' : selected_slope_value,
        'Number of major vessls coloured by fluoroscopy' : ca,
        'Status of the heart' : selected_thal_values
    }
    features = pd.DataFrame(data, index=[0])
    st.write(features)

    # fetaures that will be passed into model with same headings as in csv
    df = {
        "age" : age,
        'sex' : selected_sex_value,
        'cp' : selected_cp_value,
        'trestbps' : selected_trestbps,
        'chol' : chol,
        'fbs':selected_fbs_value,
        'restecg' : selected_restecg_value,
        'thalach' : thalach,
        'exang': selected_exang_value,
        'oldpeak' : oldpeak,
        'slope' : selected_slope_value,
        'ca' : ca,
        'thal' : selected_thal_values
    }
    input_features = pd.DataFrame(df, index=[0])

    model = create_model(dataset)

    # "Submit" button that will submit all info selected to the model
    submitted = form.form_submit_button("Submit")

    if submitted:

        predict(model,input_features)

        # create_model(dataset)    
        prediction_probablity(model,input_features)
    

    







    

