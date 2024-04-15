import pandas as pd
import streamlit as st 
# import numpy as np

from sqlalchemy import create_engine
import pickle, joblib


model1 = pickle.load(open('DT.pkl', 'rb'))
impute = joblib.load('num')
winsor = joblib.load('winsor')
minmax = joblib.load('scale')
encoding = joblib.load('encoding')


def predict_MPG(data, user, pw, db):

    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")


    clean = pd.DataFrame(impute.transform(data), columns=data.select_dtypes(exclude = ['object']).columns)
    clean1 = winsor.transform(clean)
    clean2 = pd.DataFrame(minmax.transform(clean1))
    clean3 = pd.DataFrame(encoding.transform(data))
    clean_data = pd.concat([clean2, clean3], axis = 1, ignore_index = True)
    prediction = pd.DataFrame(model1.predict(clean_data), columns = ['Predict_Drug'])
    
    final = pd.concat([prediction,data], axis = 1)
    final.to_sql('Drug', con = engine, if_exists = 'append', chunksize = 1000, index = False)
    
    return final

def main():
    

    st.title("Drug prediction")
    st.sidebar.title("Drug prediction")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Patient Drug Prediction App </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv','xlsx'], accept_multiple_files=False, key="fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame()
        
        
    else:
        st.sidebar.warning("You need to upload a CSV or an Excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Predict"):
        result = predict_MPG(data, user, pw, db)
        #st.dataframe(result) or
        st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap=cm))#.set_precision(2))

if __name__=='__main__':
    main()

