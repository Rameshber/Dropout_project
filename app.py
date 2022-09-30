# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:47:15 2022

@author: Anush Goswami
"""
import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
pickle_in = open("decisions.pkl","rb")
model=pickle.load(pickle_in)
pickle_in = open("knears.pkl","rb")
model=pickle.load(pickle_in)
pickle_in = open("randoms.pkl","rb")
model=pickle.load(pickle_in)
pickle_in = open("gaussians.pkl","rb")
model=pickle.load(pickle_in)
pickle_in = open("rbfs.pkl","rb")
model=pickle.load(pickle_in)
pickle_in = open("linears.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('Dropout_Academic Success.csv')
y = dataset.iloc[:, 36].values
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
def predict_note_authentication(Gender,Age_at_enrollment,Curricular_units1st_sem_approved,Curricular_units2nd_sem_approved):
  output= model.predict(([[Gender,Age_at_enrollment,Curricular_units1st_sem_approved,Curricular_units2nd_sem_approved]]))
  print("Target", output)
  if output==[0]:
    prediction="Dropout"
  elif output==[2]:
    prediction="Graduate"
  else:
    prediction="Enrolled"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;">Internship Project Deployment</p></center> 
   </div>
   </div>
   </div>
   """

    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Dropout_Academic Success")
    Gender = st.selectbox('Gender',('Male', 'Female'))
    if Gender == 'Male':
        Gender = 1
    else:
        Gender = 0
    Age_at_enrollment = st.number_input("Insert Age",18,60)
    Curricular_units1st_sem_approved = st.number_input("Curricular_units1st_sem_approved",30, 100)
    Curricular_units2nd_sem_approved = st.number_input("Curricular_units2nd_sem_approved",30, 100)
    resul=""
    if st.button("Linear Regression"):
      result=predict_note_authentication(Gender,Age_at_enrollment,Curricular_units1st_sem_approved,Curricular_units2nd_sem_approved)
      st.success('Model has predicted {}'.format(result))
    if st.button("Decision Tree"):
      result=predict_note_authentication(Gender,Age_at_enrollment,Curricular_units1st_sem_approved,Curricular_units2nd_sem_approved)
      st.success('Model has predicted {}'.format(result))
    if st.button("K-nearest Neighbor"):
      result=predict_note_authentication(Gender,Age_at_enrollment,Curricular_units1st_sem_approved,Curricular_units2nd_sem_approved)
      st.success('Model has predicted {}'.format(result))
    if st.button("Random Forest"):
      result=predict_note_authentication(Gender,Age_at_enrollment,Curricular_units1st_sem_approved,Curricular_units2nd_sem_approved)
      st.success('Model has predicted {}'.format(result))
    if st.button("Support Vector Machine"):
      result=predict_note_authentication(Gender,Age_at_enrollment,Curricular_units1st_sem_approved,Curricular_units2nd_sem_approved)
      st.success('Model has predicted {}'.format(result))
    if st.button("Naive Bayes"):
      result=predict_note_authentication(Gender,Age_at_enrollment,Curricular_units1st_sem_approved,Curricular_units2nd_sem_approved)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by Rameshber Goswami")
      st.subheader("Student , Department of Computer Engineering")

if __name__=='__main__':
  main()
   
