import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import streamlit as st
import numpy as np # linear algebra
from sklearn.neural_network import MLPClassifier # neural network
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from pickle import dump
from sklearn.preprocessing import StandardScaler
import librosa
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle
st.title('Project- liver disease prediction using machine learning')


info = pd.read_csv('datalabel.csv')
info.head()

st.title("Indormation / Data sets information")
st.write(info.columns)

X = info.drop(['Selector'], axis=1)  # Ganti 'target_column' dengan nama kolom target
y = info['Selector']
# split data into train and test sets
X_train,X_test,y_train, y_test= train_test_split(X, y, random_state=1, test_size=0.3)

st.title("Making predections...")

Gender = st.text_input("Entre your Sex (male = 1 and female = 2)")  # 1
age = st.text_input("Entre your age   ") # 2 
TB_total_bilirubin = st.text_input("Entre your Total_Bilirubin >  ") # 3
DB_Direct_Bilirubin = st.text_input("Entre your Direct_Bilirubin >  ")# 4
Alkphos_Alkaline_Phosphotase = st.text_input("Entre your Alkaline_Phosphotase >  ") # 5
Sgpt_Alamine_Aminotransferase = st.text_input("Entre your Alamine_Aminotransferase >  ") # 6
Sgot_Aspartate_Aminotransferase = st.text_input("Entre your Aspartate_Aminotransferase >  ") # 7
TP_Total_Protiens = st.text_input("Entre your Total_Protiens >  ")# 8
ALB_Albumin = st.text_input("Entre your Albumin >  ") # 9
AG_Ratio = st.text_input("Entre your Albumin_and_Globulin_Ratio >  ") # 10 

if st.button('Submit'):
    results = [[Gender, age, TB_total_bilirubin, DB_Direct_Bilirubin, Alkphos_Alkaline_Phosphotase, Sgpt_Alamine_Aminotransferase, Sgot_Aspartate_Aminotransferase,
                TP_Total_Protiens, ALB_Albumin,AG_Ratio ]]
    # Membuat DataFrame dari hasil normalisasi
    result = pd.DataFrame(results, columns=['Gender','Age','TB_total_bilirubin', 'DB_Direct_Bilirubin',
       'Alkphos_Alkaline_Phosphotase', 'Sgpt_Alamine_Aminotransferase',
       'Sgot_Aspartate_Aminotransferase', 'TP_Total_Protiens', 'ALB_Albumin',
       'A/G_Ratio',])  # Sesuaikan dengan nama kolom yang sesuai

# Menyimpan DataFrame ke dalam file CSV
    result.to_csv('hasil_zcr_rms1.csv', index=False)



    with open('scaler (1).pkl', 'rb') as scaler_file:
         scaler = pickle.load(scaler_file)
            
     # Lakukan standarisasi pada kolom yang telah ditentukan
    data_ternormalisasi = []
    for data in results:
        data_ternormalisasi.append(scaler.transform([data])[0])
        
     # Reduksi dimensi menggunakan PCA
    # Load the PCA model from the pickle file
    with open('model_pcal.pkl', 'rb') as file:
         sklearn_pca = pickle.load(file)
    X_test_pca = sklearn_pca.transform(data_ternormalisasi)

    # Use the loaded PCA model for dimensionality reduction


    with open('knn.pkl', 'rb') as knn_file:
        knn = pickle.load(knn_file)
    y_prediksi = knn.predict(X_test_pca)

    st.title('Check for Results')
    st.title('Predection')
    for i, final in enumerate(y_prediksi):
        if final == 1:
            st.write(f'Individu ke-{i+1}: Anda memiliki Penyakit Hati')
        else:
            st.write(f'Individu ke-{i+1}: Anda tidak memiliki Penyakit Hati')
