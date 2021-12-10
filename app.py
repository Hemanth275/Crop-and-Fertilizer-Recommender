from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder

'''df=pd.read_csv("Crop_recommendation.csv")
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)
std_scalar = StandardScaler()
X_train = std_scalar.fit_transform(X_train)
X_test = std_scalar.transform(X_test)'''
#############################################

app=Flask(__name__,template_folder='HTML')

@ app.route('/')
def home():
    
    return render_template('front.html')

@ app.route('/crop_recommend')
def crop_recommend():
    return render_template('crop.html')

@ app.route('/fertilizer_recommendation')
def fertilizer_recommendation():
    return render_template('fertilizer.html')


fertilizer_recommendation_model_path = 'fertilizer_recommender_model.pkl'
fertilizer_recommendation_model = pickle.load(
    open(fertilizer_recommendation_model_path, 'rb'))

##################################
df=pd.read_csv("Fertilizer_Prediction.csv")
soil_type_label_encoder = LabelEncoder()
df["Soil Type"] = soil_type_label_encoder.fit_transform(df["Soil Type"])
crop_type_label_encoder = LabelEncoder()
df["Crop Type"] = crop_type_label_encoder.fit_transform(df["Crop Type"])
X1 = df[df.columns[:-1]]
y1 = df[df.columns[-1]]
upsample = SMOTE()
X1, y1 = upsample.fit_resample(X1, y1)
counter = Counter(y1)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size = 0.8, random_state = 0)
std_scalar = StandardScaler()
X1_train = std_scalar.fit_transform(X1_train)
X1_test = std_scalar.transform(X1_test)
@ app.route('/fertilizer_predict', methods=['POST'])
def fertilizer_prediction():
    if request.method == 'POST':
        N = int(request.form['Nitrogen'])
        P = int(request.form['Phosphorous'])
        K = int(request.form['Potassium'])
        S =int(request.form['Soil Type'])
        C = int(request.form['Crop Type'])
        T= float(request.form['Temperature'])
        H= float(request.form['Humidity'])
        M = float(request.form['Moisture'])
        data = np.array((([[T,H,M,S,C,N,K,P]])))
        my_prediction = fertilizer_recommendation_model.predict(std_scalar.transform(data))
        final_prediction = my_prediction[0]
        
        return render_template('fertilizer_result.html', prediction=final_prediction)
    
######################
crop_recommendation_model_path = 'RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

@ app.route('/crop_predict', methods=['POST'])
def crop_prediction():

    if request.method == 'POST':
        N = int(request.form['Nitrogen'])
        P = int(request.form['Phosphorous'])
        K = int(request.form['Potassium'])
        T= float(request.form['Temperature'])
        H= float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])
        data = np.array((([[N, P, K, T, H, ph, rainfall]])))
        
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        return render_template('crop_result.html', prediction=final_prediction)
    

    
    
    
    
    
    
if __name__ =="__main__":
    app.run(debug=True)


