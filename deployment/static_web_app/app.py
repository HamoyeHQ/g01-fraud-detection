#import libraries
from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import numpy as np
import gzip
import dill

#Initialize the flask App
app = Flask(__name__)

# loading in the data
data = np.load('dataset.npy')

# loading in the data preparer
with gzip.open('scaler.gz.dill', 'rb') as f:
    scaler = dill.load(f)
        

# loading in the model
with gzip.open('calibration.gz.dill', 'rb') as f:
    model = dill.load(f)

cols = [i for i in range(1, 29) if i!=25]

#default page of our web-app
@app.route('/')
def home():
    return render_template("home.html")

#On clicking the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    input_ID = request.form['transaction_ID'] # gets user's input
    
    # checks if user's input is a number
    try:
        input_ID = int(input_ID) # makes sure the input ID is an integer
    
    except ValueError: # if input is not a number (not convertible to an integer)
        return render_template('home.html', pred='Your input ID is not valid, please enter a valid ID')
    
    if input_ID >= 0: # if ID is a positive number
        
        if input_ID in range(data.shape[0]): # checks if ID is in our dataset
            X = data[input_ID]
            
        else: # randomly generate a 1D input array from our dataset such that for each column in our dataset, a value is randomly chosen
            X = np.array([np.random.choice(data[:, i]) for i in range(data.shape[1])])
        
        Xt = X[cols].reshape(1, -1) # selects the important features
        Xtt = scaler.transform(Xt) # transforms input
        prediction = model.predict(Xtt)[0] # gets prediction

        if prediction == 1:
            value = "Fraudulent"
            proba = model.predict_proba(Xtt)[0][1] # gets probability of being positive
            
        else:
            value = "Not Fraudulent"
            proba = model.predict_proba(Xtt)[0][0] # gets probability of being negative

        return render_template('home.html', pred='This Transaction is likely {} with a probability of {}'.format(value, proba))

    else: # negative ID number not allowed!
        return render_template('home.html', pred='Your input ID is not valid, please enter a valid ID')

if __name__ == '__main__':
    app.run(debug=True)
