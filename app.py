import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json


app = Flask(__name__)
## Load the regression model
regressionModel = pickle.load(open('regressionModel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))
## first route
@app.route('/')
def home():
    return render_template('home.html')

## Creating predict API (using postman)
@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1)) ## the data from JSON will be in key pairs    
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = regressionModel.predict(new_data)
    print(output)
    return jsonify(output[0]) ## the output will be in 2d array, so we are printing index 1

if __name__ == "__main__":
    app.run(debug=True)

## now the scalling.pkl file will standardize the data, then we can do the prediction