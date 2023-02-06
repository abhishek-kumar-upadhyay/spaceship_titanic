import pickle
from flask import Flask, request, app, jsonify, url_for, render_template

import numpy as np
import pandas as pd

app = Flask(__name__)

## Load Model
rf_model = pickle.load(open('models/rf_model.pkl', 'rb'))


@app.route('/predict_titanic_api', methods = ['POST'])

def predict_titanic_api():
    data = request.json['data']
    #print(data)
    #print("new_data", np.array(list(data.values())).reshape(1,-1))
    new_data = np.array(list(data.values())).reshape(1,-1)
    output = rf_model.predict(new_data)
    print("output is:",float(output[0]))
    return jsonify(float(output[0]))

if __name__ == "__main__":
    app.run(debug = True)