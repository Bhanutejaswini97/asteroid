from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import requests
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

model = joblib.load('GB_Model.pkl')

@app.route('/prediction',methods=['POST'])
def prediction():
    data = request.json
    ad = float(data['ad'])
    data_arc= float(data['data_arc'])
    e= float(data['e'])
    H= float(data['H'])
    q= float(data['q'])
    n= float(data['n'])
    neo= float(data['neo'])
    per= float(data['per'])
    per_y= float(data['per_y'])
    dat = {'ad':[ad],'data_arc':[data_arc],'e':[e],'H':[H],'q':[q],'n':[n],'neo':[neo],'per':[per],'per_y':[per_y]}
    df = pd.DataFrame(dat)
    c_i = df.values.reshape(1, -1)
    pha = (model.predict(c_i))
    if(pha == 1):
        response = "Asteroid with the given properties is hazardous"
    else:
        response = "Asteroid with the given properties is not hazardous"
    
    output = {
        "pha":response
    }
    
    
    return jsonify(output)
    

if __name__ == "__main__":
    app.run(debug=True)