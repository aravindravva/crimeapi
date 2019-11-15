from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import pandas as pd
import json


app = Flask(__name__)


@app.route('/api', methods=['POST'])
def makecalc():
    data = request.get_json(force=True)
    d={}
    d['sex_ratio']=[data['sex_ratio']]
    d['literacy_rate']=[data['literacy_rate']]
    d['population']=[data['population']/1000]
    modelfile = 'models/final_prediction.pickle'
    model = p.load(open(modelfile, 'rb'))
    prediction=np.array2string(model.predict(pd.DataFrame(d)))    
    return jsonify(prediction)

if __name__ == '__main__':
    app.run()
    
