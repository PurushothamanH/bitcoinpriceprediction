import pickle
from flask import Flask,request,jsonify,app,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
# load the model
randomforest = pickle.load(open('random.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict_api",methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = randomforest.predict(new_data)
    ## It provides output as 2 d array so we want to take 1 value use[0]
    print(output[0])
    return jsonify(output[0])


if __name__=="__main__":
    app.run(debug=True) 