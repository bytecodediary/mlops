from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# load model
model = joblib.load('model.pkl')

# create flask app
app = Flask(__name__)

# define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # get data from request
    data = request.get_json(force=True)

    features = np.array([data['features']]).reshape(1, -1)
    
    # convert data to dataframe
    # df = pd.DataFrame(data, index=[0])
    
    # make prediction
    prediction = model.predict(features)[0]
    
    # return prediction as json
    # output = {'prediction': int(prediction[0])}
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

#  Start your application by running → docker compose up --build        
#  Your application will be available at http://localhost:8000
