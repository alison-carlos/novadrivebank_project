from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pandas as pd
import joblib
from utils import *

# Instance of flask
app = Flask(__name__)

# Load model
model = load_model('my_model.keras')

# Load selector
selector = joblib.load('./objects/selector.joblib')

# Set a flask router
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    df = pd.DataFrame(input_data)
    df = fn_load_scalers(df, ['tempoprofissao', 'renda', 'idade', 'dependentes', 'valorsolicitado', 'valortotalbem', 'proporcaosolicitadototal'])
    df = fn_load_encoders(df, ['profissao', 'tiporesidencia', 'escolaridade', 'score', 'estadocivil', 'produto'])
    df = selector.transform(df)
    predictions = model.predict(df)
    return jsonify(predictions.tolist())


if __name__ == '__main__':
    app.run(debug=True)