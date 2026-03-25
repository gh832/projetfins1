from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== HISTORIQUE =====
HISTORY_FILE = os.path.join(BASE_DIR, 'predictions.csv')

if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=[
        "age_years","BMI","ap_hi","ap_lo","cholesterol",
        "gluc","smoke","alco","active","gender",
        "probability","risk_level"
    ]).to_csv(HISTORY_FILE, index=False)


def load_model_and_scaler():
    model_path = os.path.join(BASE_DIR, '..', 'model.pkl')
    scaler_path = os.path.join(BASE_DIR, '..', 'scaler.pkl')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except:
        scaler = None

    return model, scaler


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/predictor')
def predictor():
    return render_template('index.html')


@app.route('/info')
def info():
    return render_template('info.html')


@app.route('/predict', methods=['POST'])
def predict():
    model, scaler = load_model_and_scaler()

    try:
        age_years = float(request.form['age_years'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        ap_hi = float(request.form['ap_hi'])
        ap_lo = float(request.form['ap_lo'])
        cholesterol = int(request.form['cholesterol'])
        gluc = int(request.form['gluc'])
        smoke = int(request.form['smoke'])
        alco = int(request.form['alco'])
        active = int(request.form['active'])
        gender = int(request.form['gender'])

        bmi = weight / ((height / 100) ** 2)

        data = {
            'age_years': age_years,
            'BMI': bmi,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'smoke': smoke,
            'alco': alco,
            'active': active,
            'gender': gender
        }

        df_input = pd.DataFrame([data])
        df_scaled = scaler.transform(df_input) if scaler else df_input
        proba = model.predict_proba(df_scaled)[0][1]

        result = {
            'prediction': 'Risque élevé de maladie cardio' if proba >= 0.70 else
                          'Risque modéré de maladie cardio' if proba >= 0.30 else
                          'Risque faible de maladie cardio',
            'probability': round(proba * 100, 2),
            'risk_level': 'Élevé' if proba >= 0.70 else 
                          'Modéré' if proba >= 0.30 else 
                          'Faible',
            'features': data
        }

        # ===== SAUVEGARDE HISTORIQUE =====
        history = pd.read_csv(HISTORY_FILE)
        data["probability"] = round(proba * 100, 2)
        data["risk_level"] = result["risk_level"]
        history = pd.concat([history, pd.DataFrame([data])], ignore_index=True)
        history.to_csv(HISTORY_FILE, index=False)

        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('error.html', error=str(e))


@app.route('/history')
def history():
    df = pd.read_csv(HISTORY_FILE)
    return render_template('history.html', tables=[df.to_html(classes='table table-bordered')])


if __name__ == '__main__':
    app.run(debug=True)
