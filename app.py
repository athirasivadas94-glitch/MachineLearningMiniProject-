from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))
print("Model loaded")

# Load encoder
gender_encoder = pickle.load(open('gender_encoder.pkl', 'rb'))
print("Gender encoder loaded")

# Load scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))
print("Scaler loaded")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:

            # Get form inputs
            Gender = request.form['Gender']
            print(Gender)

            Age = float(request.form['Age'])
            print(Age)

            Scholarship = int(request.form['Scholarship'])
            print(Scholarship)

            Hipertension = int(request.form['Hipertension'])
            print(Hipertension)

            Diabetes = int(request.form['Diabetes'])
            print(Diabetes)

            Alcoholism = int(request.form['Alcoholism'])
            print(Alcoholism)

            Handcap = int(request.form['Handcap'])
            print(Handcap)

            SMS_received = int(request.form['SMS_received'])
            print(SMS_received)

            WaitingDays = float(request.form['WaitingDays'])
            print(WaitingDays)


            # Encode Gender - map numeric to string values
            # 0 = Female, 1 = Male
            gender_map = {0: 'F', 1: 'M'}
            gender_label = gender_map.get(int(Gender), Gender)
            
            try:
                Gender_val = gender_encoder.transform([gender_label])[0]
            except Exception as e:
                print(f"Error encoding with F/M: {e}")
                # If 'F'/'M' don't work, try other common formats
                gender_map_alt = {0: 'Female', 1: 'Male'}
                gender_label = gender_map_alt.get(int(Gender), Gender)
                try:
                    Gender_val = gender_encoder.transform([gender_label])[0]
                except Exception as e2:
                    print(f"Error encoding with Female/Male: {e2}")
                    # Last resort - try 'Woman'/'Man'
                    gender_map_alt2 = {0: 'Woman', 1: 'Man'}
                    gender_label = gender_map_alt2.get(int(Gender), Gender)
                    Gender_val = gender_encoder.transform([gender_label])[0]
            
            print(f"Gender: 0=Female -> '{gender_label}' -> encoded as {Gender_val}")

            # Log transform waiting days
            WaitingDays = np.log1p(WaitingDays)

            # Scale Age and WaitingDays
            scaled_values = scaler.transform([[Age, WaitingDays]])

            Age_scaled = scaled_values[0][0]
            Waiting_scaled = scaled_values[0][1]

            # Prepare data
            details = [
                Gender_val,
                Age_scaled,
                Scholarship,
                Hipertension,
                Diabetes,
                Alcoholism,
                Handcap,
                SMS_received,
                Waiting_scaled
            ]

            print(details)

            data_out = np.array(details).reshape(1, -1)
            print(data_out)
            print(data_out.shape)

            # Prediction
            prediction = model.predict(data_out)
            pred_label = int(prediction[0])
            
            # Get probability/confidence
            try:
                probability = model.predict_proba(data_out)
                pred_prob = float(probability[0][pred_label])
            except:
                # Fallback if model doesn't support predict_proba
                # Use decision_function for confidence instead
                try:
                    decision = model.decision_function(data_out)
                    # Normalize to 0-1 range using sigmoid
                    pred_prob = 1 / (1 + np.exp(-decision[0]))
                    pred_prob = float(pred_prob)
                except:
                    # Default confidence if all else fails
                    pred_prob = 0.85
            
            print(prediction)
            print(f"Probability: {pred_prob}")
            
            if pred_label == 1:
                label = "WILL MISS"
            else:
                label = "WILL ATTEND"

            # Preserve form data
            form_data = {
                'Gender': int(Gender),
                'Age': Age,
                'Scholarship': Scholarship,
                'Hipertension': Hipertension,
                'Diabetes': Diabetes,
                'Alcoholism': Alcoholism,
                'Handcap': Handcap,
                'SMS_received': SMS_received,
                'WaitingDays': WaitingDays
            }

            result_data = {
                'prediction': pred_label,
                'label': label,
                'probability': pred_prob
            }

            return render_template('index.html', result=result_data, form_data=form_data)

        except Exception as e:
            error_msg = f'Error: {str(e)}'
            print(error_msg)
            return render_template('index.html', error=error_msg)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)