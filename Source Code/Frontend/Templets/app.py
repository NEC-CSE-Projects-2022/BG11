# # # # # # # # # # # # # # # # # # # from flask import Flask, request, jsonify, render_template
# # # # # # # # # # # # # # # # # # # from flask_cors import CORS
# # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # import os
# # # # # # # # # # # # # # # # # # # import traceback

# # # # # # # # # # # # # # # # # # # try:
# # # # # # # # # # # # # # # # # # #     import joblib
# # # # # # # # # # # # # # # # # # # except ImportError:
# # # # # # # # # # # # # # # # # # #     joblib = None

# # # # # # # # # # # # # # # # # # # try:
# # # # # # # # # # # # # # # # # # #     from tensorflow.keras.models import load_model as keras_load_model
# # # # # # # # # # # # # # # # # # # except ImportError:
# # # # # # # # # # # # # # # # # # #     keras_load_model = None

# # # # # # # # # # # # # # # # # # # app = Flask(__name__, template_folder='templates', static_folder='static')
# # # # # # # # # # # # # # # # # # # CORS(app)

# # # # # # # # # # # # # # # # # # # model = None
# # # # # # # # # # # # # # # # # # # model_type = None
# # # # # # # # # # # # # # # # # # # history = []

# # # # # # # # # # # # # # # # # # # # MODEL_PATHS = ["model.pkl", "model.joblib", "model.h5", "dnn1.keras"]
# # # # # # # # # # # # # # # # # # # MODEL_PATHS = ["model.pkl", "model.joblib", "model.h5", "dnn1.keras"]


# # # # # # # # # # # # # # # # # # # def load_model_file():
# # # # # # # # # # # # # # # # # # #     global model, model_type
# # # # # # # # # # # # # # # # # # #     for path in MODEL_PATHS:
# # # # # # # # # # # # # # # # # # #         if os.path.exists(path):
# # # # # # # # # # # # # # # # # # #             try:
# # # # # # # # # # # # # # # # # # #                 if (path.endswith(".h5") or path.endswith(".keras")) and keras_load_model:
# # # # # # # # # # # # # # # # # # #                     model = keras_load_model(path)
# # # # # # # # # # # # # # # # # # #                     model_type = "keras"
# # # # # # # # # # # # # # # # # # #                 elif joblib:
# # # # # # # # # # # # # # # # # # #                     model = joblib.load(path)
# # # # # # # # # # # # # # # # # # #                     model_type = "sklearn"
# # # # # # # # # # # # # # # # # # #                 print(f"✅ Model loaded from {path}")
# # # # # # # # # # # # # # # # # # #                 return
# # # # # # # # # # # # # # # # # # #             except Exception as e:
# # # # # # # # # # # # # # # # # # #                 print(f"❌ Failed to load model from {path}: {e}")
# # # # # # # # # # # # # # # # # # #     print("⚠ No model found. Please place your model file in the project root.")

# # # # # # # # # # # # # # # # # # # @app.route("/")
# # # # # # # # # # # # # # # # # # # def home():
# # # # # # # # # # # # # # # # # # #     return render_template("index.html")

# # # # # # # # # # # # # # # # # # # @app.route("/predict", methods=["POST"])
# # # # # # # # # # # # # # # # # # # def predict():
# # # # # # # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # # # # # # #         data = request.get_json()
# # # # # # # # # # # # # # # # # # #         if not data or "features" not in data:
# # # # # # # # # # # # # # # # # # #             return jsonify({"error": "Missing 'features' in request"}), 400

# # # # # # # # # # # # # # # # # # #         features = np.array(data["features"], dtype=float).reshape(1, -1)

# # # # # # # # # # # # # # # # # # #         if model is None:
# # # # # # # # # # # # # # # # # # #             return jsonify({"error": "Model not loaded"}), 500

# # # # # # # # # # # # # # # # # # #         if model_type == "keras":
# # # # # # # # # # # # # # # # # # #             raw_pred = model.predict(features).tolist()
# # # # # # # # # # # # # # # # # # #             try:
# # # # # # # # # # # # # # # # # # #                 pred = int(np.argmax(raw_pred, axis=1)[0])
# # # # # # # # # # # # # # # # # # #             except:
# # # # # # # # # # # # # # # # # # #                 pred = raw_pred[0]
# # # # # # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # # # # # #             pred = model.predict(features).tolist()[0]
# # # # # # # # # # # # # # # # # # #             raw_pred = None

# # # # # # # # # # # # # # # # # # #         result = {"prediction": pred, "raw": raw_pred}
# # # # # # # # # # # # # # # # # # #         history.append({"features": data["features"], "result": result})

# # # # # # # # # # # # # # # # # # #         return jsonify(result)
# # # # # # # # # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # # # # # # # # #         traceback.print_exc()
# # # # # # # # # # # # # # # # # # #         return jsonify({"error": str(e)}), 500

# # # # # # # # # # # # # # # # # # # @app.route("/history", methods=["GET"])
# # # # # # # # # # # # # # # # # # # def get_history():
# # # # # # # # # # # # # # # # # # #     return jsonify(history[::-1])

# # # # # # # # # # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # # # # # # # # # #     load_model_file()
# # # # # # # # # # # # # # # # # # #     app.run(debug=True)
# # # # # # # # # # # # # # # # # # from flask import Flask, request, jsonify, render_template
# # # # # # # # # # # # # # # # # # from flask_cors import CORS
# # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # import os
# # # # # # # # # # # # # # # # # # import traceback
# # # # # # # # # # # # # # # # # # from tensorflow.keras.models import load_model

# # # # # # # # # # # # # # # # # # # Flask app setup
# # # # # # # # # # # # # # # # # # app = Flask(__name__, template_folder='templates', static_folder='static')
# # # # # # # # # # # # # # # # # # CORS(app)

# # # # # # # # # # # # # # # # # # # Global variables
# # # # # # # # # # # # # # # # # # model = None
# # # # # # # # # # # # # # # # # # history = []
# # # # # # # # # # # # # # # # # # MODEL_PATH = "dnn1.keras"  # Only using your DNN model

# # # # # # # # # # # # # # # # # # # Load the model
# # # # # # # # # # # # # # # # # # def load_model_file():
# # # # # # # # # # # # # # # # # #     global model
# # # # # # # # # # # # # # # # # #     if os.path.exists(MODEL_PATH):
# # # # # # # # # # # # # # # # # #         try:
# # # # # # # # # # # # # # # # # #             model = load_model(MODEL_PATH)
# # # # # # # # # # # # # # # # # #             print(f"✅ Model loaded from {MODEL_PATH}")
# # # # # # # # # # # # # # # # # #         except Exception as e:
# # # # # # # # # # # # # # # # # #             print(f"❌ Failed to load model: {e}")
# # # # # # # # # # # # # # # # # #     else:
# # # # # # # # # # # # # # # # # #         print("⚠ Model file not found. Please place dnn1.keras in the project root.")

# # # # # # # # # # # # # # # # # # @app.route("/")
# # # # # # # # # # # # # # # # # # def home():
# # # # # # # # # # # # # # # # # #     return render_template("index.html")

# # # # # # # # # # # # # # # # # # @app.route("/predict", methods=["POST"])
# # # # # # # # # # # # # # # # # # def predict():
# # # # # # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # # # # # #         data = request.get_json()
# # # # # # # # # # # # # # # # # #         if not data or "features" not in data:
# # # # # # # # # # # # # # # # # #             return jsonify({"error": "Missing 'features' in request"}), 400

# # # # # # # # # # # # # # # # # #         features = np.array(data["features"], dtype=float).reshape(1, -1)

# # # # # # # # # # # # # # # # # #         if model is None:
# # # # # # # # # # # # # # # # # #             return jsonify({"error": "Model not loaded"}), 500

# # # # # # # # # # # # # # # # # #         raw_pred = model.predict(features).tolist()
# # # # # # # # # # # # # # # # # #         try:
# # # # # # # # # # # # # # # # # #             pred = int(np.argmax(raw_pred, axis=1)[0])
# # # # # # # # # # # # # # # # # #         except:
# # # # # # # # # # # # # # # # # #             pred = raw_pred[0]

# # # # # # # # # # # # # # # # # #         result = {"prediction": pred, "raw": raw_pred}
# # # # # # # # # # # # # # # # # #         history.append({"features": data["features"], "result": result})

# # # # # # # # # # # # # # # # # #         return jsonify(result)
# # # # # # # # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # # # # # # # #         traceback.print_exc()
# # # # # # # # # # # # # # # # # #         return jsonify({"error": str(e)}), 500

# # # # # # # # # # # # # # # # # # @app.route("/history", methods=["GET"])
# # # # # # # # # # # # # # # # # # def get_history():
# # # # # # # # # # # # # # # # # #     return jsonify(history[::-1])

# # # # # # # # # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # # # # # # # # #     load_model_file()
# # # # # # # # # # # # # # # # # #     app.run(debug=True)

# # # # # # # # # # # # # # # # # # from flask import Flask, request, jsonify, render_template
# # # # # # # # # # # # # # # # # # import pandas as pd
# # # # # # # # # # # # # # # # # # import joblib
# # # # # # # # # # # # # # # # # # import tensorflow as tf
# # # # # # # # # # # # # # # # # # import numpy as np

# # # # # # # # # # # # # # # # # # app = Flask(__name__)

# # # # # # # # # # # # # # # # # # # Load model and encoder/scaler if you have them
# # # # # # # # # # # # # # # # # # model = tf.keras.models.load_model("dnn1.keras")
# # # # # # # # # # # # # # # # # # encoder = joblib.load("encoder.pkl")  # Label/OneHot encoder
# # # # # # # # # # # # # # # # # # scaler = joblib.load("scaler.pkl")    # StandardScaler

# # # # # # # # # # # # # # # # # # # Define all columns in the exact order used in training
# # # # # # # # # # # # # # # # # # feature_columns = [
# # # # # # # # # # # # # # # # # #     'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
# # # # # # # # # # # # # # # # # #     'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status',
# # # # # # # # # # # # # # # # # #     'extra_feature1', 'extra_feature2', 'extra_feature3', 'extra_feature4', 'extra_feature5'
# # # # # # # # # # # # # # # # # # ]

# # # # # # # # # # # # # # # # # # @app.route('/')
# # # # # # # # # # # # # # # # # # def home():
# # # # # # # # # # # # # # # # # #     return render_template("index.html")

# # # # # # # # # # # # # # # # # # @app.route('/predict', methods=['POST'])
# # # # # # # # # # # # # # # # # # def predict():
# # # # # # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # # # # # #         data = request.json
# # # # # # # # # # # # # # # # # #         df = pd.DataFrame([data])

# # # # # # # # # # # # # # # # # #         # Ensure all features exist
# # # # # # # # # # # # # # # # # #         for col in feature_columns:
# # # # # # # # # # # # # # # # # #             if col not in df.columns:
# # # # # # # # # # # # # # # # # #                 df[col] = 0  # Default value for missing columns

# # # # # # # # # # # # # # # # # #         # Apply preprocessing
# # # # # # # # # # # # # # # # # #         df[["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]] = encoder.transform(
# # # # # # # # # # # # # # # # # #             df[["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]]
# # # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # # #         df_scaled = scaler.transform(df)

# # # # # # # # # # # # # # # # # #         # Predict
# # # # # # # # # # # # # # # # # #         pred = model.predict(df_scaled)
# # # # # # # # # # # # # # # # # #         result = int(pred[0][0] > 0.5)

# # # # # # # # # # # # # # # # # #         return jsonify({"prediction": result, "probability": float(pred[0][0])})

# # # # # # # # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # # # # # # # #         return jsonify({"error": str(e)}), 500

# # # # # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # # # # #     app.run(debug=True)


# # # # # # # # # # # # # # # # # from flask import Flask, render_template, request, jsonify
# # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # import tensorflow as tf
# # # # # # # # # # # # # # # # # import joblib

# # # # # # # # # # # # # # # # # app = Flask(__name__)

# # # # # # # # # # # # # # # # # # Load the trained model
# # # # # # # # # # # # # # # # # model = tf.keras.models.load_model("dnn1.keras")

# # # # # # # # # # # # # # # # # # If you used a scaler/encoder during training, load it
# # # # # # # # # # # # # # # # # # scaler = joblib.load("scaler.pkl")
# # # # # # # # # # # # # # # # # # encoder = joblib.load("encoder.pkl")

# # # # # # # # # # # # # # # # # @app.route('/')
# # # # # # # # # # # # # # # # # def home():
# # # # # # # # # # # # # # # # #     return render_template('index.html')

# # # # # # # # # # # # # # # # # @app.route('/predict', methods=['POST'])
# # # # # # # # # # # # # # # # # def predict():
# # # # # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # # # # #         data = request.json
        
# # # # # # # # # # # # # # # # #         # Extract features from request
# # # # # # # # # # # # # # # # #         features = [
# # # # # # # # # # # # # # # # #             float(data['age']),
# # # # # # # # # # # # # # # # #             float(data['hypertension']),
# # # # # # # # # # # # # # # # #             float(data['heart_disease']),
# # # # # # # # # # # # # # # # #             float(data['avg_glucose_level']),
# # # # # # # # # # # # # # # # #             float(data['bmi']),
# # # # # # # # # # # # # # # # #             float(data['gender']),
# # # # # # # # # # # # # # # # #             float(data['ever_married']),
# # # # # # # # # # # # # # # # #             float(data['work_type']),
# # # # # # # # # # # # # # # # #             float(data['residence_type']),
# # # # # # # # # # # # # # # # #             float(data['smoking_status'])
# # # # # # # # # # # # # # # # #         ]
        
# # # # # # # # # # # # # # # # #         # Preprocess if needed
# # # # # # # # # # # # # # # # #         # features = scaler.transform([features])
        
# # # # # # # # # # # # # # # # #         # Predict
# # # # # # # # # # # # # # # # #         prediction = model.predict(np.array([features]))[0][0]
# # # # # # # # # # # # # # # # #         pred_label = 1 if prediction >= 0.5 else 0

# # # # # # # # # # # # # # # # #         return jsonify({
# # # # # # # # # # # # # # # # #             'prediction': int(pred_label),
# # # # # # # # # # # # # # # # #             'probability': round(float(prediction), 4)
# # # # # # # # # # # # # # # # #         })
# # # # # # # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # # # # # # #         return jsonify({'error': str(e)})

# # # # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # # # #     app.run(debug=True)


# # # # # # # # # # # # # # # # from flask import Flask, render_template, request, jsonify
# # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # import tensorflow as tf

# # # # # # # # # # # # # # # # app = Flask(__name__)

# # # # # # # # # # # # # # # # # Load the trained model
# # # # # # # # # # # # # # # # model = tf.keras.models.load_model("dnn1.keras")


# # # # # # # # # # # # # # # # def preprocess_input(data):
# # # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # # #     Convert raw form data into the same format the model was trained on.
# # # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # # #     # Base numeric fields
# # # # # # # # # # # # # # # #     features = [
# # # # # # # # # # # # # # # #         float(data['age']),
# # # # # # # # # # # # # # # #         float(data['hypertension']),
# # # # # # # # # # # # # # # #         float(data['heart_disease']),
# # # # # # # # # # # # # # # #         float(data['avg_glucose_level']),
# # # # # # # # # # # # # # # #         float(data['bmi'])
# # # # # # # # # # # # # # # #     ]
    
# # # # # # # # # # # # # # # #     # Gender (binary)
# # # # # # # # # # # # # # # #     features.append(float(data['gender']))
    
# # # # # # # # # # # # # # # #     # Ever Married (binary)
# # # # # # # # # # # # # # # #     features.append(float(data['ever_married']))
    
# # # # # # # # # # # # # # # #     # One-hot encode work type (5 categories)
# # # # # # # # # # # # # # # #     work_type_onehot = [0, 0, 0, 0, 0]
# # # # # # # # # # # # # # # #     work_type_onehot[int(data['work_type'])] = 1
# # # # # # # # # # # # # # # #     features.extend(work_type_onehot)
    
# # # # # # # # # # # # # # # #     # One-hot encode residence type (2 categories)
# # # # # # # # # # # # # # # #     residence_type_onehot = [0, 0]
# # # # # # # # # # # # # # # #     residence_type_onehot[int(data['residence_type'])] = 1
# # # # # # # # # # # # # # # #     features.extend(residence_type_onehot)
    
# # # # # # # # # # # # # # # #     # One-hot encode smoking status (4 categories)
# # # # # # # # # # # # # # # #     smoking_status_onehot = [0, 0, 0, 0]
# # # # # # # # # # # # # # # #     smoking_status_onehot[int(data['smoking_status'])] = 1
# # # # # # # # # # # # # # # #     features.extend(smoking_status_onehot)
    
# # # # # # # # # # # # # # # #     return features


# # # # # # # # # # # # # # # # @app.route('/')
# # # # # # # # # # # # # # # # def home():
# # # # # # # # # # # # # # # #     return render_template('index.html')


# # # # # # # # # # # # # # # # @app.route('/predict', methods=['POST'])
# # # # # # # # # # # # # # # # def predict():
# # # # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # # # #         data = request.json
        
# # # # # # # # # # # # # # # #         # Preprocess input
# # # # # # # # # # # # # # # #         features = preprocess_input(data)
        
# # # # # # # # # # # # # # # #         # Convert to numpy array
# # # # # # # # # # # # # # # #         features_array = np.array([features])
        
# # # # # # # # # # # # # # # #         # Predict
# # # # # # # # # # # # # # # #         prediction = model.predict(features_array)[0][0]
# # # # # # # # # # # # # # # #         pred_label = 1 if prediction >= 0.5 else 0

# # # # # # # # # # # # # # # #         return jsonify({
# # # # # # # # # # # # # # # #             'prediction': int(pred_label),
# # # # # # # # # # # # # # # #             'probability': round(float(prediction), 4)
# # # # # # # # # # # # # # # #         })
    
# # # # # # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # # # # # #         return jsonify({'error': str(e)})


# # # # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # # # #     app.run(debug=True)
# # # # # # # # # # # # # # # from flask import Flask, render_template, request, jsonify
# # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # import tensorflow as tf

# # # # # # # # # # # # # # # app = Flask(__name__)

# # # # # # # # # # # # # # # # Load your trained model (make sure dnn1.keras is in the project root)
# # # # # # # # # # # # # # # model = tf.keras.models.load_model("dnn1.keras")


# # # # # # # # # # # # # # # def preprocess_input(data):
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     Build the feature vector in the exact order used during training
# # # # # # # # # # # # # # #     (pd.get_dummies(..., drop_first=True) with categorical_cols =
# # # # # # # # # # # # # # #      ['gender','ever_married','work_type','Residence_type','smoking_status']).
# # # # # # # # # # # # # # #     RETURNS: Python list of length 15
# # # # # # # # # # # # # # #     Order:
# # # # # # # # # # # # # # #     [age, hypertension, heart_disease, avg_glucose_level, bmi,
# # # # # # # # # # # # # # #      gender_Male, gender_Other,
# # # # # # # # # # # # # # #      ever_married_Yes,
# # # # # # # # # # # # # # #      work_type_Govt_job, work_type_Never_worked, work_type_Private, work_type_Self-employed,
# # # # # # # # # # # # # # #      Residence_type_Urban,
# # # # # # # # # # # # # # #      smoking_status_never smoked, smoking_status_smokes]
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     features = []

# # # # # # # # # # # # # # #     # numeric base
# # # # # # # # # # # # # # #     features.append(float(data.get("age", 0)))
# # # # # # # # # # # # # # #     features.append(int(data.get("hypertension", 0)))
# # # # # # # # # # # # # # #     features.append(int(data.get("heart_disease", 0)))
# # # # # # # # # # # # # # #     features.append(float(data.get("avg_glucose_level", 0.0)))
# # # # # # # # # # # # # # #     # BMI may be missing; handle gracefully
# # # # # # # # # # # # # # #     bmi_val = data.get("bmi", None)
# # # # # # # # # # # # # # #     features.append(float(bmi_val) if bmi_val is not None and bmi_val != "" else 0.0)

# # # # # # # # # # # # # # #     # CATEGORICALS -> produce dummy flags matching drop_first=True behaviour
# # # # # # # # # # # # # # #     gender = data.get("gender", "Female")  # default Female (dropped)
# # # # # # # # # # # # # # #     # gender_Male, gender_Other (gender_Female is dropped)
# # # # # # # # # # # # # # #     features.append(1.0 if gender == "Male" else 0.0)        # gender_Male
# # # # # # # # # # # # # # #     features.append(1.0 if gender == "Other" else 0.0)       # gender_Other

# # # # # # # # # # # # # # #     ever_married = data.get("ever_married", "No")
# # # # # # # # # # # # # # #     # ever_married_Yes (No is dropped)
# # # # # # # # # # # # # # #     features.append(1.0 if ever_married == "Yes" else 0.0)

# # # # # # # # # # # # # # #     # Work type: expected full set (5 categories) -> keep these 4 dummies (drop one)
# # # # # # # # # # # # # # #     wt = data.get("work_type", "Private")
# # # # # # # # # # # # # # #     features.append(1.0 if wt == "Govt_job" else 0.0)       # work_type_Govt_job
# # # # # # # # # # # # # # #     features.append(1.0 if wt == "Never_worked" else 0.0)   # work_type_Never_worked
# # # # # # # # # # # # # # #     features.append(1.0 if wt == "Private" else 0.0)        # work_type_Private
# # # # # # # # # # # # # # #     features.append(1.0 if wt == "Self-employed" else 0.0)  # work_type_Self-employed

# # # # # # # # # # # # # # #     # Residence_type -> Residence_type_Urban (Rural dropped)
# # # # # # # # # # # # # # #     residence = data.get("Residence_type", "Rural")
# # # # # # # # # # # # # # #     features.append(1.0 if residence == "Urban" else 0.0)

# # # # # # # # # # # # # # #     # Smoking status: seen values (formerly smoked, never smoked, smokes)
# # # # # # # # # # # # # # #     # drop_first=True removed 'formerly smoked' -> keep two dummies:
# # # # # # # # # # # # # # #     # smoking_status_never smoked, smoking_status_smokes
# # # # # # # # # # # # # # #     smoke = data.get("smoking_status", "formerly smoked")
# # # # # # # # # # # # # # #     features.append(1.0 if smoke == "never smoked" else 0.0)
# # # # # # # # # # # # # # #     features.append(1.0 if smoke == "smokes" else 0.0)

# # # # # # # # # # # # # # #     # ensure length:
# # # # # # # # # # # # # # #     if len(features) != 15:
# # # # # # # # # # # # # # #         raise ValueError(f"Preprocessed feature length mismatch: expected 15 got {len(features)}")

# # # # # # # # # # # # # # #     return features


# # # # # # # # # # # # # # # @app.route("/")
# # # # # # # # # # # # # # # def home():
# # # # # # # # # # # # # # #     return render_template("index.html")


# # # # # # # # # # # # # # # @app.route("/predict", methods=["POST"])
# # # # # # # # # # # # # # # def predict():
# # # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # # #         data = request.json

# # # # # # # # # # # # # # #         # build feature vector
# # # # # # # # # # # # # # #         features = preprocess_input(data)

# # # # # # # # # # # # # # #         # NOTE: if you used an imputer/scaler during training, load and apply it here
# # # # # # # # # # # # # # #         # Example (if you saved a scaler): scaler = joblib.load("scaler.pkl"); features = scaler.transform([features])

# # # # # # # # # # # # # # #         arr = np.array([features], dtype=np.float32)  # shape (1,15)
# # # # # # # # # # # # # # #         pred_prob = float(model.predict(arr, verbose=0)[0][0])  # model outputs probability
# # # # # # # # # # # # # # #         pred_label = 1 if pred_prob >= 0.5 else 0

# # # # # # # # # # # # # # #         return jsonify({
# # # # # # # # # # # # # # #             "prediction": int(pred_label),
# # # # # # # # # # # # # # #             "probability": round(pred_prob, 4)
# # # # # # # # # # # # # # #         })
# # # # # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # # # # #         # return error message for debugging (in production, send cleaner message)
# # # # # # # # # # # # # # #         return jsonify({"error": str(e)}), 400


# # # # # # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # # # # # #     app.run(debug=True)





# # # # # # # # # # # # # # from flask import Flask, render_template, request, jsonify
# # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # import tensorflow as tf

# # # # # # # # # # # # # # app = Flask(__name__)

# # # # # # # # # # # # # # # Load the trained model
# # # # # # # # # # # # # # model = tf.keras.models.load_model("dnn1.keras")

# # # # # # # # # # # # # # def preprocess_input(data):
# # # # # # # # # # # # # #     # Match EXACT training column order (15 features)
# # # # # # # # # # # # # #     features = [
# # # # # # # # # # # # # #         float(data['age']),
# # # # # # # # # # # # # #         float(data['hypertension']),
# # # # # # # # # # # # # #         float(data['heart_disease']),
# # # # # # # # # # # # # #         float(data['avg_glucose_level']),
# # # # # # # # # # # # # #         float(data['bmi'])
# # # # # # # # # # # # # #     ]
    
# # # # # # # # # # # # # #     # gender_Male
# # # # # # # # # # # # # #     features.append(1.0 if data['gender'] == 'Male' else 0.0)
    
# # # # # # # # # # # # # #     # ever_married_Yes
# # # # # # # # # # # # # #     features.append(1.0 if data['ever_married'] == 'Yes' else 0.0)
    
# # # # # # # # # # # # # #     # work_type one-hot (Govt_job, Never_worked, Private, Self-employed)
# # # # # # # # # # # # # #     work_types = ['Govt_job', 'Never_worked', 'Private', 'Self-employed']
# # # # # # # # # # # # # #     for wt in work_types:
# # # # # # # # # # # # # #         features.append(1.0 if data['work_type'] == wt else 0.0)
    
# # # # # # # # # # # # # #     # Residence_type_Urban
# # # # # # # # # # # # # #     features.append(1.0 if data['Residence_type'] == 'Urban' else 0.0)
    
# # # # # # # # # # # # # #     # smoking_status one-hot (formerly smoked, never smoked, smokes)
# # # # # # # # # # # # # #     smoking_types = ['formerly smoked', 'never smoked', 'smokes']
# # # # # # # # # # # # # #     for st in smoking_types:
# # # # # # # # # # # # # #         features.append(1.0 if data['smoking_status'] == st else 0.0)
    
# # # # # # # # # # # # # #     return features

# # # # # # # # # # # # # # @app.route('/')
# # # # # # # # # # # # # # def home():
# # # # # # # # # # # # # #     return render_template('index.html')

# # # # # # # # # # # # # # @app.route('/predict', methods=['POST'])
# # # # # # # # # # # # # # def predict():
# # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # #         data = request.json
# # # # # # # # # # # # # #         features = preprocess_input(data)

# # # # # # # # # # # # # #         prediction = model.predict(np.array([features]))[0][0]
# # # # # # # # # # # # # #         pred_label = 1 if prediction >= 0.5 else 0

# # # # # # # # # # # # # #         return jsonify({
# # # # # # # # # # # # # #             'prediction': int(pred_label),
# # # # # # # # # # # # # #             'probability': round(float(prediction), 4)
# # # # # # # # # # # # # #         })
# # # # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # # # #         return jsonify({'error': str(e)})

# # # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # # #     app.run(debug=True)


# # # # # # # # # # # # # from flask import Flask, render_template, request, jsonify
# # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # import tensorflow as tf

# # # # # # # # # # # # # app = Flask(__name__)

# # # # # # # # # # # # # # Load trained model
# # # # # # # # # # # # # model = tf.keras.models.load_model("dnn1.keras")

# # # # # # # # # # # # # # Expected features in correct order (must match training)
# # # # # # # # # # # # # expected_features = [
# # # # # # # # # # # # #     'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
# # # # # # # # # # # # #     'gender_Male', 'ever_married_Yes',
# # # # # # # # # # # # #     'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
# # # # # # # # # # # # #     'work_type_Self-employed',
# # # # # # # # # # # # #     'Residence_type_Urban',
# # # # # # # # # # # # #     'smoking_status_formerly smoked', 'smoking_status_never smoked',
# # # # # # # # # # # # #     'smoking_status_smokes'
# # # # # # # # # # # # # ]

# # # # # # # # # # # # # # Min and Max values from training dataset (example values — replace with your real training stats)
# # # # # # # # # # # # # min_values = {
# # # # # # # # # # # # #     'age': 0,
# # # # # # # # # # # # #     'hypertension': 0,
# # # # # # # # # # # # #     'heart_disease': 0,
# # # # # # # # # # # # #     'avg_glucose_level': 55.12,
# # # # # # # # # # # # #     'bmi': 10.0
# # # # # # # # # # # # # }

# # # # # # # # # # # # # max_values = {
# # # # # # # # # # # # #     'age': 82,
# # # # # # # # # # # # #     'hypertension': 1,
# # # # # # # # # # # # #     'heart_disease': 1,
# # # # # # # # # # # # #     'avg_glucose_level': 271.74,
# # # # # # # # # # # # #     'bmi': 97.6
# # # # # # # # # # # # # }

# # # # # # # # # # # # # def scale_value(val, min_val, max_val):
# # # # # # # # # # # # #     """Min-Max scaling"""
# # # # # # # # # # # # #     return (val - min_val) / (max_val - min_val) if max_val != min_val else 0

# # # # # # # # # # # # # def preprocess_input(data):
# # # # # # # # # # # # #     # Create base feature dict
# # # # # # # # # # # # #     feature_dict = {col: 0 for col in expected_features}

# # # # # # # # # # # # #     # Numeric with scaling
# # # # # # # # # # # # #     feature_dict['age'] = scale_value(float(data['age']), min_values['age'], max_values['age'])
# # # # # # # # # # # # #     feature_dict['hypertension'] = scale_value(int(data['hypertension']), min_values['hypertension'], max_values['hypertension'])
# # # # # # # # # # # # #     feature_dict['heart_disease'] = scale_value(int(data['heart_disease']), min_values['heart_disease'], max_values['heart_disease'])
# # # # # # # # # # # # #     feature_dict['avg_glucose_level'] = scale_value(float(data['avg_glucose_level']), min_values['avg_glucose_level'], max_values['avg_glucose_level'])
# # # # # # # # # # # # #     feature_dict['bmi'] = scale_value(float(data['bmi']), min_values['bmi'], max_values['bmi'])

# # # # # # # # # # # # #     # One-hot categorical
# # # # # # # # # # # # #     if data['gender'] == 'Male':
# # # # # # # # # # # # #         feature_dict['gender_Male'] = 1

# # # # # # # # # # # # #     if data['ever_married'] == 'Yes':
# # # # # # # # # # # # #         feature_dict['ever_married_Yes'] = 1

# # # # # # # # # # # # #     work_type_col = f"work_type_{data['work_type']}"
# # # # # # # # # # # # #     if work_type_col in feature_dict:
# # # # # # # # # # # # #         feature_dict[work_type_col] = 1

# # # # # # # # # # # # #     if data['Residence_type'] == 'Urban':
# # # # # # # # # # # # #         feature_dict['Residence_type_Urban'] = 1

# # # # # # # # # # # # #     smoking_col = f"smoking_status_{data['smoking_status']}"
# # # # # # # # # # # # #     if smoking_col in feature_dict:
# # # # # # # # # # # # #         feature_dict[smoking_col] = 1

# # # # # # # # # # # # #     # Convert dict to array in expected order
# # # # # # # # # # # # #     return np.array([[feature_dict[col] for col in expected_features]], dtype=float)

# # # # # # # # # # # # # @app.route('/')
# # # # # # # # # # # # # def home():
# # # # # # # # # # # # #     return render_template('index.html')

# # # # # # # # # # # # # @app.route('/predict', methods=['POST'])
# # # # # # # # # # # # # def predict():
# # # # # # # # # # # # #     try:
# # # # # # # # # # # # #         data = request.json
# # # # # # # # # # # # #         features = preprocess_input(data)
# # # # # # # # # # # # #         prediction = model.predict(features)[0][0]
# # # # # # # # # # # # #         label = "Stroke" if prediction >= 0.5 else "No Stroke"
# # # # # # # # # # # # #         return jsonify({
# # # # # # # # # # # # #             'prediction': label,
# # # # # # # # # # # # #             'probability': round(float(prediction), 4)
# # # # # # # # # # # # #         })
# # # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # # #         return jsonify({'error': str(e)})

# # # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # # #     app.run(debug=True)



# # # # # # # # # # # # from flask import Flask, render_template, request, jsonify
# # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # import tensorflow as tf

# # # # # # # # # # # # app = Flask(__name__)

# # # # # # # # # # # # # Load trained model
# # # # # # # # # # # # model = tf.keras.models.load_model("dnn1.keras")

# # # # # # # # # # # # # Expected features in correct order (must match training)
# # # # # # # # # # # # expected_features = [
# # # # # # # # # # # #     'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
# # # # # # # # # # # #     'gender_Male', 'ever_married_Yes',
# # # # # # # # # # # #     'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
# # # # # # # # # # # #     'work_type_Self-employed',
# # # # # # # # # # # #     'Residence_type_Urban',
# # # # # # # # # # # #     'smoking_status_formerly smoked', 'smoking_status_never smoked',
# # # # # # # # # # # #     'smoking_status_smokes'
# # # # # # # # # # # # ]

# # # # # # # # # # # # # Min and Max values from training dataset
# # # # # # # # # # # # # Replace with your actual dataset values for best accuracy
# # # # # # # # # # # # min_values = {
# # # # # # # # # # # #     'age': 0,
# # # # # # # # # # # #     'hypertension': 0,
# # # # # # # # # # # #     'heart_disease': 0,
# # # # # # # # # # # #     'avg_glucose_level': 55.12,
# # # # # # # # # # # #     'bmi': 10.0
# # # # # # # # # # # # }

# # # # # # # # # # # # max_values = {
# # # # # # # # # # # #     'age': 82,
# # # # # # # # # # # #     'hypertension': 1,
# # # # # # # # # # # #     'heart_disease': 1,
# # # # # # # # # # # #     'avg_glucose_level': 271.74,
# # # # # # # # # # # #     'bmi': 97.6
# # # # # # # # # # # # }

# # # # # # # # # # # # def scale_value(val, min_val, max_val):
# # # # # # # # # # # #     """Min-Max scaling"""
# # # # # # # # # # # #     return (val - min_val) / (max_val - min_val) if max_val != min_val else 0

# # # # # # # # # # # # def preprocess_input(data):
# # # # # # # # # # # #     # Normalize strings to avoid mismatch
# # # # # # # # # # # #     data['gender'] = data['gender'].strip().title()
# # # # # # # # # # # #     data['ever_married'] = data['ever_married'].strip().title()
# # # # # # # # # # # #     data['work_type'] = data['work_type'].strip().replace("_", " ").title()
# # # # # # # # # # # #     data['Residence_type'] = data['Residence_type'].strip().title()
# # # # # # # # # # # #     data['smoking_status'] = data['smoking_status'].strip().lower()

# # # # # # # # # # # #     # Create base feature dict
# # # # # # # # # # # #     feature_dict = {col: 0 for col in expected_features}

# # # # # # # # # # # #     # Numeric with scaling
# # # # # # # # # # # #     feature_dict['age'] = scale_value(float(data['age']), min_values['age'], max_values['age'])
# # # # # # # # # # # #     feature_dict['hypertension'] = scale_value(int(data['hypertension']), min_values['hypertension'], max_values['hypertension'])
# # # # # # # # # # # #     feature_dict['heart_disease'] = scale_value(int(data['heart_disease']), min_values['heart_disease'], max_values['heart_disease'])
# # # # # # # # # # # #     feature_dict['avg_glucose_level'] = scale_value(float(data['avg_glucose_level']), min_values['avg_glucose_level'], max_values['avg_glucose_level'])
# # # # # # # # # # # #     feature_dict['bmi'] = scale_value(float(data['bmi']), min_values['bmi'], max_values['bmi'])

# # # # # # # # # # # #     # One-hot categorical
# # # # # # # # # # # #     if data['gender'] == 'Male':
# # # # # # # # # # # #         feature_dict['gender_Male'] = 1

# # # # # # # # # # # #     if data['ever_married'] == 'Yes':
# # # # # # # # # # # #         feature_dict['ever_married_Yes'] = 1

# # # # # # # # # # # #     work_type_col = f"work_type_{data['work_type']}"
# # # # # # # # # # # #     if work_type_col in feature_dict:
# # # # # # # # # # # #         feature_dict[work_type_col] = 1

# # # # # # # # # # # #     if data['Residence_type'] == 'Urban':
# # # # # # # # # # # #         feature_dict['Residence_type_Urban'] = 1

# # # # # # # # # # # #     smoking_col = f"smoking_status_{data['smoking_status']}"
# # # # # # # # # # # #     if smoking_col in feature_dict:
# # # # # # # # # # # #         feature_dict[smoking_col] = 1

# # # # # # # # # # # #     # Convert dict to array in expected order
# # # # # # # # # # # #     return np.array([[feature_dict[col] for col in expected_features]], dtype=float)

# # # # # # # # # # # # @app.route('/')
# # # # # # # # # # # # def home():
# # # # # # # # # # # #     return render_template('index.html')

# # # # # # # # # # # # @app.route('/predict', methods=['POST'])
# # # # # # # # # # # # def predict():
# # # # # # # # # # # #     try:
# # # # # # # # # # # #         data = request.json
# # # # # # # # # # # #         features = preprocess_input(data)

# # # # # # # # # # # #         # Predict and force probability output
# # # # # # # # # # # #         raw_prediction = model.predict(features)[0][0]
# # # # # # # # # # # #         probability = float(tf.sigmoid(raw_prediction))  # ensures value between 0–1

# # # # # # # # # # # #         label = "Stroke" if probability >= 0.5 else "No Stroke"

# # # # # # # # # # # #         # Debugging output
# # # # # # # # # # # #         print("DEBUG: Raw model output =", raw_prediction)
# # # # # # # # # # # #         print("DEBUG: Probability =", probability)
# # # # # # # # # # # #         print("DEBUG: Final Label =", label)

# # # # # # # # # # # #         return jsonify({
# # # # # # # # # # # #             'prediction': label,
# # # # # # # # # # # #             'probability': round(probability, 4)
# # # # # # # # # # # #         })
# # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # #         return jsonify({'error': str(e)})

# # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # #     app.run(debug=True)



# # # # # # # # # # # # from flask import Flask, render_template, request, jsonify
# # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # import tensorflow as tf

# # # # # # # # # # # # app = Flask(__name__)

# # # # # # # # # # # # # Load trained model
# # # # # # # # # # # # model = tf.keras.models.load_model("dnn1.keras")

# # # # # # # # # # # # # Expected features in correct order (must match training)
# # # # # # # # # # # # expected_features = [
# # # # # # # # # # # #     'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
# # # # # # # # # # # #     'gender_Male', 'ever_married_Yes',
# # # # # # # # # # # #     'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
# # # # # # # # # # # #     'work_type_Self-employed',
# # # # # # # # # # # #     'Residence_type_Urban',
# # # # # # # # # # # #     'smoking_status_formerly smoked', 'smoking_status_never smoked',
# # # # # # # # # # # #     'smoking_status_smokes'
# # # # # # # # # # # # ]

# # # # # # # # # # # # # Min and Max values from training dataset
# # # # # # # # # # # # min_values = {
# # # # # # # # # # # #     'age': 0,
# # # # # # # # # # # #     'hypertension': 0,
# # # # # # # # # # # #     'heart_disease': 0,
# # # # # # # # # # # #     'avg_glucose_level': 55.12,
# # # # # # # # # # # #     'bmi': 10.0
# # # # # # # # # # # # }

# # # # # # # # # # # # max_values = {
# # # # # # # # # # # #     'age': 82,
# # # # # # # # # # # #     'hypertension': 1,
# # # # # # # # # # # #     'heart_disease': 1,
# # # # # # # # # # # #     'avg_glucose_level': 271.74,
# # # # # # # # # # # #     'bmi': 97.6
# # # # # # # # # # # # }

# # # # # # # # # # # # def scale_value(val, min_val, max_val):
# # # # # # # # # # # #     """Min-Max scaling"""
# # # # # # # # # # # #     return (val - min_val) / (max_val - min_val) if max_val != min_val else 0

# # # # # # # # # # # # def preprocess_input(data):
# # # # # # # # # # # #     # Create base feature dict
# # # # # # # # # # # #     feature_dict = {col: 0 for col in expected_features}

# # # # # # # # # # # #     # Numeric with scaling
# # # # # # # # # # # #     feature_dict['age'] = scale_value(float(data['age']), min_values['age'], max_values['age'])
# # # # # # # # # # # #     feature_dict['hypertension'] = scale_value(int(data['hypertension']), min_values['hypertension'], max_values['hypertension'])
# # # # # # # # # # # #     feature_dict['heart_disease'] = scale_value(int(data['heart_disease']), min_values['heart_disease'], max_values['heart_disease'])
# # # # # # # # # # # #     feature_dict['avg_glucose_level'] = scale_value(float(data['avg_glucose_level']), min_values['avg_glucose_level'], max_values['avg_glucose_level'])
# # # # # # # # # # # #     feature_dict['bmi'] = scale_value(float(data['bmi']), min_values['bmi'], max_values['bmi'])

# # # # # # # # # # # #     # One-hot categorical
# # # # # # # # # # # #     if data['gender'] == 'Male':
# # # # # # # # # # # #         feature_dict['gender_Male'] = 1

# # # # # # # # # # # #     if data['ever_married'] == 'Yes':
# # # # # # # # # # # #         feature_dict['ever_married_Yes'] = 1

# # # # # # # # # # # #     work_type_col = f"work_type_{data['work_type']}"
# # # # # # # # # # # #     if work_type_col in feature_dict:
# # # # # # # # # # # #         feature_dict[work_type_col] = 1

# # # # # # # # # # # #     if data['Residence_type'] == 'Urban':
# # # # # # # # # # # #         feature_dict['Residence_type_Urban'] = 1

# # # # # # # # # # # #     smoking_col = f"smoking_status_{data['smoking_status']}"
# # # # # # # # # # # #     if smoking_col in feature_dict:
# # # # # # # # # # # #         feature_dict[smoking_col] = 1

# # # # # # # # # # # #     # Convert dict to array in expected order
# # # # # # # # # # # #     return np.array([[feature_dict[col] for col in expected_features]], dtype=float)

# # # # # # # # # # # # @app.route('/')
# # # # # # # # # # # # def home():
# # # # # # # # # # # #     return render_template('index.html')

# # # # # # # # # # # # @app.route('/predict', methods=['POST'])
# # # # # # # # # # # # def predict():
# # # # # # # # # # # #     try:
# # # # # # # # # # # #         data = request.json
# # # # # # # # # # # #         features = preprocess_input(data)
# # # # # # # # # # # #         prediction_prob = float(model.predict(features)[0][0])

# # # # # # # # # # # #         label = "Stroke" if prediction_prob >= 0.5 else "No Stroke"

# # # # # # # # # # # #         return jsonify({
# # # # # # # # # # # #             'prediction': label,
# # # # # # # # # # # #             'probability': round(prediction_prob, 4)
# # # # # # # # # # # #         })
# # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # #         return jsonify({'error': str(e)})

# # # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # # #     app.run(debug=True)

# # # # # # # # # # # from flask import Flask, render_template, request, jsonify
# # # # # # # # # # # import numpy as np
# # # # # # # # # # # import pandas as pd
# # # # # # # # # # # import tensorflow as tf
# # # # # # # # # # # import joblib

# # # # # # # # # # # app = Flask(__name__)

# # # # # # # # # # # # ===============================
# # # # # # # # # # # # Load trained model + preprocessors
# # # # # # # # # # # # ===============================
# # # # # # # # # # # model = tf.keras.models.load_model("dnn1.keras")

# # # # # # # # # # # # Load imputer and scaler
# # # # # # # # # # # imputer = joblib.load("stroke_imputer.pkl")
# # # # # # # # # # # scaler = joblib.load("stroke_scaler.pkl")

# # # # # # # # # # # # Load original dataset to get expected features
# # # # # # # # # # # dataset = pd.read_csv("dataset.csv")

# # # # # # # # # # # if 'stroke' in dataset.columns:
# # # # # # # # # # #     dataset = dataset.drop('stroke', axis=1)

# # # # # # # # # # # # One-hot encode training dataset to get expected feature order
# # # # # # # # # # # dataset_encoded = pd.get_dummies(dataset, drop_first=True)
# # # # # # # # # # # expected_features = dataset_encoded.columns.tolist()

# # # # # # # # # # # # ===============================
# # # # # # # # # # # # Helper: preprocess input sample
# # # # # # # # # # # # ===============================
# # # # # # # # # # # def preprocess_input(data):
# # # # # # # # # # #     # Convert input to dataframe
# # # # # # # # # # #     sample_df = pd.DataFrame([data])

# # # # # # # # # # #     # One-hot encode (same as training)
# # # # # # # # # # #     sample_encoded = pd.get_dummies(sample_df, drop_first=True)

# # # # # # # # # # #     # Align with training features
# # # # # # # # # # #     for col in expected_features:
# # # # # # # # # # #         if col not in sample_encoded.columns:
# # # # # # # # # # #             sample_encoded[col] = 0

# # # # # # # # # # #     sample_encoded = sample_encoded[expected_features]

# # # # # # # # # # #     # Apply imputer + scaler
# # # # # # # # # # #     sample_imputed = imputer.transform(sample_encoded)
# # # # # # # # # # #     sample_scaled = scaler.transform(sample_imputed)

# # # # # # # # # # #     return sample_scaled

# # # # # # # # # # # # ===============================
# # # # # # # # # # # # Routes
# # # # # # # # # # # # ===============================
# # # # # # # # # # # @app.route('/')
# # # # # # # # # # # def home():
# # # # # # # # # # #     return render_template('index.html')

# # # # # # # # # # # @app.route('/predict', methods=['POST'])
# # # # # # # # # # # def predict():
# # # # # # # # # # #     try:
# # # # # # # # # # #         data = request.json  # input from frontend
# # # # # # # # # # #         features = preprocess_input(data)

# # # # # # # # # # #         prediction_prob = float(model.predict(features)[0][0])
# # # # # # # # # # #         label = "Stroke" if prediction_prob >= 0.5 else "No Stroke"

# # # # # # # # # # #         return jsonify({
# # # # # # # # # # #             'prediction': label,
# # # # # # # # # # #             'probability': round(prediction_prob, 4)
# # # # # # # # # # #         })

# # # # # # # # # # #     except Exception as e:
# # # # # # # # # # #         return jsonify({'error': str(e)})

# # # # # # # # # # # # ===============================
# # # # # # # # # # # # Run app
# # # # # # # # # # # # ===============================
# # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # #     app.run(debug=True)

# # # # # # # # # # # from flask import Flask, render_template, request, jsonify
# # # # # # # # # # # import numpy as np
# # # # # # # # # # # import pandas as pd
# # # # # # # # # # # import joblib
# # # # # # # # # # # import tensorflow as tf

# # # # # # # # # # # app = Flask(__name__)

# # # # # # # # # # # # ==============================
# # # # # # # # # # # # 1. Load trained model + preprocessors
# # # # # # # # # # # # ==============================
# # # # # # # # # # # model = tf.keras.models.load_model("dnn5.keras")
# # # # # # # # # # # imputer = joblib.load("stroke_imputer.pkl")
# # # # # # # # # # # scaler = joblib.load("stroke_scaler.pkl")

# # # # # # # # # # # # Load dataset (only for feature reference, not training again)
# # # # # # # # # # # dataset = pd.read_csv("dataset.csv")

# # # # # # # # # # # # Drop target + id column if they exist
# # # # # # # # # # # for col in ["stroke", "id"]:
# # # # # # # # # # #     if col in dataset.columns:
# # # # # # # # # # #         dataset = dataset.drop(col, axis=1)

# # # # # # # # # # # # One-hot encode like in training
# # # # # # # # # # # dataset_encoded = pd.get_dummies(dataset, drop_first=True)

# # # # # # # # # # # # ✅ This is the exact feature set used during training
# # # # # # # # # # # expected_features = dataset_encoded.columns.tolist()

# # # # # # # # # # # # ==============================
# # # # # # # # # # # # 2. Preprocessing Function
# # # # # # # # # # # # ==============================
# # # # # # # # # # # def preprocess_input(data):
# # # # # # # # # # #     # Convert dict to DataFrame
# # # # # # # # # # #     input_df = pd.DataFrame([data])

# # # # # # # # # # #     # One-hot encode like training (drop_first=True)
# # # # # # # # # # #     input_encoded = pd.get_dummies(input_df, drop_first=True)

# # # # # # # # # # #     # Add missing columns from training
# # # # # # # # # # #     for col in expected_features:
# # # # # # # # # # #         if col not in input_encoded.columns:
# # # # # # # # # # #             input_encoded[col] = 0

# # # # # # # # # # #     # Ensure extra columns not in training are dropped
# # # # # # # # # # #     input_encoded = input_encoded[expected_features]

# # # # # # # # # # #     # Impute + scale
# # # # # # # # # # #     input_imputed = imputer.transform(input_encoded)
# # # # # # # # # # #     input_scaled = scaler.transform(input_imputed)

# # # # # # # # # # #     return input_scaled


# # # # # # # # # # # # ==============================
# # # # # # # # # # # # 3. Routes
# # # # # # # # # # # # ==============================
# # # # # # # # # # # @app.route("/")
# # # # # # # # # # # def home():
# # # # # # # # # # #     return render_template("index.html")

# # # # # # # # # # # @app.route("/predict", methods=["POST"])
# # # # # # # # # # # def predict():
# # # # # # # # # # #     try:
# # # # # # # # # # #         data = request.json  # JSON from frontend
# # # # # # # # # # #         features = preprocess_input(data)

# # # # # # # # # # #         prediction_prob = float(model.predict(features)[0][0])
# # # # # # # # # # #         label = "Stroke" if prediction_prob >= 0.5 else "No Stroke"

# # # # # # # # # # #         return jsonify({
# # # # # # # # # # #             "prediction": label,
# # # # # # # # # # #             "probability": round(prediction_prob, 4)
# # # # # # # # # # #         })
# # # # # # # # # # #     except Exception as e:
# # # # # # # # # # #         return jsonify({"error": str(e)})

# # # # # # # # # # # # ==============================
# # # # # # # # # # # # 4. Run
# # # # # # # # # # # # ==============================
# # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # #     app.run(debug=True)
# # # # # # # # # # from flask import Flask, render_template, request, jsonify
# # # # # # # # # # import pandas as pd
# # # # # # # # # # import joblib
# # # # # # # # # # import tensorflow as tf
# # # # # # # # # # from sklearn.preprocessing import OneHotEncoder, StandardScaler
# # # # # # # # # # from sklearn.impute import SimpleImputer
# # # # # # # # # # from sklearn.compose import ColumnTransformer
# # # # # # # # # # from sklearn.pipeline import Pipeline
# # # # # # # # # # import numpy as np

# # # # # # # # # # app = Flask(__name__)

# # # # # # # # # # # ==============================
# # # # # # # # # # # 1. Load trained model
# # # # # # # # # # # ==============================
# # # # # # # # # # model = tf.keras.models.load_model("dnn5.keras")

# # # # # # # # # # # ==============================
# # # # # # # # # # # 2. Load dataset for reference
# # # # # # # # # # # ==============================
# # # # # # # # # # dataset = pd.read_csv("dataset.csv")

# # # # # # # # # # # Drop target + id if exist
# # # # # # # # # # for col in ["stroke", "id"]:
# # # # # # # # # #     if col in dataset.columns:
# # # # # # # # # #         dataset = dataset.drop(col, axis=1)

# # # # # # # # # # # Separate features
# # # # # # # # # # categorical_features = dataset.select_dtypes(include=['object']).columns.tolist()
# # # # # # # # # # numeric_features = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()

# # # # # # # # # # # ==============================
# # # # # # # # # # # 3. Create & load preprocessing pipeline
# # # # # # # # # # # ==============================
# # # # # # # # # # # If you already saved your imputer + scaler, you can combine into a pipeline
# # # # # # # # # # # Here, we create it dynamically like training for demonstration
# # # # # # # # # # categorical_transformer = Pipeline([
# # # # # # # # # #     ('imputer', SimpleImputer(strategy='most_frequent')),
# # # # # # # # # #     ('onehot', OneHotEncoder(handle_unknown='ignore'))  # ignore unseen categories
# # # # # # # # # # ])

# # # # # # # # # # numeric_transformer = Pipeline([
# # # # # # # # # #     ('imputer', SimpleImputer(strategy='mean')),
# # # # # # # # # #     ('scaler', StandardScaler())
# # # # # # # # # # ])

# # # # # # # # # # preprocessor = ColumnTransformer([
# # # # # # # # # #     ('cat', categorical_transformer, categorical_features),
# # # # # # # # # #     ('num', numeric_transformer, numeric_features)
# # # # # # # # # # ])

# # # # # # # # # # # Fit the pipeline on dataset (only for demo; in real, load fitted pipeline)
# # # # # # # # # # preprocessor.fit(dataset)

# # # # # # # # # # # Save pipeline for reuse (optional)
# # # # # # # # # # # joblib.dump(preprocessor, 'stroke_pipeline.pkl')
# # # # # # # # # # # preprocessor = joblib.load('stroke_pipeline.pkl')

# # # # # # # # # # # ==============================
# # # # # # # # # # # 4. Preprocessing function
# # # # # # # # # # # ==============================
# # # # # # # # # # def preprocess_input(data):
# # # # # # # # # #     input_df = pd.DataFrame([data])
# # # # # # # # # #     features = preprocessor.transform(input_df)
# # # # # # # # # #     return features

# # # # # # # # # # # ==============================
# # # # # # # # # # # 5. Flask routes
# # # # # # # # # # # ==============================
# # # # # # # # # # @app.route("/")
# # # # # # # # # # def home():
# # # # # # # # # #     return render_template("index.html")

# # # # # # # # # # @app.route("/predict", methods=["POST"])
# # # # # # # # # # def predict():
# # # # # # # # # #     try:
# # # # # # # # # #         data = request.json
# # # # # # # # # #         features = preprocess_input(data)

# # # # # # # # # #         prediction_prob = float(model.predict(features)[0][0])
# # # # # # # # # #         label = "No Stroke" if prediction_prob >= 0.5 else "Stroke"


# # # # # # # # # #         return jsonify({
# # # # # # # # # #             "prediction": label,
# # # # # # # # # #             "probability": round(prediction_prob, 4)
# # # # # # # # # #         })
# # # # # # # # # #     except Exception as e:
# # # # # # # # # #         return jsonify({"error": str(e)})

# # # # # # # # # # # ==============================
# # # # # # # # # # # 6. Run Flask app
# # # # # # # # # # # ==============================
# # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # #     app.run(debug=True)


# # # # # # # # # from flask import Flask, render_template, request, jsonify
# # # # # # # # # import numpy as np
# # # # # # # # # import pandas as pd
# # # # # # # # # import tensorflow as tf
# # # # # # # # # import joblib

# # # # # # # # # app = Flask(__name__)

# # # # # # # # # # ==============================
# # # # # # # # # # 1. Load trained model + preprocessors
# # # # # # # # # # ==============================
# # # # # # # # # model = tf.keras.models.load_model("dnn5.keras")
# # # # # # # # # imputer = joblib.load("sroke_imputer.pkl")
# # # # # # # # # scaler = joblib.load("stroke_scaler.pkl")

# # # # # # # # # # ==============================
# # # # # # # # # # 2. Expected features & min-max values (same as training)
# # # # # # # # # # ==============================
# # # # # # # # # expected_features = [
# # # # # # # # #     'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
# # # # # # # # #     'gender_Male', 'ever_married_Yes',
# # # # # # # # #     'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
# # # # # # # # #     'work_type_Self-employed',
# # # # # # # # #     'Residence_type_Urban',
# # # # # # # # #     'smoking_status_formerly smoked', 'smoking_status_never smoked',
# # # # # # # # #     'smoking_status_smokes'
# # # # # # # # # ]

# # # # # # # # # min_values = {
# # # # # # # # #     'age': 0,
# # # # # # # # #     'hypertension': 0,
# # # # # # # # #     'heart_disease': 0,
# # # # # # # # #     'avg_glucose_level': 55.12,
# # # # # # # # #     'bmi': 10.0
# # # # # # # # # }

# # # # # # # # # max_values = {
# # # # # # # # #     'age': 82,
# # # # # # # # #     'hypertension': 1,
# # # # # # # # #     'heart_disease': 1,
# # # # # # # # #     'avg_glucose_level': 271.74,
# # # # # # # # #     'bmi': 97.6
# # # # # # # # # }

# # # # # # # # # # ==============================
# # # # # # # # # # 3. Scaling function
# # # # # # # # # # ==============================
# # # # # # # # # def scale_value(val, min_val, max_val):
# # # # # # # # #     """Min-Max scaling"""
# # # # # # # # #     return (val - min_val) / (max_val - min_val) if max_val != min_val else 0

# # # # # # # # # # ==============================
# # # # # # # # # # 4. Preprocess input function
# # # # # # # # # # ==============================
# # # # # # # # # def preprocess_input(data):
# # # # # # # # #     # Base feature dict
# # # # # # # # #     feature_dict = {col: 0 for col in expected_features}

# # # # # # # # #     # Numeric scaling
# # # # # # # # #     feature_dict['age'] = scale_value(float(data['age']), min_values['age'], max_values['age'])
# # # # # # # # #     feature_dict['hypertension'] = scale_value(int(data['hypertension']), min_values['hypertension'], max_values['hypertension'])
# # # # # # # # #     feature_dict['heart_disease'] = scale_value(int(data['heart_disease']), min_values['heart_disease'], max_values['heart_disease'])
# # # # # # # # #     feature_dict['avg_glucose_level'] = scale_value(float(data['avg_glucose_level']), min_values['avg_glucose_level'], max_values['avg_glucose_level'])
# # # # # # # # #     feature_dict['bmi'] = scale_value(float(data['bmi']), min_values['bmi'], max_values['bmi'])

# # # # # # # # #     # One-hot categorical
# # # # # # # # #     if data['gender'] == 'Male':
# # # # # # # # #         feature_dict['gender_Male'] = 1

# # # # # # # # #     if data['ever_married'] == 'Yes':
# # # # # # # # #         feature_dict['ever_married_Yes'] = 1

# # # # # # # # #     work_type_col = f"work_type_{data['work_type']}"
# # # # # # # # #     if work_type_col in feature_dict:
# # # # # # # # #         feature_dict[work_type_col] = 1

# # # # # # # # #     if data['Residence_type'] == 'Urban':
# # # # # # # # #         feature_dict['Residence_type_Urban'] = 1

# # # # # # # # #     smoking_col = f"smoking_status_{data['smoking_status']}"
# # # # # # # # #     if smoking_col in feature_dict:
# # # # # # # # #         feature_dict[smoking_col] = 1

# # # # # # # # #     # Convert dict to DataFrame for imputer/scaler
# # # # # # # # #     input_df = pd.DataFrame([feature_dict])

# # # # # # # # #     # Impute missing values
# # # # # # # # #     input_imputed = imputer.transform(input_df)

# # # # # # # # #     # Scale features
# # # # # # # # #     input_scaled = scaler.transform(input_imputed)

# # # # # # # # #     return input_scaled

# # # # # # # # # # ==============================
# # # # # # # # # # 5. Flask routes
# # # # # # # # # # ==============================
# # # # # # # # # @app.route("/")
# # # # # # # # # def home():
# # # # # # # # #     return render_template("index.html")

# # # # # # # # # @app.route("/predict", methods=["POST"])
# # # # # # # # # def predict():
# # # # # # # # #     try:
# # # # # # # # #         data = request.json
# # # # # # # # #         features = preprocess_input(data)
# # # # # # # # #         prediction_prob = float(model.predict(features)[0][0])

# # # # # # # # #         # Adjust label based on training encoding
# # # # # # # # #         label = "Stroke" if prediction_prob >= 0.5 else "No Stroke"

# # # # # # # # #         return jsonify({
# # # # # # # # #             'prediction': label,
# # # # # # # # #             'probability': round(prediction_prob, 4)
# # # # # # # # #         })
# # # # # # # # #     except Exception as e:
# # # # # # # # #         return jsonify({'error': str(e)})

# # # # # # # # # # ==============================
# # # # # # # # # # 6. Run Flask app
# # # # # # # # # # ==============================
# # # # # # # # # if __name__ == "__main__":
# # # # # # # # #     app.run(debug=True)


# # # # # # # # from flask import Flask, render_template, request, jsonify
# # # # # # # # import numpy as np
# # # # # # # # import pandas as pd
# # # # # # # # import tensorflow as tf
# # # # # # # # import joblib

# # # # # # # # app = Flask(__name__)

# # # # # # # # # ==============================
# # # # # # # # # 1. Load trained model + preprocessors
# # # # # # # # # ==============================
# # # # # # # # model = tf.keras.models.load_model("dnn5.keras")
# # # # # # # # imputer = joblib.load("stroke_imputer.pkl")
# # # # # # # # scaler = joblib.load("stroke_scaler.pkl")

# # # # # # # # # ==============================
# # # # # # # # # 2. Full expected features from training
# # # # # # # # # ==============================
# # # # # # # # expected_features = [
# # # # # # # #     'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    
# # # # # # # #     'gender_Female', 'gender_Male', 'gender_Other',
# # # # # # # #     'ever_married_No', 'ever_married_Yes',
# # # # # # # #     'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
# # # # # # # #     'work_type_Self-employed', 'work_type_children',
# # # # # # # #     'Residence_type_Rural', 'Residence_type_Urban',
# # # # # # # #     'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
# # # # # # # # ]

# # # # # # # # # ==============================
# # # # # # # # # 3. Preprocessing function
# # # # # # # # # ==============================
# # # # # # # # def preprocess_input(data):
# # # # # # # #     # Base dict with all features 0
# # # # # # # #     feature_dict = {col: 0 for col in expected_features}

# # # # # # # #     # Numeric features (scale with min-max or leave as-is if using scaler)
# # # # # # # #     feature_dict['age'] = float(data.get('age', 0))
# # # # # # # #     feature_dict['hypertension'] = int(data.get('hypertension', 0))
# # # # # # # #     feature_dict['heart_disease'] = int(data.get('heart_disease', 0))
# # # # # # # #     feature_dict['avg_glucose_level'] = float(data.get('avg_glucose_level', 0))
# # # # # # # #     feature_dict['bmi'] = float(data.get('bmi', 0))

# # # # # # # #     # One-hot categorical
# # # # # # # #     gender_col = f"gender_{data.get('gender', 'Other')}"
# # # # # # # #     if gender_col in feature_dict:
# # # # # # # #         feature_dict[gender_col] = 1

# # # # # # # #     married_col = f"ever_married_{data.get('ever_married', 'No')}"
# # # # # # # #     if married_col in feature_dict:
# # # # # # # #         feature_dict[married_col] = 1

# # # # # # # #     work_col = f"work_type_{data.get('work_type', 'Private')}"
# # # # # # # #     if work_col in feature_dict:
# # # # # # # #         feature_dict[work_col] = 1

# # # # # # # #     residence_col = f"Residence_type_{data.get('Residence_type', 'Rural')}"
# # # # # # # #     if residence_col in feature_dict:
# # # # # # # #         feature_dict[residence_col] = 1

# # # # # # # #     smoke_col = f"smoking_status_{data.get('smoking_status', 'never smoked')}"
# # # # # # # #     if smoke_col in feature_dict:
# # # # # # # #         feature_dict[smoke_col] = 1

# # # # # # # #     # Convert to DataFrame
# # # # # # # #     input_df = pd.DataFrame([feature_dict])

# # # # # # # #     # Impute + scale
# # # # # # # #     input_imputed = imputer.transform(input_df)
# # # # # # # #     input_scaled = scaler.transform(input_imputed)

# # # # # # # #     return input_scaled

# # # # # # # # # ==============================
# # # # # # # # # 4. Flask routes
# # # # # # # # # ==============================
# # # # # # # # @app.route("/")
# # # # # # # # def home():
# # # # # # # #     return render_template("index.html")

# # # # # # # # @app.route("/predict", methods=["POST"])
# # # # # # # # def predict():
# # # # # # # #     try:
# # # # # # # #         data = request.json
# # # # # # # #         features = preprocess_input(data)
# # # # # # # #         prediction_prob = float(model.predict(features)[0][0])

# # # # # # # #         # Adjust label based on training encoding
# # # # # # # #         label = "Stroke" if prediction_prob >= 0.5 else "No Stroke"

# # # # # # # #         return jsonify({
# # # # # # # #             'prediction': label,
# # # # # # # #             'probability': round(prediction_prob, 4)
# # # # # # # #         })
# # # # # # # #     except Exception as e:
# # # # # # # #         return jsonify({'error': str(e)})

# # # # # # # # # ==============================
# # # # # # # # # 5. Run Flask app
# # # # # # # # # ==============================
# # # # # # # # if __name__ == "__main__":
# # # # # # # #     app.run(debug=True)


# # # # # # # from flask import Flask, render_template, request, jsonify
# # # # # # # import numpy as np
# # # # # # # import pandas as pd
# # # # # # # import tensorflow as tf
# # # # # # # import joblib

# # # # # # # app = Flask(__name__)

# # # # # # # # -------------------------
# # # # # # # # Load model + preprocessors
# # # # # # # # -------------------------
# # # # # # # model = tf.keras.models.load_model("dnn5.keras")
# # # # # # # imputer = joblib.load("stroke_imputer.pkl")
# # # # # # # scaler = joblib.load("stroke_scaler.pkl")

# # # # # # # expected_features = [
# # # # # # #     'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
# # # # # # #     'gender_Female', 'gender_Male', 'gender_Other',
# # # # # # #     'ever_married_No', 'ever_married_Yes',
# # # # # # #     'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
# # # # # # #     'work_type_Self-employed', 'work_type_children',
# # # # # # #     'Residence_type_Rural', 'Residence_type_Urban',
# # # # # # #     'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
# # # # # # # ]

# # # # # # # def preprocess_input(data):
# # # # # # #     feature_dict = {col: 0 for col in expected_features}
# # # # # # #     feature_dict['age'] = float(data.get('age', 0))
# # # # # # #     feature_dict['hypertension'] = int(data.get('hypertension', 0))
# # # # # # #     feature_dict['heart_disease'] = int(data.get('heart_disease', 0))
# # # # # # #     feature_dict['avg_glucose_level'] = float(data.get('avg_glucose_level', 0))
# # # # # # #     feature_dict['bmi'] = float(data.get('bmi', 0))

# # # # # # #     gender_col = f"gender_{data.get('gender', 'Other')}"
# # # # # # #     if gender_col in feature_dict:
# # # # # # #         feature_dict[gender_col] = 1

# # # # # # #     married_col = f"ever_married_{data.get('ever_married', 'No')}"
# # # # # # #     if married_col in feature_dict:
# # # # # # #         feature_dict[married_col] = 1

# # # # # # #     work_col = f"work_type_{data.get('work_type', 'Private')}"
# # # # # # #     if work_col in feature_dict:
# # # # # # #         feature_dict[work_col] = 1

# # # # # # #     residence_col = f"Residence_type_{data.get('Residence_type', 'Rural')}"
# # # # # # #     if residence_col in feature_dict:
# # # # # # #         feature_dict[residence_col] = 1

# # # # # # #     smoke_col = f"smoking_status_{data.get('smoking_status', 'never smoked')}"
# # # # # # #     if smoke_col in feature_dict:
# # # # # # #         feature_dict[smoke_col] = 1

# # # # # # #     input_df = pd.DataFrame([feature_dict])
# # # # # # #     input_imputed = imputer.transform(input_df)
# # # # # # #     input_scaled = scaler.transform(input_imputed)
# # # # # # #     return input_scaled

# # # # # # # # -------------------------
# # # # # # # # Routes: pages
# # # # # # # # -------------------------
# # # # # # # @app.route("/")
# # # # # # # def home():
# # # # # # #     return render_template("index.html")


# # # # # # # @app.route("/about")
# # # # # # # def about():
# # # # # # #     return render_template("about.html")


# # # # # # # @app.route("/predict_page")
# # # # # # # def predict_page():
# # # # # # #     # Prediction page (form + ajax)
# # # # # # #     return render_template("predict.html")


# # # # # # # # -------------------------
# # # # # # # # API: predict (expects JSON POST)
# # # # # # # # -------------------------
# # # # # # # @app.route("/predict", methods=["POST"])
# # # # # # # def predict():
# # # # # # #     try:
# # # # # # #         data = request.get_json(force=True)
# # # # # # #         features = preprocess_input(data)
# # # # # # #         prediction_prob = float(model.predict(features)[0][0])
# # # # # # #         label = "Stroke" if prediction_prob >= 0.5 else "No Stroke"
# # # # # # #         return jsonify({
# # # # # # #             'prediction': label,
# # # # # # #             'probability': round(prediction_prob, 4)
# # # # # # #         })
# # # # # # #     except Exception as e:
# # # # # # #         return jsonify({'error': str(e)}), 400


# # # # # # # # -------------------------
# # # # # # # # Run app
# # # # # # # # -------------------------
# # # # # # # if __name__ == "__main__":
# # # # # # #     app.run(debug=True)


# # # # # # from flask import Flask, render_template, request
# # # # # # import pickle
# # # # # # import numpy as np

# # # # # # app = Flask(__name__)

# # # # # # # Load trained model
# # # # # # model = pickle.load(open("model.pkl", "rb"))

# # # # # # @app.route("/")
# # # # # # def home():
# # # # # #     return render_template("index.html")

# # # # # # @app.route("/about")
# # # # # # def about():
# # # # # #     return render_template("about.html")

# # # # # # @app.route("/predict", methods=["GET", "POST"])
# # # # # # def predict():
# # # # # #     if request.method == "POST":
# # # # # #         try:
# # # # # #             age = int(request.form["age"])
# # # # # #             hypertension = int(request.form["hypertension"])
# # # # # #             heart_disease = int(request.form["heart_disease"])
# # # # # #             avg_glucose = float(request.form["avg_glucose_level"])
# # # # # #             bmi = float(request.form["bmi"])
# # # # # #             gender = 1 if request.form["gender"] == "Male" else 0
# # # # # #             married = 1 if request.form["ever_married"] == "Yes" else 0
# # # # # #             work_private = 1 if request.form["work_type"] == "Private" else 0
# # # # # #             work_self = 1 if request.form["work_type"] == "Self-employed" else 0
# # # # # #             work_govt = 1 if request.form["work_type"] == "Govt_job" else 0
# # # # # #             work_never = 1 if request.form["work_type"] == "Never_worked" else 0
# # # # # #             residence = 1 if request.form["residence_type"] == "Urban" else 0
# # # # # #             smoking_formerly = 1 if request.form["smoking_status"] == "formerly smoked" else 0
# # # # # #             smoking_never = 1 if request.form["smoking_status"] == "never smoked" else 0
# # # # # #             smoking_sometimes = 1 if request.form["smoking_status"] == "smokes" else 0

# # # # # #             features = np.array([[
# # # # # #                 age, hypertension, heart_disease, avg_glucose, bmi,
# # # # # #                 gender, married,
# # # # # #                 work_govt, work_never, work_private, work_self,
# # # # # #                 residence,
# # # # # #                 smoking_formerly, smoking_never, smoking_sometimes
# # # # # #             ]])

# # # # # #             prediction = model.predict(features)[0]
# # # # # #             result = "High Risk of Stroke ❌" if prediction == 1 else "Low Risk of Stroke ✅"

# # # # # #             return render_template("predict.html", result=result)

# # # # # #         except Exception as e:
# # # # # #             return render_template("predict.html", result=f"Error: {str(e)}")

# # # # # #     return render_template("predict.html")

# # # # # # if __name__ == "__main__":
# # # # # #     app.run(debug=True)

# # # # # from flask import Flask, render_template, request, jsonify
# # # # # import numpy as np
# # # # # import pickle
# # # # # from tensorflow.keras.models import load_model

# # # # # app = Flask(__name__)

# # # # # # Load preprocessing objects
# # # # # scaler = pickle.load(open("stroke_scaler.pkl", "rb"))
# # # # # imputer = pickle.load(open("stroke_imputer.pkl", "rb"))

# # # # # # Load trained models (you can choose dnn1 or dnn5 based on requirement)
# # # # # model = load_model("dnn5.keras")

# # # # # @app.route('/')
# # # # # def home():
# # # # #     return render_template("index.html")

# # # # # @app.route('/about')
# # # # # def about():
# # # # #     return render_template("about.html")

# # # # # @app.route('/predict', methods=['GET', 'POST'])
# # # # # def predict():
# # # # #     if request.method == "POST":
# # # # #         try:
# # # # #             # Collect data from form
# # # # #             age = float(request.form['age'])
# # # # #             hypertension = int(request.form['hypertension'])
# # # # #             heart_disease = int(request.form['heart_disease'])
# # # # #             avg_glucose_level = float(request.form['avg_glucose_level'])
# # # # #             bmi = float(request.form['bmi'])
# # # # #             gender = request.form['gender']
# # # # #             ever_married = request.form['ever_married']
# # # # #             work_type = request.form['work_type']
# # # # #             residence_type = request.form['Residence_type']
# # # # #             smoking_status = request.form['smoking_status']

# # # # #             # One-hot encoding manually (must match training phase)
# # # # #             input_data = {
# # # # #                 'age': age,
# # # # #                 'hypertension': hypertension,
# # # # #                 'heart_disease': heart_disease,
# # # # #                 'avg_glucose_level': avg_glucose_level,
# # # # #                 'bmi': bmi,
# # # # #                 'gender_Male': 1 if gender == "Male" else 0,
# # # # #                 'ever_married_Yes': 1 if ever_married == "Yes" else 0,
# # # # #                 'work_type_Govt_job': 1 if work_type == "Govt_job" else 0,
# # # # #                 'work_type_Never_worked': 1 if work_type == "Never_worked" else 0,
# # # # #                 'work_type_Private': 1 if work_type == "Private" else 0,
# # # # #                 'work_type_Self-employed': 1 if work_type == "Self-employed" else 0,
# # # # #                 'Residence_type_Urban': 1 if residence_type == "Urban" else 0,
# # # # #                 'smoking_status_formerly smoked': 1 if smoking_status == "formerly smoked" else 0,
# # # # #                 'smoking_status_never smoked': 1 if smoking_status == "never smoked" else 0,
# # # # #                 'smoking_status_smokes': 1 if smoking_status == "smokes" else 0
# # # # #             }

# # # # #             # Convert to array
# # # # #             features = np.array(list(input_data.values())).reshape(1, -1)

# # # # #             # Handle missing values & scale
# # # # #             features = imputer.transform(features)
# # # # #             features = scaler.transform(features)

# # # # #             # Prediction
# # # # #             prediction = model.predict(features)[0][0]
# # # # #             result = "High risk of stroke ❌" if prediction > 0.5 else "Low risk of stroke ✅"

# # # # #             return render_template("predict.html", result=result)

# # # # #         except Exception as e:
# # # # #             return render_template("predict.html", result=f"Error: {str(e)}")

# # # # #     return render_template("predict.html")


# # # # # if __name__ == "__main__":
# # # # #     app.run(debug=True)

# # # # from flask import Flask, request, jsonify, render_template
# # # # import numpy as np
# # # # import joblib

# # # # app = Flask(__name__)

# # # # # Load model and scaler
# # # # model = joblib.load(".pkl")
# # # # scaler = joblib.load("stroke_scaler.pkl")

# # # # # Full expected feature order (must match training data)
# # # # expected_features = [
# # # #     'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke',
# # # #     'gender_Female', 'gender_Male', 'gender_Other',
# # # #     'ever_married_No', 'ever_married_Yes',
# # # #     'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
# # # #     'work_type_Self-employed', 'work_type_children',
# # # #     'Residence_type_Rural', 'Residence_type_Urban',
# # # #     'smoking_status_formerly smoked', 'smoking_status_never smoked',
# # # #     'smoking_status_smokes'
# # # # ]

# # # # @app.route("/")
# # # # def home():
# # # #     return render_template("index.html")

# # # # @app.route("/predict", methods=["POST"])
# # # # def predict():
# # # #     try:
# # # #         # Collect form values
# # # #         age = float(request.form["age"])
# # # #         hypertension = int(request.form["hypertension"])
# # # #         heart_disease = int(request.form["heart_disease"])
# # # #         avg_glucose_level = float(request.form["avg_glucose_level"])
# # # #         bmi = float(request.form["bmi"])
# # # #         gender = request.form["gender"]
# # # #         ever_married = request.form["ever_married"]
# # # #         work_type = request.form["work_type"]
# # # #         residence_type = request.form["Residence_type"]
# # # #         smoking_status = request.form["smoking_status"]

# # # #         # Initialize all features to 0
# # # #         feature_dict = {col: 0 for col in expected_features}

# # # #         # Fill numeric values
# # # #         feature_dict["age"] = age
# # # #         feature_dict["hypertension"] = hypertension
# # # #         feature_dict["heart_disease"] = heart_disease
# # # #         feature_dict["avg_glucose_level"] = avg_glucose_level
# # # #         feature_dict["bmi"] = bmi
# # # #         feature_dict["stroke"] = 0  # we don't know this at prediction time

# # # #         # One-hot encoding
# # # #         if f"gender_{gender}" in feature_dict:
# # # #             feature_dict[f"gender_{gender}"] = 1
# # # #         if f"ever_married_{ever_married}" in feature_dict:
# # # #             feature_dict[f"ever_married_{ever_married}"] = 1
# # # #         if f"work_type_{work_type}" in feature_dict:
# # # #             feature_dict[f"work_type_{work_type}"] = 1
# # # #         if f"Residence_type_{residence_type}" in feature_dict:
# # # #             feature_dict[f"Residence_type_{residence_type}"] = 1
# # # #         if f"smoking_status_{smoking_status}" in feature_dict:
# # # #             feature_dict[f"smoking_status_{smoking_status}"] = 1

# # # #         # Convert into ordered list
# # # #         input_features = [feature_dict[feat] for feat in expected_features]

# # # #         # Scale input
# # # #         scaled_features = scaler.transform([input_features])

# # # #         # Predict
# # # #         prediction = model.predict(scaled_features)[0]
# # # #         result = "Stroke Risk" if prediction == 1 else "No Stroke Risk"

# # # #         return render_template("index.html", prediction_text=f"Prediction: {result}")

# # # #     except Exception as e:
# # # #         return jsonify({"error": str(e)})

# # # # if __name__ == "__main__":
# # # #     app.run(debug=True)


# # # # import numpy as np
# # # # import pandas as pd
# # # # from flask import Flask, request, jsonify, render_template
# # # # import joblib
# # # # from tensorflow.keras.models import load_model

# # # # # -------------------- Load Pretrained Objects --------------------
# # # # # load imputer and scaler
# # # # imputer = joblib.load("stroke_imputer.pkl")   # or .pkl (change if needed)
# # # # scaler = joblib.load("stroke_scaler.pkl")     # or .pkl (change if needed)

# # # # # load DNN model
# # # # model = load_model("dnn5.keras")

# # # # # -------------------- Define Features --------------------
# # # # expected_features = [
# # # #     'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke',
# # # #     'gender_Female', 'gender_Male', 'gender_Other',
# # # #     'ever_married_No', 'ever_married_Yes',
# # # #     'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
# # # #     'work_type_Self-employed', 'work_type_children',
# # # #     'Residence_type_Rural', 'Residence_type_Urban',
# # # #     'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
# # # # ]

# # # # # -------------------- Flask App --------------------
# # # # app = Flask(__name__)

# # # # @app.route("/")
# # # # def home():
# # # #     return render_template("index.html")

# # # # @app.route("/predict", methods=["POST"])
# # # # def predict():
# # # #     try:
# # # #         # Get form data
# # # #         input_data = request.form.to_dict()

# # # #         # Convert to dataframe
# # # #         df = pd.DataFrame([input_data])

# # # #         # Convert numeric values properly
# # # #         for col in ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']:
# # # #             df[col] = pd.to_numeric(df[col], errors='coerce')

# # # #         # Ensure all expected features are present
# # # #         for feature in expected_features:
# # # #             if feature not in df.columns:
# # # #                 df[feature] = 0

# # # #         # Reorder columns
# # # #         df = df[expected_features]

# # # #         # Impute missing values
# # # #         df_imputed = imputer.transform(df)

# # # #         # Scale features
# # # #         df_scaled = scaler.transform(df_imputed)

# # # #         # Predict
# # # #         prediction = model.predict(df_scaled)
# # # #         probability = float(prediction[0][0])
# # # #         result = "High Risk of Stroke" if probability >= 0.5 else "Low Risk of Stroke"

# # # #         return render_template("index.html", prediction_text=f"{result} (Probability: {probability:.2f})")

# # # #     except Exception as e:
# # # #         return jsonify({"error": str(e)})

# # # # if __name__ == "__main__":
# # # #     app.run(debug=True)


# # # # import numpy as np
# # # # import pandas as pd
# # # # from flask import Flask, request, jsonify, render_template
# # # # import joblib
# # # # from tensorflow.keras.models import load_model

# # # # # -------------------- Load Pretrained Objects --------------------
# # # # imputer = joblib.load("stroke_imputer.pkl")
# # # # scaler = joblib.load("stroke_scaler.pkl")
# # # # model = load_model("dnn5.keras")

# # # # # -------------------- Define Features --------------------
# # # # expected_features = [
# # # #     'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
# # # #     'gender_Female', 'gender_Male', 'gender_Other',
# # # #     'ever_married_No', 'ever_married_Yes',
# # # #     'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
# # # #     'work_type_Self-employed', 'work_type_children',
# # # #     'Residence_type_Rural', 'Residence_type_Urban',
# # # #     'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
# # # # ]

# # # # # -------------------- Flask App --------------------
# # # # app = Flask(__name__)

# # # # @app.route("/")
# # # # def home():
# # # #     return render_template("index.html")

# # # # @app.route("/about")
# # # # def about():
# # # #     return render_template("about.html")

# # # # @app.route("/predict", methods=["POST"])
# # # # def predict():
# # # #     try:
# # # #         # Get form data
# # # #         input_data = request.form.to_dict()

# # # #         # Convert to dataframe
# # # #         df = pd.DataFrame([input_data])

# # # #         # Convert numeric values
# # # #         for col in ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']:
# # # #             if col in df.columns:
# # # #                 df[col] = pd.to_numeric(df[col], errors='coerce')
# # # #             else:
# # # #                 df[col] = 0

# # # #         # One-hot encoding for categorical values
# # # #         categorical_map = {
# # # #             "gender": ["Female", "Male", "Other"],
# # # #             "ever_married": ["No", "Yes"],
# # # #             "work_type": ["Govt_job", "Never_worked", "Private", "Self-employed", "children"],
# # # #             "Residence_type": ["Rural", "Urban"],
# # # #             "smoking_status": ["formerly smoked", "never smoked", "smokes"]
# # # #         }

# # # #         for field, categories in categorical_map.items():
# # # #             value = input_data.get(field, None)
# # # #             for cat in categories:
# # # #                 col_name = f"{field}_{cat}" if field != "Residence_type" else f"Residence_type_{cat}"
# # # #                 df[col_name] = 1 if value == cat else 0

# # # #         # Ensure all expected features exist
# # # #         for feature in expected_features:
# # # #             if feature not in df.columns:
# # # #                 df[feature] = 0

# # # #         # Reorder
# # # #         df = df[expected_features]

# # # #         # Impute + Scale
# # # #         df_imputed = imputer.transform(df)
# # # #         df_scaled = scaler.transform(df_imputed)

# # # #         # Predict
# # # #         prediction = model.predict(df_scaled)
# # # #         probability = float(prediction[0][0])
# # # #         result = "High Risk of Stroke" if probability >= 0.5 else "Low Risk of Stroke"

# # # #         return render_template("predict.html",
# # # #                                result=result,
# # # #                                probability=f"{probability:.2f}")

# # # #     except Exception as e:
# # # #         return jsonify({"error": str(e)})


# # # # if __name__ == "__main__":
# # # #     app.run(debug=True)


# # # from flask import Flask, render_template, request
# # # import numpy as np
# # # import joblib
# # # import tensorflow as tf

# # # app = Flask(__name__)

# # # # Load pre-trained files
# # # imputer = joblib.load("stroke_imputer.pkl")
# # # scaler = joblib.load("stroke_scaler.pkl")
# # # model = tf.keras.models.load_model("dnn5.keras")

# # # # Expected features
# # # expected_features = [
# # #     'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
# # #     'gender_Female', 'gender_Male', 'gender_Other',
# # #     'ever_married_No', 'ever_married_Yes',
# # #     'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
# # #     'work_type_Self-employed', 'work_type_children',
# # #     'Residence_type_Rural', 'Residence_type_Urban',
# # #     'smoking_status_formerly smoked', 'smoking_status_never smoked',
# # #     'smoking_status_smokes'
# # # ]

# # # @app.route("/")
# # # def index():
# # #     return render_template("index.html")

# # # @app.route("/about")
# # # def about():
# # #     return render_template("about.html")

# # # @app.route("/predict", methods=["GET", "POST"])
# # # def predict():
# # #     if request.method == "POST":
# # #         try:
# # #             # Collect input values
# # #             data = []
# # #             for feature in expected_features:
# # #                 val = request.form.get(feature, 0)
# # #                 data.append(float(val))

# # #             # Convert to numpy
# # #             input_data = np.array([data])

# # #             # Impute missing
# # #             input_data = imputer.transform(input_data)

# # #             # Scale
# # #             input_data = scaler.transform(input_data)

# # #             # Predict
# # #             prediction = model.predict(input_data)[0][0]
# # #             result = "Stroke Risk Detected" if prediction > 0.5 else "No Stroke Risk"

# # #             return render_template("predict.html", prediction=result)

# # #         except Exception as e:
# # #             return render_template("predict.html", prediction=f"Error: {str(e)}")

# # #     # For GET request, just show form
# # #     return render_template("predict.html", prediction=None)


# # # if __name__ == "__main__":
# # #     app.run(debug=True)

# # from flask import Flask, render_template, request
# # import joblib
# # import numpy as np
# # import tensorflow as tf

# # app = Flask(__name__)

# # # Load pre-trained objects
# # scaler = joblib.load("stroke_scaler.pkl")
# # imputer = joblib.load("stroke_imputer.pkl")
# # model = tf.keras.models.load_model("dnn5.keras")

# # # Home route
# # @app.route("/")
# # def index():
# #     return render_template("index.html")

# # # About route
# # @app.route("/about")
# # def about():
# #     return render_template("about.html")

# # # Predict form page
# # @app.route("/predict")
# # def predict_form():
# #     return render_template("predict.html")

# # # Handle form submission
# # @app.route("/predict", methods=["POST"])
# # def predict():
# #     try:
# #         # Collect form data
# #         age = float(request.form["age"])
# #         hypertension = int(request.form["hypertension"])
# #         heart_disease = int(request.form["heart_disease"])
# #         avg_glucose_level = float(request.form["avg_glucose_level"])
# #         bmi = float(request.form["bmi"])
# #         gender = request.form["gender"]
# #         ever_married = request.form["ever_married"]
# #         work_type = request.form["work_type"]
# #         residence_type = request.form["Residence_type"]
# #         smoking_status = request.form["smoking_status"]

# #         # One-hot encoding
# #         gender_map = {"Female": [1,0,0], "Male": [0,1,0], "Other": [0,0,1]}
# #         married_map = {"No": [1,0], "Yes": [0,1]}
# #         work_map = {"Govt_job":[1,0,0,0,0],"Never_worked":[0,1,0,0,0],
# #                     "Private":[0,0,1,0,0],"Self-employed":[0,0,0,1,0],
# #                     "children":[0,0,0,0,1]}
# #         residence_map = {"Rural":[1,0], "Urban":[0,1]}
# #         smoke_map = {"formerly smoked":[1,0,0], "never smoked":[0,1,0], "smokes":[0,0,1]}

# #         features = [
# #             age, hypertension, heart_disease, avg_glucose_level, bmi,
# #             *gender_map[gender],
# #             *married_map[ever_married],
# #             *work_map[work_type],
# #             *residence_map[residence_type],
# #             *smoke_map[smoking_status]
# #         ]

# #         # Convert to numpy array
# #         features = np.array(features).reshape(1, -1)

# #         # Impute missing
# #         features = imputer.transform(features)

# #         # Scale
# #         features = scaler.transform(features)

# #         # Predict
# #         prediction = model.predict(features)[0][0]
# #         result = "Stroke Risk" if prediction >= 0.5 else "No Stroke Risk"

# #         return render_template("predict.html", prediction_text=f"Prediction: {result} (Score: {prediction:.2f})")

# #     except Exception as e:
# #         return render_template("predict.html", prediction_text=f"Error: {str(e)}")

# # if __name__ == "__main__":
# #     app.run(debug=True)

# from flask import Flask, render_template, request
# import joblib
# import numpy as np
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # Load preprocessing objects and model
# imputer = joblib.load("stroke_imputer.pkl")
# scaler = joblib.load("stroke_scaler.pkl")
# model = load_model("dnn5.keras")

# # Home page
# @app.route('/')
# def index():
#     return render_template("index.html")

# # About page
# @app.route('/about')
# def about():
#     return render_template("about.html")

# # Predict page (form)
# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == "POST":
#         try:
#             # Collect data from form
#             age = float(request.form['age'])
#             hypertension = int(request.form['hypertension'])
#             heart_disease = int(request.form['heart_disease'])
#             avg_glucose_level = float(request.form['avg_glucose_level'])
#             bmi = float(request.form['bmi'])
#             gender = request.form['gender']
#             ever_married = request.form['ever_married']
#             work_type = request.form['work_type']
#             residence_type = request.form['residence_type']
#             smoking_status = request.form['smoking_status']

#             # One-hot encoding (order must match training)
#             features = [
#                 age, hypertension, heart_disease,
#                 avg_glucose_level, bmi,
#                 1 if gender == "Female" else 0,
#                 1 if gender == "Male" else 0,
#                 1 if gender == "Other" else 0,
#                 1 if ever_married == "No" else 0,
#                 1 if ever_married == "Yes" else 0,
#                 1 if work_type == "Govt_job" else 0,
#                 1 if work_type == "Never_worked" else 0,
#                 1 if work_type == "Private" else 0,
#                 1 if work_type == "Self-employed" else 0,
#                 1 if work_type == "children" else 0,
#                 1 if residence_type == "Rural" else 0,
#                 1 if residence_type == "Urban" else 0,
#                 1 if smoking_status == "formerly smoked" else 0,
#                 1 if smoking_status == "never smoked" else 0,
#                 1 if smoking_status == "smokes" else 0
#             ]

#             # Preprocess
#             features = imputer.transform([features])
#             features = scaler.transform(features)

#             # Predict
#             prediction = model.predict(features)[0][0]
#             result = "High Risk of Stroke" if prediction > 0.5 else "Low Risk of Stroke"

#             return render_template("predict.html", prediction_text=f"Prediction: {result}")

#         except Exception as e:
#             return render_template("predict.html", prediction_text=f"Error: {str(e)}")

#     return render_template("predict.html")


# if __name__ == "__main__":
#     app.run(debug=True)

# from flask import Flask, render_template, request
# import joblib
# import numpy as np
# import pandas as pd
# import tensorflow as tf

# app = Flask(__name__)

# # Load pre-trained objects
# scaler = joblib.load("stroke_scaler.pkl")
# imputer = joblib.load("stroke_imputer.pkl")
# model = tf.keras.models.load_model("dnn5.keras")


# # Home route
# @app.route("/")
# def index():
#     return render_template("index.html")


# # About route
# @app.route("/about")
# def about():
#     return render_template("about.html")

# @app.route("/risk")
# def risk():
#     return render_template("risk.html")

# # Show prediction form (GET)
# @app.route("/predict", methods=["GET"])
# def predict_form():
#     return render_template("predict.html")


# # Handle form submission (POST)
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Get form inputs
#         data = request.form

#         # Map categorical selections to one-hot encoded format
#         input_data = {
#             'age': float(data['age']),
#             'hypertension': int(data['hypertension']),
#             'heart_disease': int(data['heart_disease']),
#             'avg_glucose_level': float(data['avg_glucose_level']),
#             'bmi': float(data['bmi']),
#             'gender_Female': 1 if data['gender'] == 'gender_Female' else 0,
#             'gender_Male': 1 if data['gender'] == 'gender_Male' else 0,
#             'gender_Other': 1 if data['gender'] == 'gender_Other' else 0,
#             'ever_married_No': 1 if data['ever_married'] == 'ever_married_No' else 0,
#             'ever_married_Yes': 1 if data['ever_married'] == 'ever_married_Yes' else 0,
#             'work_type_Govt_job': 1 if data['work_type'] == 'work_type_Govt_job' else 0,
#             'work_type_Never_worked': 1 if data['work_type'] == 'work_type_Never_worked' else 0,
#             'work_type_Private': 1 if data['work_type'] == 'work_type_Private' else 0,
#             'work_type_Self-employed': 1 if data['work_type'] == 'work_type_Self-employed' else 0,
#             'work_type_children': 1 if data['work_type'] == 'work_type_children' else 0,
#             'Residence_type_Rural': 1 if data['Residence_type'] == 'Residence_type_Rural' else 0,
#             'Residence_type_Urban': 1 if data['Residence_type'] == 'Residence_type_Urban' else 0,
#             'smoking_status_formerly smoked': 1 if data['smoking_status'] == 'smoking_status_formerly smoked' else 0,
#             'smoking_status_never smoked': 1 if data['smoking_status'] == 'smoking_status_never smoked' else 0,
#             'smoking_status_smokes': 1 if data['smoking_status'] == 'smoking_status_smokes' else 0,
#         }

#         # Convert to DataFrame
#         df = pd.DataFrame([input_data])

#         # Impute + Scale
#         df_imputed = imputer.transform(df)
#         df_scaled = scaler.transform(df_imputed)

#         # Predict
#         prediction = model.predict(df_scaled)[0][0]
#         result = "⚠️ Stroke Risk Detected" if prediction > 0.5 else "✅ No Stroke Risk"

#         return render_template("predict.html", prediction_text=f"{result} (Score: {prediction:.2f})")

#     except Exception as e:
#         return render_template("predict.html", prediction_text=f"Error: {str(e)}")


# if __name__ == "__main__":
#     app.run(debug=True)

# from flask import Flask, render_template, request
# import joblib
# import numpy as np
# import pandas as pd
# import tensorflow as tf


# app = Flask(__name__)

# # Load pre-trained objects
# scaler = joblib.load("stroke_scaler.pkl")
# imputer = joblib.load("stroke_imputer.pkl")
# model = tf.keras.models.load_model("dnn5.keras")

# # One-hot encoded columns used during training
# final_columns = [
#     'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
#     'gender_Female', 'gender_Male', 'gender_Other',
#     'ever_married_No', 'ever_married_Yes',
#     'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
#     'work_type_Self-employed', 'work_type_children',
#     'Residence_type_Rural', 'Residence_type_Urban',
#     'smoking_status_formerly smoked', 'smoking_status_never smoked',
#     'smoking_status_smokes'
# ]


# # Home route
# @app.route("/")
# def index():
#     return render_template("index.html")


# # About route
# @app.route("/about")
# def about():
#     return render_template("about.html")


# @app.route("/risk")
# def risk():
#     return render_template("risk.html")


# # ---------------- SINGLE PREDICTION (already working) -----------------
# @app.route("/predict", methods=["GET"])
# def predict_form():
#     return render_template("predict.html")


# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.form

#         input_data = {
#             'age': float(data['age']),
#             'hypertension': int(data['hypertension']),
#             'heart_disease': int(data['heart_disease']),
#             'avg_glucose_level': float(data['avg_glucose_level']),
#             'bmi': float(data['bmi']),
#             'gender_Female': 1 if data['gender'] == 'gender_Female' else 0,
#             'gender_Male': 1 if data['gender'] == 'gender_Male' else 0,
#             'gender_Other': 1 if data['gender'] == 'gender_Other' else 0,
#             'ever_married_No': 1 if data['ever_married'] == 'ever_married_No' else 0,
#             'ever_married_Yes': 1 if data['ever_married'] == 'ever_married_Yes' else 0,
#             'work_type_Govt_job': 1 if data['work_type'] == 'work_type_Govt_job' else 0,
#             'work_type_Never_worked': 1 if data['work_type'] == 'work_type_Never_worked' else 0,
#             'work_type_Private': 1 if data['work_type'] == 'work_type_Private' else 0,
#             'work_type_Self-employed': 1 if data['work_type'] == 'work_type_Self-employed' else 0,
#             'work_type_children': 1 if data['work_type'] == 'work_type_children' else 0,
#             'Residence_type_Rural': 1 if data['Residence_type'] == 'Residence_type_Rural' else 0,
#             'Residence_type_Urban': 1 if data['Residence_type'] == 'Residence_type_Urban' else 0,
#             'smoking_status_formerly smoked': 1 if data['smoking_status'] == 'smoking_status_formerly smoked' else 0,
#             'smoking_status_never smoked': 1 if data['smoking_status'] == 'smoking_status_never smoked' else 0,
#             'smoking_status_smokes': 1 if data['smoking_status'] == 'smoking_status_smokes' else 0,
#         }

#         df = pd.DataFrame([input_data])

#         df_imputed = imputer.transform(df)
#         df_scaled = scaler.transform(df_imputed)

#         prediction = model.predict(df_scaled)[0][0]
#         result = "⚠️ Stroke Risk Detected" if prediction > 0.5 else "✅ No Stroke Risk"

#         return render_template("predict.html", prediction_text=f"{result} (Score: {prediction:.2f})")

#     except Exception as e:
#         return render_template("predict.html", prediction_text=f"Error: {str(e)}")


# # ---------------- EXCEL UPLOAD PREDICTION (NEW FEATURE) -----------------

# @app.route("/excel_predict", methods=["GET"])
# def excel_form():
#     return render_template("excel_predict.html")


# @app.route("/excel_predict", methods=["POST"])
# def excel_predict():
#     try:
#         file = request.files["file"]
#         df = pd.read_excel(file)

#         # Required raw columns
#         needed_cols = [
#             "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
#             "gender", "ever_married", "work_type",
#             "Residence_type", "smoking_status"
#         ]

#         missing = [c for c in needed_cols if c not in df.columns]
#         if missing:
#             return render_template("excel_predict.html",
#                                    results=[{"Error": f"Missing columns: {missing}"}])

#         # ---------------- Convert RAW columns → One-Hot ----------------
#         df_encoded = pd.get_dummies(df, columns=[
#             "gender", "ever_married", "work_type", "Residence_type", "smoking_status"
#         ])

#         # Ensure all required training columns exist
#         for col in final_columns:
#             if col not in df_encoded.columns:
#                 df_encoded[col] = 0

#         # Order columns properly
#         df_encoded = df_encoded[final_columns]

#         # ---------------- Impute & Scale ----------------
#         df_imputed = imputer.transform(df_encoded)
#         df_scaled = scaler.transform(df_imputed)

#         # ---------------- Predict ----------------
#         preds = model.predict(df_scaled).flatten()

#         df["Prediction"] = ["Stroke" if p > 0.5 else "No Stroke" for p in preds]
#         df["Probability"] = preds.round(3)

#         results = df.to_dict(orient="records")

#         return render_template("excel_predict.html", results=results)

#     except Exception as e:
#         return render_template("excel_predict.html", results=[{"Error": str(e)}])


# # ---------------- MAIN ----------------
# if __name__ == "__main__":
#     app.run(debug=True)                                MAIN MAIN MAIN


# from flask import Flask, render_template, request
# import joblib
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import plotly.express as px
# import plotly.io as pio

# app = Flask(__name__)

# # Load pre-trained objects
# scaler = joblib.load("stroke_scaler.pkl")
# imputer = joblib.load("stroke_imputer.pkl")
# model = tf.keras.models.load_model("dnn5.keras")

# # One-hot encoded columns used during training
# final_columns = [
#     'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
#     'gender_Female', 'gender_Male', 'gender_Other',
#     'ever_married_No', 'ever_married_Yes',
#     'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
#     'work_type_Self-employed', 'work_type_children',
#     'Residence_type_Rural', 'Residence_type_Urban',
#     'smoking_status_formerly smoked', 'smoking_status_never smoked',
#     'smoking_status_smokes'
# ]

# # Home route
# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/about")
# def about():
#     return render_template("about.html")

# @app.route("/risk")
# def risk():
#     return render_template("risk.html")

# # ---------------- SINGLE PREDICTION -----------------
# @app.route("/predict", methods=["GET"])
# def predict_form():
#     return render_template("predict.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.form

#         input_data = {
#             'age': float(data['age']),
#             'hypertension': int(data['hypertension']),
#             'heart_disease': int(data['heart_disease']),
#             'avg_glucose_level': float(data['avg_glucose_level']),
#             'bmi': float(data['bmi']),
#             'gender_Female': 1 if data['gender'] == 'gender_Female' else 0,
#             'gender_Male': 1 if data['gender'] == 'gender_Male' else 0,
#             'gender_Other': 1 if data['gender'] == 'gender_Other' else 0,
#             'ever_married_No': 1 if data['ever_married'] == 'ever_married_No' else 0,
#             'ever_married_Yes': 1 if data['ever_married'] == 'ever_married_Yes' else 0,
#             'work_type_Govt_job': 1 if data['work_type'] == 'work_type_Govt_job' else 0,
#             'work_type_Never_worked': 1 if data['work_type'] == 'work_type_Never_worked' else 0,
#             'work_type_Private': 1 if data['work_type'] == 'work_type_Private' else 0,
#             'work_type_Self-employed': 1 if data['work_type'] == 'work_type_Self-employed' else 0,
#             'work_type_children': 1 if data['work_type'] == 'work_type_children' else 0,
#             'Residence_type_Rural': 1 if data['Residence_type'] == 'Residence_type_Rural' else 0,
#             'Residence_type_Urban': 1 if data['Residence_type'] == 'Residence_type_Urban' else 0,
#             'smoking_status_formerly smoked': 1 if data['smoking_status'] == 'smoking_status_formerly smoked' else 0,
#             'smoking_status_never smoked': 1 if data['smoking_status'] == 'smoking_status_never smoked' else 0,
#             'smoking_status_smokes': 1 if data['smoking_status'] == 'smoking_status_smokes' else 0,
#         }

#         df = pd.DataFrame([input_data])

#         df_imputed = imputer.transform(df)
#         df_scaled = scaler.transform(df_imputed)

#         prediction = model.predict(df_scaled)[0][0]
#         result = "⚠️ Stroke Risk Detected" if prediction > 0.5 else "✅ No Stroke Risk"

#         return render_template("predict.html", prediction_text=f"{result} (Score: {prediction:.2f})")

#     except Exception as e:
#         return render_template("predict.html", prediction_text=f"Error: {str(e)}")

# # ---------------- EXCEL UPLOAD PREDICTION + EDA -----------------
# @app.route("/excel_predict", methods=["GET"])
# def excel_form():
#     return render_template("excel_predict.html")

# @app.route("/excel_predict", methods=["POST"])
# def excel_predict():
#     try:
#         file = request.files["file"]
#         df = pd.read_excel(file)

#         # Required raw columns
#         needed_cols = [
#             "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
#             "gender", "ever_married", "work_type",
#             "Residence_type", "smoking_status"
#         ]

#         missing = [c for c in needed_cols if c not in df.columns]
#         if missing:
#             return render_template("excel_predict.html",
#                                    results=[{"Error": f"Missing columns: {missing}"}])

#         # ---------------- Convert RAW columns → One-Hot ----------------
#         df_encoded = pd.get_dummies(df, columns=[
#             "gender", "ever_married", "work_type", "Residence_type", "smoking_status"
#         ])

#         # Ensure all required training columns exist
#         for col in final_columns:
#             if col not in df_encoded.columns:
#                 df_encoded[col] = 0

#         # Order columns properly
#         df_encoded = df_encoded[final_columns]

#         # ---------------- Impute & Scale ----------------
#         df_imputed = imputer.transform(df_encoded)
#         df_scaled = scaler.transform(df_imputed)

#         # ---------------- Predict ----------------
#         preds = model.predict(df_scaled).flatten()

#         df["Prediction"] = ["Stroke" if p > 0.5 else "No Stroke" for p in preds]
#         df["Probability"] = preds.round(3)

#         results = df.to_dict(orient="records")

#         # ---------------- EDA: Gender Distribution ----------------
#         if "gender" in df.columns:
#             gender_counts = df["gender"].value_counts()
#             color_sequence = ['#1f77b4', '#ff7f0e', '#2ca02c']
#             fig = px.pie(
#                 names=gender_counts.index,
#                 values=gender_counts.values,
#                 title="Gender Distribution",
#                 color=gender_counts.index,  # Assign colors based on category
#                 color_discrete_sequence=color_sequence
#             )
#             graph_html = pio.to_html(fig, full_html=False)
#         else:
#             graph_html = "<p>No gender column found for EDA.</p>"

#         return render_template("excel_predict.html", results=results, graph_html=graph_html)

#     except Exception as e:
#         return render_template("excel_predict.html", results=[{"Error": str(e)}], graph_html="")

# # ---------------- MAIN ----------------
# if __name__ == "__main__":
#     app.run(debug=True)




# from flask import Flask, render_template, request
# import joblib
# import pandas as pd
# import tensorflow as tf
# import plotly.express as px
# import plotly.io as pio

# app = Flask(__name__)

# # Load pre-trained objects
# scaler = joblib.load("stroke_scaler.pkl")
# imputer = joblib.load("stroke_imputer.pkl")
# model = tf.keras.models.load_model("dnn5.keras")

# # One-hot encoded columns used during training
# final_columns = [
#     'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
#     'gender_Female', 'gender_Male', 'gender_Other',
#     'ever_married_No', 'ever_married_Yes',
#     'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
#     'work_type_Self-employed', 'work_type_children',
#     'Residence_type_Rural', 'Residence_type_Urban',
#     'smoking_status_formerly smoked', 'smoking_status_never smoked',
#     'smoking_status_smokes'
# ]

# # Home route
# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/about")
# def about():
#     return render_template("about.html")

# @app.route("/risk")
# def risk():
#     return render_template("risk.html")

# # Single prediction routes omitted for brevity...

# # ---------------- EXCEL UPLOAD PREDICTION + EDA -----------------
# @app.route("/excel_predict", methods=["GET"])
# def excel_form():
#     return render_template("excel_predict.html")

# @app.route("/excel_predict", methods=["POST"])
# def excel_predict():
#     try:
#         file = request.files["file"]
#         df = pd.read_excel(file)

#         # Required raw columns
#         needed_cols = [
#             "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
#             "gender", "ever_married", "work_type",
#             "Residence_type", "smoking_status"
#         ]

#         missing = [c for c in needed_cols if c not in df.columns]
#         if missing:
#             return render_template("excel_predict.html",
#                                    results=[{"Error": f"Missing columns: {missing}"}])

#         # Convert RAW columns → One-Hot
#         df_encoded = pd.get_dummies(df, columns=[
#             "gender", "ever_married", "work_type", "Residence_type", "smoking_status"
#         ])

#         # Ensure all required training columns exist
#         for col in final_columns:
#             if col not in df_encoded.columns:
#                 df_encoded[col] = 0

#         # Order columns properly
#         df_encoded = df_encoded[final_columns]

#         # Impute & Scale
#         df_imputed = imputer.transform(df_encoded)
#         df_scaled = scaler.transform(df_imputed)

#         # Predict
#         preds = model.predict(df_scaled).flatten()
#         df["Prediction"] = ["Stroke" if p > 0.5 else "No Stroke" for p in preds]
#         df["Probability"] = preds.round(3)

#         results = df.to_dict(orient="records")

#         # ---------------- EDA: Pie Charts ----------------
#         graph_htmls = []

#         # Function to generate pie chart HTML
#         def generate_pie_chart(data, title, colors=None):
#             counts = data.value_counts()
#             fig = px.pie(
#                 names=counts.index,
#                 values=counts.values,
#                 title=title,
#                 color=counts.index,
#                 color_discrete_sequence=colors or px.colors.qualitative.Safe
#             )
#             return pio.to_html(fig, full_html=False)

#         # Gender chart
#         if "gender" in df.columns:
#             graph_htmls.append(generate_pie_chart(df["gender"], "Gender Distribution", ['#1f77b4', '#ff7f0e', '#2ca02c']))

#         # Hypertension chart
#         if "hypertension" in df.columns:
#             graph_htmls.append(generate_pie_chart(df["hypertension"], "Hypertension Distribution", ['#ff7f0e', '#2ca02c']))

#         # Heart Disease chart
#         if "heart_disease" in df.columns:
#             graph_htmls.append(generate_pie_chart(df["heart_disease"], "Heart Disease Distribution", ['#ff7f0e', '#2ca02c']))

#         # Smoking Status chart
#         if "smoking_status" in df.columns:
#             graph_htmls.append(generate_pie_chart(df["smoking_status"], "Smoking Status Distribution", ['#8c564b', '#e377c2', '#7f7f7f']))

#         return render_template("excel_predict.html", results=results, graph_htmls=graph_htmls)

#     except Exception as e:
#         return render_template("excel_predict.html", results=[{"Error": str(e)}], graph_htmls=[])

# # Main
# if __name__ == "__main__":
#     app.run(debug=True)



# from flask import Flask, render_template, request
# import joblib
# import pandas as pd
# import tensorflow as tf
# import plotly.express as px
# import plotly.io as pio

# app = Flask(__name__)

# # Load pre-trained objects
# scaler = joblib.load("stroke_scaler.pkl")
# imputer = joblib.load("stroke_imputer.pkl")
# model = tf.keras.models.load_model("dnn5.keras")

# # One-hot encoded columns used during training
# final_columns = [
#     'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
#     'gender_Female', 'gender_Male', 'gender_Other',
#     'ever_married_No', 'ever_married_Yes',
#     'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
#     'work_type_Self-employed', 'work_type_children',
#     'Residence_type_Rural', 'Residence_type_Urban',
#     'smoking_status_formerly smoked', 'smoking_status_never smoked',
#     'smoking_status_smokes'
# ]



# # Home route
# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/about")
# def about():
#     return render_template("about.html")

# @app.route("/risk")
# def risk():
#     return render_template("risk.html")
# @app.route("/predict")
# def predict():
#     return render_template("predict.html")

# @app.route("/predict", methods=["POST"])
#  def predict():
#     try:
#         data = request.form

#         input_data = {
#             'age': float(data['age']),
#             'hypertension': int(data['hypertension']),
#             'heart_disease': int(data['heart_disease']),
#             'avg_glucose_level': float(data['avg_glucose_level']),
#             'bmi': float(data['bmi']),
#             'gender_Female': 1 if data['gender'] == 'gender_Female' else 0,
#             'gender_Male': 1 if data['gender'] == 'gender_Male' else 0,
#             'gender_Other': 1 if data['gender'] == 'gender_Other' else 0,
#             'ever_married_No': 1 if data['ever_married'] == 'ever_married_No' else 0,
#             'ever_married_Yes': 1 if data['ever_married'] == 'ever_married_Yes' else 0,
#             'work_type_Govt_job': 1 if data['work_type'] == 'work_type_Govt_job' else 0,
#             'work_type_Never_worked': 1 if data['work_type'] == 'work_type_Never_worked' else 0,
#             'work_type_Private': 1 if data['work_type'] == 'work_type_Private' else 0,
#             'work_type_Self-employed': 1 if data['work_type'] == 'work_type_Self-employed' else 0,
#             'work_type_children': 1 if data['work_type'] == 'work_type_children' else 0,
#             'Residence_type_Rural': 1 if data['Residence_type'] == 'Residence_type_Rural' else 0,
#             'Residence_type_Urban': 1 if data['Residence_type'] == 'Residence_type_Urban' else 0,
#             'smoking_status_formerly smoked': 1 if data['smoking_status'] == 'smoking_status_formerly smoked' else 0,
#             'smoking_status_never smoked': 1 if data['smoking_status'] == 'smoking_status_never smoked' else 0,
#             'smoking_status_smokes': 1 if data['smoking_status'] == 'smoking_status_smokes' else 0,
#         }

#         df = pd.DataFrame([input_data])

#         df_imputed = imputer.transform(df)
#         df_scaled = scaler.transform(df_imputed)

#         prediction = model.predict(df_scaled)[0][0]
#         result = "⚠️ Stroke Risk Detected" if prediction > 0.5 else "✅ No Stroke Risk"

#         return render_template("predict.html", prediction_text=f"{result} (Score: {prediction:.2f})")

#     except Exception as e:
#         return render_template("predict.html", prediction_text=f"Error: {str(e)}")




# # ---------------- EXCEL UPLOAD PREDICTION + EDA -----------------
# @app.route("/excel_predict", methods=["GET"])
# def excel_form():
#     return render_template("excel_predict.html")

# @app.route("/excel_predict", methods=["POST"])
# def excel_predict():
#     try:
#         file = request.files["file"]
#         df = pd.read_excel(file)

#         # Required raw columns
#         needed_cols = [
#             "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
#             "gender", "ever_married", "work_type",
#             "Residence_type", "smoking_status"
#         ]

#         missing = [c for c in needed_cols if c not in df.columns]
#         if missing:
#             return render_template("excel_predict.html",
#                                    results=[{"Error": f"Missing columns: {missing}"}])

#         # Convert RAW columns → One-Hot
#         df_encoded = pd.get_dummies(df, columns=[
#             "gender", "ever_married", "work_type", "Residence_type", "smoking_status"
#         ])

#         # Ensure all required training columns exist
#         for col in final_columns:
#             if col not in df_encoded.columns:
#                 df_encoded[col] = 0

#         # Order columns properly
#         df_encoded = df_encoded[final_columns]

#         # Impute & Scale
#         df_imputed = imputer.transform(df_encoded)
#         df_scaled = scaler.transform(df_imputed)

#         # Predict
#         preds = model.predict(df_scaled).flatten()
#         df["Prediction"] = ["Stroke" if p > 0.5 else "No Stroke" for p in preds]
#         df["Probability"] = preds.round(3)

#         results = df.to_dict(orient="records")

#         # ---------------- EDA: Charts ----------------
#         graph_htmls = []

#         def pie_chart(data, title, colors=None):
#             counts = data.value_counts()
#             fig = px.pie(
#                 names=counts.index,
#                 values=counts.values,
#                 title=title,
#                 color=counts.index,
#                 color_discrete_sequence=colors or px.colors.qualitative.Safe
#             )
#             return pio.to_html(fig, full_html=False)

#         def histogram_chart(data, title):
#             fig = px.histogram(data, x=data, nbins=20, title=title, color_discrete_sequence=['#2e86de'])
#             return pio.to_html(fig, full_html=False)

#         # Gender chart
#         if "gender" in df.columns:
#             graph_htmls.append(pie_chart(df["gender"], "Gender Distribution", ['#1f77b4', '#ff7f0e', '#2ca02c']))

#         # Hypertension chart
#         if "hypertension" in df.columns:
#             graph_htmls.append(pie_chart(df["hypertension"], "Hypertension Distribution", ['#ff7f0e', '#2ca02c']))

#         # Heart Disease chart
#         if "heart_disease" in df.columns:
#             graph_htmls.append(pie_chart(df["heart_disease"], "Heart Disease Distribution", ['#d62728', '#9467bd']))

#         # Smoking Status chart
#         if "smoking_status" in df.columns:
#             graph_htmls.append(pie_chart(df["smoking_status"], "Smoking Status Distribution", ['#8c564b', '#e377c2', '#7f7f7f']))

#         # Stroke vs Non-Stroke chart
#         graph_htmls.append(pie_chart(df["Prediction"], "Stroke vs Non-Stroke", ['#ff7f0e', '#2ca02c']))

#         # BMI histogram
#         if "bmi" in df.columns:
#             graph_htmls.append(histogram_chart(df["bmi"], "BMI Distribution"))

#         # Average Glucose histogram
#         if "avg_glucose_level" in df.columns:
#             graph_htmls.append(histogram_chart(df["avg_glucose_level"], "Average Glucose Level Distribution"))

#         return render_template("excel_predict.html", results=results, graph_htmls=graph_htmls)

#     except Exception as e:
#         return render_template("excel_predict.html", results=[{"Error": str(e)}], graph_htmls=[])

# # Main
# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, render_template, request
import joblib
import os
import pandas as pd
import tensorflow as tf
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

# ------------------ Load pre-trained objects ------------------
scaler = joblib.load("stroke_scaler.pkl")
imputer = joblib.load("stroke_imputer.pkl")
model = tf.keras.models.load_model("dnn5.keras")

# One-hot encoded columns used during training
final_columns = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'gender_Female', 'gender_Male', 'gender_Other',
    'ever_married_No', 'ever_married_Yes',
    'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
    'work_type_Self-employed', 'work_type_children',
    'Residence_type_Rural', 'Residence_type_Urban',
    'smoking_status_formerly smoked', 'smoking_status_never smoked',
    'smoking_status_smokes'
]

# ------------------ PAGE ROUTES ------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/risk")
def risk():
    return render_template("risk.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")


# ------------------ FIXED PREDICT POST ROUTE ------------------
@app.route("/predict", methods=["POST"])
def predict_post():
    try:
        data = request.form

        input_data = {
            'age': float(data['age']),
            'hypertension': int(data['hypertension']),
            'heart_disease': int(data['heart_disease']),
            'avg_glucose_level': float(data['avg_glucose_level']),
            'bmi': float(data['bmi']),
            'gender_Female': 1 if data['gender'] == 'gender_Female' else 0,
            'gender_Male': 1 if data['gender'] == 'gender_Male' else 0,
            'gender_Other': 1 if data['gender'] == 'gender_Other' else 0,
            'ever_married_No': 1 if data['ever_married'] == 'ever_married_No' else 0,
            'ever_married_Yes': 1 if data['ever_married'] == 'ever_married_Yes' else 0,
            'work_type_Govt_job': 1 if data['work_type'] == 'work_type_Govt_job' else 0,
            'work_type_Never_worked': 1 if data['work_type'] == 'work_type_Never_worked' else 0,
            'work_type_Private': 1 if data['work_type'] == 'work_type_Private' else 0,
            'work_type_Self-employed': 1 if data['work_type'] == 'work_type_Self-employed' else 0,
            'work_type_children': 1 if data['work_type'] == 'work_type_children' else 0,
            'Residence_type_Rural': 1 if data['Residence_type'] == 'Residence_type_Rural' else 0,
            'Residence_type_Urban': 1 if data['Residence_type'] == 'Residence_type_Urban' else 0,
            'smoking_status_formerly smoked': 1 if data['smoking_status'] == 'smoking_status_formerly smoked' else 0,
            'smoking_status_never smoked': 1 if data['smoking_status'] == 'smoking_status_never smoked' else 0,
            'smoking_status_smokes': 1 if data['smoking_status'] == 'smoking_status_smokes' else 0,
        }

        df = pd.DataFrame([input_data])

        df_imputed = imputer.transform(df)
        df_scaled = scaler.transform(df_imputed)

        prediction = model.predict(df_scaled)[0][0]
        result = "⚠️ Stroke Risk Detected" if prediction > 0.5 else "✅ No Stroke Risk"

        return render_template("predict.html",
                               prediction_text=f"{result} (Score: {prediction:.2f})")
    except Exception as e:
        return render_template("predict.html",
                               prediction_text=f"Error: {str(e)}")


# ------------------ EXCEL UPLOAD + EDA ------------------
@app.route("/excel_predict", methods=["GET"])
def excel_form():
    return render_template("excel_predict.html")


@app.route("/excel_predict", methods=["POST"])
def excel_predict():
    try:
        file = request.files["file"]
        df = pd.read_excel(file)

        needed_cols = [
            "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
            "gender", "ever_married", "work_type",
            "Residence_type", "smoking_status"
        ]

        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            return render_template("excel_predict.html",
                                   results=[{"Error": f"Missing columns: {missing}"}])

        df_encoded = pd.get_dummies(df, columns=[
            "gender", "ever_married", "work_type", "Residence_type", "smoking_status"
        ])

        for col in final_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded[final_columns]

        df_imputed = imputer.transform(df_encoded)
        df_scaled = scaler.transform(df_imputed)

        preds = model.predict(df_scaled).flatten()
        df["Prediction"] = ["Stroke" if p > 0.5 else "No Stroke" for p in preds]
        df["Probability"] = preds.round(3)

        results = df.to_dict(orient="records")

        # --------- EDA CHARTS ---------
        graph_htmls = []

        def pie_chart(data, title, colors=None):
            counts = data.value_counts()
            fig = px.pie(names=counts.index, values=counts.values,
                         title=title,
                         color=counts.index,
                         color_discrete_sequence=colors or px.colors.qualitative.Safe)
            return pio.to_html(fig, full_html=False)

        def histogram_chart(data, title):
            fig = px.histogram(data, x=data, nbins=20,
                               title=title,
                               color_discrete_sequence=['#2e86de'])
            return pio.to_html(fig, full_html=False)

        # Charts
        if "gender" in df.columns:
            graph_htmls.append(pie_chart(df["gender"], "Gender Distribution"))

        if "hypertension" in df.columns:
            graph_htmls.append(pie_chart(df["hypertension"], "Hypertension Distribution"))

        if "heart_disease" in df.columns:
            graph_htmls.append(pie_chart(df["heart_disease"], "Heart Disease Distribution"))

        if "smoking_status" in df.columns:
            graph_htmls.append(pie_chart(df["smoking_status"], "Smoking Status Distribution"))

        graph_htmls.append(pie_chart(df["Prediction"], "Stroke vs Non-Stroke"))

        if "bmi" in df.columns:
            graph_htmls.append(histogram_chart(df["bmi"], "BMI Distribution"))

        if "avg_glucose_level" in df.columns:
            graph_htmls.append(histogram_chart(df["avg_glucose_level"], "Glucose Level Distribution"))

        return render_template("excel_predict.html",
                               results=results,
                               graph_htmls=graph_htmls)

    except Exception as e:
        return render_template("excel_predict.html",
                               results=[{"Error": str(e)}],
                               graph_htmls=[])


# ------------------ RUN FLASK APP ------------------
if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

