
# BG11 – Deep Stroke Detect: Optimizing Stroke Risk Prediction Using SMOTEENN and Transfer Learning


## Team Info
- 22471A05D3 — **Nazeema** ( [LinkedIn](https://www.linkedin.com/in/nazeema-shaik-9549b236b/) )
_Work Done: Led the project planning and system design, handled data preprocessing and feature engineering, implemented SMOTEENN for class imbalance handling, developed and trained deep learning and transfer learning models, integrated the prediction pipeline, and ensured overall system accuracy and reliability.

- 22471A05B4 — **Akhila** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: Performed exploratory data analysis (EDA), managed dataset cleaning and normalization, prepared training and testing datasets, evaluated model performance using Precision, Recall, F1-score, and ROC-AUC metrics, and analyzed results for medical relevance.

- 22471A0586 — **Sandhya** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: Set up the Python and deep learning environment, assisted in implementing transfer learning models, tuned hyperparameters, developed the user interface for stroke risk prediction, integrated the trained model with the application, and conducted system testing and documentation.



---

## Abstract
Stroke remains a major global health concern, being
one of the top causes of mortality and a leading contributor to
long-term disability. This is the very reason why recognizing the
signs of risk is critical to treatment—timing is everything. In
this paper, we present a novel approach to assess the risk of
stroke using deep learning. The system is applied to real-world
healthcare datasets, which often exhibit a greater prevalence
of non-stroke cases compared to stroke cases. To address the
imbalance and enhance the data quality, we implemented several
data preparation steps, SMOTEENN, and a cross-validated deep
neural network model. We also applied transfer learning to
mitigate the impact of scarce data on model performance.
The model’s accuracy was assessed using the common metrics
of accuracy, precision, recall, F1-score, and ROC-AUC. The
outcomes were quite encouraging, achieving 95.24% accuracy
on average and 95.52% F1-score, confirming that the model is
extremely accurate. Apart from the outcomes, this study attempts
to solve significant real-world challenges such as the sparse nature
of stroke cases, data sparsity, and selection of relevant health
indicators. The findings indicate that deep learning approaches
are very effective, provided there is thorough data preparation
along with intelligent preprocessing.Feature selection can be an
impactful method for predicting stroke risk— potentially helping
doctors take action earlier and save lives.

---

## Paper Reference (Inspiration)
👉 **[https://ieeexplore.ieee.org/document/10599507/]**
Stroke Prediction Using Deep Learning and
Transfer Learning Approaches  author:Ting-Wei Wu

---

## Our Improvement Over Existing Paper

- Implemented **SMOTEENN** technique to handle class imbalance and improve stroke case prediction.
- Applied **transfer learning** to increase model accuracy and reduce training time.
- Performed improved **data cleaning and feature engineering** for more reliable results.
- Evaluated the model using multiple metrics such as **Precision, Recall, F1-score, and ROC-AUC** instead of only accuracy.
- Developed a complete **end-to-end stroke prediction system** for practical usage.
- Achieved **higher accuracy and better minority class (stroke) detection performance** compared to the existing approach.


---

## About the Project

This project focuses on predicting the risk of stroke using machine learning and deep learning techniques. It takes patient health information such as age, gender, hypertension, heart disease, BMI, glucose level, and lifestyle factors as input and predicts whether a person is at high or low risk of stroke.

The system is useful for early stroke risk detection, helping doctors and healthcare professionals take preventive actions before a stroke occurs. It improves medical decision support by providing accurate and reliable predictions.

### Project Workflow

1. **Input:** Patient health data (age, blood pressure, BMI, glucose level, etc.)  
2. **Processing:** Data cleaning, normalization, and handling class imbalance using SMOTEENN  
3. **Model:** Deep learning and transfer learning models are trained on the processed dataset  
4. **Output:** Stroke risk prediction (High Risk / Low Risk) along with performance evaluation metrics

---

## Dataset Used
👉 **[Cerebral Stroke Prediction – Imbalanced Dataset (Kaggle)](https://www.kaggle.com/datasets/shashwatwork/cerebral-stroke-predictionimbalaced-dataset)**

**Dataset Details:**
- This dataset contains medical and lifestyle information of patients for predicting stroke risk.
- It includes features such as age, gender, hypertension, heart disease, BMI, average glucose level, smoking status, and work type.
- The target column is **stroke**, where 0 indicates no stroke and 1 indicates stroke.
- The dataset is highly imbalanced with fewer stroke cases compared to non-stroke cases.
- Data preprocessing and class balancing were performed using the **SMOTEENN** technique.
- The dataset is used to train and evaluate deep learning and transfer learning models for early stroke risk prediction.


---

## Dependencies Used

Python, NumPy, Pandas, Scikit-learn, TensorFlow/Keras, Imbalanced-learn (SMOTEENN), Matplotlib, Seaborn, Flask


## EDA & Preprocessing

- Performed Exploratory Data Analysis (EDA) to understand data distribution and detect missing values and outliers.
- Handled missing values using appropriate imputation techniques.
- Converted categorical features into numerical values using label encoding and one-hot encoding.
- Normalized and scaled numerical features for better model performance.
- Addressed class imbalance using the **SMOTEENN** technique.
- Split the dataset into training and testing sets for model evaluation.

## Model Training Info

- Implemented deep learning and transfer learning models for stroke risk prediction.
- Trained the models on the preprocessed and balanced dataset using the SMOTEENN technique.
- Used an 80:20 split for training and testing the data.
- Applied hyperparameter tuning to improve model performance.
- Evaluated the trained models using Precision, Recall, F1-score, Accuracy, and ROC-AUC metrics.
- Selected the best-performing model for final stroke risk prediction.


## Model Testing / Evaluation

- Tested the trained model on unseen test data to measure its performance.
- Used evaluation metrics such as Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Analyzed confusion matrix to understand correct and incorrect predictions.
- Compared results with the existing paper to verify improvements.
- Ensured better detection of minority class (stroke cases).
- Validated the reliability of the model for early stroke risk prediction.


## Results

- The proposed model achieved high accuracy and improved performance compared to the existing approach.
- SMOTEENN significantly enhanced the detection of stroke (minority class) cases.
- Transfer learning improved model generalization and reduced training time.
- The model showed better Precision, Recall, and F1-score values.
- ROC-AUC score indicated strong classification capability.
- The system proved effective for early and reliable stroke risk prediction.


## Limitations & Future Work

### Limitations
- The model is trained on a limited dataset, which may affect generalization to different populations.
- The dataset contains imbalanced classes, which can still introduce bias despite using SMOTEENN.
- Only structured tabular data is used; real-time clinical data and imaging data are not included.
- The model performance depends on the quality and completeness of the input data.
- The system is not yet integrated with real hospital or clinical decision systems.
- Interpretability of deep learning models is limited, making it harder to explain predictions to medical professionals.

### Future Work
- Train the model on larger and more diverse datasets for improved robustness.
- Integrate real-time patient monitoring data from wearable devices.
- Extend the model to include medical imaging data such as CT or MRI scans.
- Improve explainability using techniques like SHAP or LIME.
- Deploy the system as a full web or mobile application for clinical use.
- Perform clinical validation with healthcare professionals.
- Explore advanced ensemble and hybrid deep learning models for higher accuracy.
- Add multilingual and user-friendly interfaces for broader accessibility.


## Deployment Info

- The trained stroke prediction model is integrated into a simple application for user input and output display.
- A Flask-based web interface is used to deploy the model locally.
- Users can enter patient health details through a form and receive stroke risk prediction results.
- The system runs on a local server for testing and demonstration purposes.
- The model file is saved and loaded using pickle/Joblib for prediction.
- This deployment allows easy testing and practical usage of the trained model.

