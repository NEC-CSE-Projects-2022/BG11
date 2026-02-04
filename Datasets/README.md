## 📁 Dataset

This project uses two different datasets: one for model pretraining and another for testing and evaluation to improve generalization using transfer learning.

### 🔹 Pretraining Dataset (Surgical Dataset)

The model is first trained on a surgical binary classification dataset to learn general medical patterns.

🔗 **[Download Surgical Dataset](https://www.kaggle.com/datasets/omnamahshivai/surgical-dataset-binary-classification)**

**Dataset Includes:**
- **Train Dataset** – Used for initial model training  
- **Validation Dataset** – Used for tuning hyperparameters  
- **Test Dataset** – Used for internal evaluation  

---

### 🔹 Testing Dataset (Cerebral Stroke Dataset)

After pretraining, the model is tested on a cerebral stroke prediction dataset to evaluate its performance on stroke risk prediction.

🔗 **[Download Cerebral Stroke Dataset](https://www.kaggle.com/datasets/shashwatwork/cerebral-stroke-predictionimbalaced-dataset)**

**Dataset Includes:**
- **Train Dataset** – Used for fine-tuning the pretrained model  
- **Validation Dataset** – Used for performance tuning  
- **Test Dataset** – Used for final evaluation  

---

🔗 **Original Dataset Sources:**
- Surgical Dataset – Kaggle (Binary Classification Medical Dataset)  
- Cerebral Stroke Dataset – Kaggle (Stroke Prediction Dataset)

These datasets help improve the robustness and accuracy of the model by applying transfer learning from one medical domain to another.
