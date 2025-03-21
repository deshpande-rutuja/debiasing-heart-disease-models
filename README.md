# debiasing-heart-disease-models

This notebook focuses on developing a machine learning model for **multiclass classification** of heart disease severity using **Random Forests**. It combines and processes datasets from multiple sources, balances class distributions, and evaluates model performance through metrics and visualization.

---

## 📂 Project Structure

- **Notebook Name:** `heart_disease_rf_classifier.ipynb`
- **Datasets Used:**
  - [Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data)
  - [Hungary Heart Disease Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data)
- **Goal:** Predict the presence and severity of heart disease on a multiclass scale.

---

## 📌 Key Features

- ✅ Loads and merges multiple datasets from the UCI Heart Disease repository.
- ✅ Cleans missing values and normalizes input features.
- ✅ Encodes categorical labels for multiclass classification.
- ✅ Balances dataset using stratified techniques for fair evaluation.
- ✅ Splits data into train/test sets using `train_test_split`.
- ✅ Trains a **Random Forest Classifier** for robust predictions.
- ✅ Evaluates the model using:
  - Accuracy
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)
- ✅ Visualizes class distributions and confusion matrices with `seaborn` and `matplotlib`.

---

## 🛠️ Libraries Required

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

---

## 🧪 Model Training Details

- **Classifier:** RandomForestClassifier
- **Scaler:** StandardScaler
- **Validation:** Train-Test Split (default 70/30)
- **Evaluation Metrics:**
  - `accuracy_score`
  - `classification_report`
  - `confusion_matrix`

---

## 📊 Output Visualization

- Heatmaps for confusion matrices
- Countplots showing class distribution
- Plots for comparative performance across classes

---

## 📁 File Structure

```
.
├── heart_disease_rf_classifier.ipynb   # Main notebook
├── README.md                           # Project documentation
```

---

## 🧮 Credits

- Dataset: UCI Machine Learning Repository
- Notebook Author: Rutuja Deshpande
- Libraries: scikit-learn, pandas, seaborn, matplotlib

---

## 💬 Feedback & Contribution

If you'd like to contribute improvements or encounter any issues, feel free to submit a pull request or open an issue.

