

### Diabetic Patient Detection Project

#### **Overview**  
The **Diabetic Patient Detection** project is a machine learning-based classification system designed to predict whether a patient is diabetic based on various health-related features. The project utilizes **data preprocessing, exploratory data analysis (EDA), feature scaling, and multiple classification algorithms** to build a predictive model with optimal performance.  

#### **Technologies Used**  
- **Python**  
- **Pandas** (Data Manipulation)  
- **NumPy** (Numerical Computations)  
- **Matplotlib & Seaborn** (Data Visualization)  
- **Scikit-Learn** (Machine Learning Models & Evaluation Metrics)  

#### **Dataset**  
The dataset used for training and testing contains several health-related parameters such as:  
- **Glucose Level**  
- **Blood Pressure**  
- **Insulin Level**  
- **BMI (Body Mass Index)**  
- **Age**  
- **Diabetes Pedigree Function**  
- **Pregnancy Count**  

#### **Project Workflow**  

1. **Data Loading & Exploration**  
   - The dataset is loaded using Pandas (`pd.read_csv`) and basic exploratory analysis is performed.  
   - Missing values and data types are examined using `.info()` and `.describe()`.  

2. **Data Preprocessing**  
   - Standardization is applied using `StandardScaler()` to normalize feature values.  
   - The dataset is split into **training** and **testing sets** using `train_test_split()`.  

3. **Model Selection & Training**  
   - **Support Vector Machine (SVM)** classifier is implemented to classify diabetic vs. non-diabetic patients.  
   - The model is trained using the training data and evaluated on the test set.  

4. **Model Evaluation**  
   - Key performance metrics such as **accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC score** are computed.  
   - `accuracy_score`, `classification_report`, and `roc_curve` functions from **Scikit-Learn** are used to assess model performance.  

5. **Predictions & Insights**  
   - The trained model is tested on new patient data to predict diabetes status.  
   - Visualization tools (Seaborn & Matplotlib) are used to analyze the data distribution and feature correlations.  

#### **Results**  
- The project successfully classifies diabetic patients using machine learning.  
- Model performance is optimized by **feature scaling and hyperparameter tuning**.  
- The final model is evaluated with multiple metrics for reliability.  

