# Loan-Approval-Prediction-using-Machine-Learning
This project aims to predict whether a loan application will be approved or not based on various applicant features such as marital status, income, education, credit history, and more. The model uses a Random Forest Classifier to predict loan approval status. The project demonstrates end-to-end machine learning workflow, from data preprocessing to model evaluation.

Table of Contents
Project Overview
Dataset
Features
Steps Involved
Model Evaluation
Requirements
How to Run
Project Overview
The Loan Approval Prediction model predicts whether an applicant's loan application will be approved or rejected. The dataset used includes both numerical and categorical data, such as income, education, credit history, marital status, and more.

Dataset
The dataset contains the following columns:

Loan_ID: Unique identifier for each loan.
Gender: Gender of the applicant (Male/Female).
Married: Marital status (Married/Single).
Dependents: Number of dependents.
Education: Education level (Graduate/Not Graduate).
Self_Employed: Whether the applicant is self-employed.
ApplicantIncome: The applicant’s income.
CoapplicantIncome: Coapplicant’s income.
LoanAmount: The loan amount applied for.
Loan_Amount_Term: Term of the loan in months.
Credit_History: Credit history (1.0 or 0.0).
Property_Area: Area type (Urban/Semiurban/Rural).
Loan_Status: Target variable (Y/N, Approved/Not Approved).
Features
Gender: Categorical variable (Male, Female).
Marital Status: Categorical variable (Married, Single).
Dependents: Number of dependents.
Education: Categorical variable (Graduate, Not Graduate).
Self_Employed: Categorical variable (Yes, No).
ApplicantIncome: Numeric, annual income of the applicant.
CoapplicantIncome: Numeric, annual income of the coapplicant.
LoanAmount: Numeric, loan amount applied for.
Loan_Amount_Term: Numeric, loan duration in months.
Credit_History: Categorical (1.0 for good credit, 0.0 for bad credit).
Property_Area: Categorical variable (Urban, Semiurban, Rural).
Loan_Status: Target variable (Y = Loan approved, N = Loan not approved).
Steps Involved
Data Preprocessing:

Handling missing values using median for numerical data and mode for categorical data.
Label encoding categorical variables to make them suitable for machine learning models.
Model Building:

Splitting data into features and target variables.
Splitting data into training and testing sets.
Training the model using Random Forest Classifier.
Model Evaluation:

Evaluating the model performance using accuracy, confusion matrix, and classification report.
Analyzing feature importance to identify key factors affecting loan approvals.
Model Evaluation
Accuracy: The model achieved an accuracy of X% on the test set.
Confusion Matrix: Provides insights into the number of correct and incorrect predictions.
Classification Report: Displays precision, recall, F1-score, and support for both classes (approved and not approved).
Feature Importance: Highlights the most important features for loan approval predictions.
Requirements
Python 3.x
Pandas
NumPy
Scikit-Learn
Matplotlib
Seaborn

# Install the required libraries using pip:
pip install pandas numpy scikit-learn matplotlib seaborn
# How to Run
1. Clone the repository:- git clone https://github.com/your-username/Loan-Approval-Prediction-using-Machine-Learning.git
2. Navigate to the project directory:- cd Loan-Approval-Prediction-using-Machine-Learning
3. Place your CSV dataset (e.g., LoanApprovalPrediction.csv) in the project directory.
4. Run the Jupyter notebook or Python script:- jupyter notebook Loan_Approval_Prediction.ipynb 
# or 
python Loan_Approval_Prediction.py

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Dataset sourced from Kaggle.
Machine Learning algorithms and evaluation using Scikit-Learn.
Special thanks to open-source contributors.


