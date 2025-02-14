import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import shap
from lime.lime_tabular import LimeTabularExplainer
import os
import time

start_time = time.time()

# Load the dataset with the full path
data_path = ##Link to Kaggle Dataset present in repository folder
data = pd.read_csv(data_path, delimiter=';')
print("Data loaded successfully.")
print(f"Time taken to load data: {time.time() - start_time:.2f} seconds")

# Preprocess the data
start_time = time.time()
print("Preprocessing data...")
data = pd.get_dummies(data, drop_first=True)
print("Data preprocessing completed.")
print(f"Time taken to preprocess data: {time.time() - start_time:.2f} seconds")

# Define features and target
X = data.drop('y_yes', axis=1)
y = data['y_yes']

# Split the data into train and test sets
start_time = time.time()
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Data split completed.")
print(f"Time taken to split data: {time.time() - start_time:.2f} seconds")

# Standardize the features
start_time = time.time()
print("Standardizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Feature standardization completed.")
print(f"Time taken to standardize features: {time.time() - start_time:.2f} seconds")

# Build a classification model
start_time = time.time()
print("Building classification model...")
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model training completed.")
print(f"Time taken to train model: {time.time() - start_time:.2f} seconds")

# Evaluate the model
start_time = time.time()
print("Evaluating model...")
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"Time taken to evaluate model: {time.time() - start_time:.2f} seconds")

# Global interpretability with SHAP
start_time = time.time()
print("Calculating SHAP values for global interpretability...")
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test[:500])  # Use a smaller subset of the test data
print("SHAP values calculated.")
print(f"Time taken to calculate SHAP values: {time.time() - start_time:.2f} seconds")

# Commenting out the plotting section for now
# start_time = time.time()
# print("Generating SHAP summary plot...")
# shap.summary_plot(shap_values, X_test[:500], feature_names=X.columns)
# print("SHAP summary plot generated.")
# print(f"Time taken to generate SHAP summary plot: {time.time() - start_time:.2f} seconds")

# Save global feature importance to CSV
start_time = time.time()
global_importance_path = '/Users/karlaguzman/Downloads/archive/global_feature_importance.csv'
print(f"Saving global feature importance to: {global_importance_path}")
shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
shap_values_df.to_csv(global_importance_path, index=False)
print("Global feature importance saved successfully.")
print(f"Time taken to save global feature importance: {time.time() - start_time:.2f} seconds")

# Local interpretability with LIME
start_time = time.time()
print("Calculating LIME explanations for local interpretability...")
explainer = LimeTabularExplainer(X_train, feature_names=X.columns, class_names=['No', 'Yes'], discretize_continuous=True)

# Explain specific observations
obs_indices = [3, 19]  # Indices of observations to explain
lime_explanations = {}

for obs_index in obs_indices:
    print(f"Explaining observation {obs_index} with LIME...")
    exp = explainer.explain_instance(X_test[obs_index], model.predict_proba, num_features=10)
    lime_explanations[obs_index] = exp
    print(f"LIME explanation for observation {obs_index} completed.")
    exp.save_to_file(f'/Users/karlaguzman/Downloads/archive/lime_explanation_{obs_index}.html')
    print(f"LIME explanation for observation {obs_index} saved successfully.")
    print(f"Time taken for LIME explanation {obs_index}: {time.time() - start_time:.2f} seconds")

print("All tasks completed successfully.")
