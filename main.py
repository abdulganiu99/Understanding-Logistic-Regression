# Import necessary libraries for data manipulation, visualization, and machine learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import specific modules from scikit-learn for the logistic regression model and utilities
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
from sklearn.metrics import roc_curve, roc_auc_score 
from sklearn.preprocessing import LabelBinarizer 
from sklearn.metrics import f1_score 

# Import warnings and filter them to ignore potential warnings during execution
import warnings
warnings.filterwarnings('ignore')

# Data Loading & Initial Exploratory Data Analysis (EDA) 
print("--- Data Loading & Initial EDA ---")

try:
    df = pd.read_csv('data/Pumpkin_Seeds_Dataset.csv', encoding='latin1')# this because the dataset contains non-ASCII characters instead of UTF-8
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Pumpkin_Seeds_Dataset.csv' not found.")
    print("Please ensure the dataset is in a 'data' subdirectory in the same directory as the script.")
    exit() # Exit the script if the file is not found

# Display the first 5 rows of the DataFrame to get a sense of the data structure.
print("\n--- Head of the dataset ---")
print(df.head())
print("\n" + "="*30 + "\n") # Separator for readability

# Print concise information about the DataFrame, including data types and non-null counts.
print("--- Dataset Info ---")
print(df.info())
print("\n" + "="*30 + "\n")

# Generate descriptive statistics of the numerical columns (mean, std, min, max, quartiles).
print("--- Dataset Description ---")
print(df.describe())
print("\n" + "="*30 + "\n")

# Count the occurrences of each unique value in the 'Class' column to check class distribution.
print("--- Class Distribution ---")
print(df['Class'].value_counts())
print("\n" + "="*30 + "\n")

# Store original column names before separating X and y for later reference
original_feature_names = df.drop(columns=['Class']).columns.tolist()

# Separate features (X) and the target variable (y) from the original DataFrame
X_original = df.drop(columns=['Class'])
y = df['Class']

# Test-train split (using the original feature set)
# 80% of the data is used for training (test_size = 0.2 means 20% for testing).
# random_state = 42 ensures reproducibility of the split.
X_train_original, X_test_original, y_train, y_test = train_test_split(
    X_original, y, test_size=0.2, random_state=42
)

# Feature Scaling (using the original feature set)
# Standardize features by removing the mean and scaling to unit variance.
scaler_original = StandardScaler()
# Fit the scaler on the training data and transform it.
X_train_scaled_original = scaler_original.fit_transform(X_train_original)
# Transform the test data using the *same* scaler fitted on the training data.
X_test_scaled_original = scaler_original.transform(X_test_original)

# --- 2. Train and Evaluate the Initial Model (all features) ---
print("--- 2. Train and Evaluate the Initial Model (all features) ---")
# Initialize the Logistic Regression model.
model_original = LogisticRegression()
# Train the model using the scaled training data.
model_original.fit(X_train_scaled_original, y_train)
# Make predictions on the scaled test data.
y_pred_original = model_original.predict(X_test_scaled_original)

# Evaluate the model's performance using common metrics at the default threshold (0.5).
print("--- Model Evaluation (Original Data - Default Threshold) ---")
accuracy_original = metrics.accuracy_score(y_test, y_pred_original)
classification_report_original = metrics.classification_report(y_test, y_pred_original)
confusion_matrix_original = metrics.confusion_matrix(y_test, y_pred_original)

print("Accuracy:", accuracy_original)
print("Classification Report:\n", classification_report_original)
print("Confusion Matrix:\n", confusion_matrix_original)
print("\n" + "="*30 + "\n")

# Plotting the Confusion Matrix heatmap (Original Data)
cm_original = confusion_matrix(y_test, y_pred_original)
labels_original = model_original.classes_

plt.figure(figsize=(10, 7))
sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', xticklabels=labels_original, yticklabels=labels_original)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression (Original Features)')
plt.savefig('confusion_matrix_logistic_regression_original.png')
# plt.show() # Uncomment this line if you are running this in an environment that displays plots

# ROC AUC (Original Data)
print("--- ROC AUC Analysis (Original Model) ---")
# Convert the true test labels to binary format (0 and 1).
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test).ravel() # Flatten the array to 1D

# Get the predicted probabilities for the positive class from the original model.
y_probs_original = model_original.predict_proba(X_test_scaled_original)[:, 1]

# Compute and print the ROC AUC score.
auc_score_original = roc_auc_score(y_test_bin, y_probs_original)
print(f"ROC AUC Score (Original Model): {auc_score_original:.3f}")
print("\n" + "="*30 + "\n")

# Plot Predicted Score Distributions (Original Data)
# Separate the predicted probabilities based on the actual class.
neg_scores_original = y_probs_original[y_test_bin == 0] # Probabilities for actual negative instances
pos_scores_original = y_probs_original[y_test_bin == 1] # Probabilities for actual positive instances
class_names_original = model_original.classes_ # Get the actual class names
neg_class_name_original = class_names_original[0] # 'Çerçevelik'
pos_class_name_original = class_names_original[1] # 'Ürgüp Sivrisi'

plt.figure(figsize=(8,5))
# KDE plot for the negative class scores, using the actual class name with a '-' prefix as the label.
sns.kdeplot(neg_scores_original, shade=True, label= (f"-{neg_class_name_original}"))
# KDE plot for the positive class scores, using the actual class name with a '+' prefix as the label.
sns.kdeplot(pos_scores_original, shade=True, label= (f"+{pos_class_name_original}"))
plt.xlabel('Predicted Probability') # Label for the x-axis
plt.ylabel('Density') # Label for the y-axis
plt.title('Predicted Score Distributions (KDE) - Original Features') # Title of the plot
plt.legend() # Display the legend
plt.savefig('predicted_score_distributions_original.png')
# plt.show()

# --- 3. Interpret the Coefficients of the Initial Model ---
print("--- 3. Interpret the Coefficients of the Initial Model ---")

# Get the intercept (log-odds when all features are 0, i.e., at their mean after scaling)
intercept_original = model_original.intercept_[0] # For binary classification, intercept_ is an array with one element

# Get the coefficients for each feature
# coef_ is an array of arrays for multi-class, but for binary it's usually a single array
coefficients_original = model_original.coef_[0]

print("--- Model Intercept (Original Model) ---")
print(f"Intercept: {intercept_original:.4f}")
print("\n" + "="*30 + "\n")

print("--- Model Coefficients (Original Model) ---")
# Print each feature name and its corresponding coefficient
# The coefficients are in the same order as the original_feature_names
for feature, coef in zip(original_feature_names, coefficients_original):
    print(f"Feature: {feature:<20} | Coefficient: {coef:.4f}")
print("\n" + "="*30 + "\n")

# Optional: Sort coefficients by absolute magnitude to see most influential features
print("--- Model Coefficients (Original Model - Sorted by Absolute Magnitude) ---")
# Create a list of tuples (absolute_coefficient, feature_name, coefficient)
sorted_coefs_original = sorted(zip(np.abs(coefficients_original), original_feature_names, coefficients_original), reverse=True)

for abs_coef, feature, coef in sorted_coefs_original:
    print(f"Feature: {feature:<20} | Coefficient: {coef:.4f} (Abs Magnitude: {abs_coef:.4f})")
print("\n" + "="*30 + "\n")

print("--- Interpretation and Next Steps ---")
print("The coefficients indicate the change in the log-odds of the positive class (Ürgüp Sivrisi)")
print("for a one-unit increase in the scaled feature value, holding other features constant.")
print("Features with absolute coefficients close to zero have a weaker linear association.")
print("Based on these results, 'Major_Axis_Length' (-0.0983) and 'Extent' (0.0963) have the smallest absolute coefficients.")
print("We will now explore if removing these features impacts the model's performance.")
print("\n" + "="*30 + "\n")


# --- 4. Based on Interpretation, Drop Features ---
print("--- 4. Dropping Features with Small Coefficients ---")
# Define the list of column names to drop based on their small coefficients identified in step 3.
columns_to_drop = ['Major_Axis_Length', 'Extent']

print(f"Attempting to drop columns: {columns_to_drop}")

# Create a new DataFrame by dropping the specified columns from the original DataFrame (df)
# This ensures we start from the original data structure before dropping.
try:
    df_reduced = df.drop(columns=columns_to_drop)
    print("\n__________df after droping columns__________")
    print(df_reduced.columns)
except KeyError as e:
    print(f"\nError dropping columns: {e}")
    print("Please ensure the column names to drop are correct and exist in the DataFrame.")
    print("Exiting script.")
    exit() # Exit the script if columns cannot be dropped

# --- 5. Train and Evaluate the Second Model (reduced features) ---
print("\n--- 5. Train and Evaluate the Second Model (reduced features) ---")

# Separate features (X) and the target variable (y) from the reduced DataFrame
X_reduced = df_reduced.drop(columns=['Class'])
# y remains the same as it was not affected by dropping feature columns

# Test-train split (using the reduced feature set)
# Use the same random_state as before for consistency in the split
X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

# Feature Scaling (using the reduced feature set)
# Need a new scaler for the reduced feature set as the number of features changed
scaler_reduced = StandardScaler()
X_train_scaled_reduced = scaler_reduced.fit_transform(X_train_reduced)
X_test_scaled_reduced = scaler_reduced.transform(X_test_reduced)

# Train the Logistic Regression model on the reduced data
model_reduced = LogisticRegression()
model_reduced.fit(X_train_scaled_reduced, y_train)
y_pred_reduced = model_reduced.predict(X_test_scaled_reduced)

# --- 6. Compare the performance of the two models (at default threshold) ---
print("--- 6. Comparison of Model Performance (Default Threshold) ---")

# Evaluate the reduced model at the default threshold
accuracy_reduced = metrics.accuracy_score(y_test, y_pred_reduced)
classification_report_reduced = metrics.classification_report(y_test, y_pred_reduced)
confusion_matrix_reduced = metrics.confusion_matrix(y_test, y_pred_reduced)

print("--- Evaluation for Original Model (Default Threshold) ---")
print("Accuracy:", accuracy_original)
print("Classification Report:\n", classification_report_original)
print("Confusion Matrix:\n", confusion_matrix_original)
print("\n")

print("--- Evaluation for Reduced Model (Default Threshold) ---")
print("Accuracy:", accuracy_reduced)
print("Classification Report:\n", classification_report_reduced)
print("Confusion Matrix:\n", confusion_matrix_reduced)
print("\n" + "="*30 + "\n")

# Plotting the Confusion Matrix heatmap (Reduced Data) for visual comparison
cm_reduced = confusion_matrix(y_test, y_pred_reduced)
labels_reduced = model_reduced.classes_

plt.figure(figsize=(10, 7))
sns.heatmap(cm_reduced, annot=True, fmt='d', cmap='Blues', xticklabels=labels_reduced, yticklabels=labels_reduced)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression (Reduced Features)')
plt.savefig('confusion_matrix_logistic_regression_reduced.png')
# plt.show() # Uncomment to display plot


# --- 7. Perform Threshold Optimization (on the Reduced Model) ---
print("--- 7. Threshold Optimization (on the Reduced Model) ---")
print("We will now find the optimal classification threshold for the reduced model")
print("to maximize the F1-score.")

# Get the predicted probabilities for the positive class from the reduced model
y_probs_reduced = model_reduced.predict_proba(X_test_scaled_reduced)[:, 1]

# Calculate ROC AUC for the reduced model
auc_score_reduced = roc_auc_score(y_test_bin, y_probs_reduced)
print(f"ROC AUC Score (Reduced Model): {auc_score_reduced:.3f}")
print("\n")

# choosing the optimal threshold for classification (on Reduced Data)
thresholds = np.linspace(0, 1, 101) # Check 101 thresholds between 0 and 1
f1_scores_reduced = []
# Iterate through each threshold to calculate the F1 score.
for t in thresholds:
    # Classify instances based on the current threshold.
    preds_reduced = (y_probs_reduced >= t).astype(int)
    # Calculate the F1 score for the current predictions and true labels.
    f1_scores_reduced.append(f1_score(y_test_bin, preds_reduced))

# Find the index of the threshold that resulted in the highest F1 score.
best_idx_reduced = np.argmax(f1_scores_reduced)
# Get the best threshold value.
best_t_reduced = thresholds[best_idx_reduced]
# Get the corresponding best F1 score.
best_f1_reduced = f1_scores_reduced[best_idx_reduced]
print(f"Best threshold (Reduced Model, F1-optimized) = {best_t_reduced:.2f}, F1 = {best_f1_reduced:.2f}")
print("\n")

# Separate scores for plotting (using probabilities from the reduced model)
neg_scores_reduced = y_probs_reduced[y_test_bin == 0]
pos_scores_reduced = y_probs_reduced[y_test_bin == 1]
# Class names are the same as before
class_names_reduced = model_reduced.classes_
neg_class_name_reduced = class_names_reduced[0]
pos_class_name_reduced = class_names_reduced[1]


# Plot Predicted Score Distributions with Optimal Threshold (Reduced Data)
plt.figure(figsize=(8,5))
sns.kdeplot(neg_scores_reduced, shade=True, label= (f"-{neg_class_name_reduced}"))
sns.kdeplot(pos_scores_reduced, shade=True, label= (f"+{pos_class_name_reduced}"))

# Add a vertical line at the optimal threshold for the reduced model
plt.axvline(x=best_t_reduced, color='red', linestyle='--', label=f'Optimal Threshold = {best_t_reduced:.2f}')

plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Predicted Score Distributions (KDE) - Reduced Features with Optimal Threshold')
plt.legend()
plt.savefig('predicted_score_distributions_reduced_with_threshold.png')
# plt.show()


# Applying the best threshold to get final predictions (on Reduced Data)
final_preds_reduced = (y_probs_reduced >= best_t_reduced).astype(int)

print("--- Model Evaluation (Reduced Model - Adjusted Threshold) ---")
print("Accuracy after adjusting threshold :", metrics.accuracy_score(y_test_bin, final_preds_reduced))
print("Confusion Matrix after adjusting threshold:\n", confusion_matrix(y_test_bin, final_preds_reduced))
print("\nClassification Report after adjusting threshold:\n", metrics.classification_report(y_test_bin, final_preds_reduced))
print("\n" + "="*30 + "\n")

print("--- Analysis Complete ---")
print("The script has performed logistic regression, interpreted coefficients,")
print("explored feature selection, compared models, and optimized the classification threshold.")
print("Review the output and generated plots to understand the model's performance.")

