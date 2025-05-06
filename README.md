# 🧪 Understanding Logistic Regression: A Pumpkin Seeds Case Study

[![GitHub Repo](https://img.shields.io/badge/GitHub-abdulganiu99/Understanding--Logistic--Regression-181717?logo=github&logoColor=white)](https://github.com/abdulganiu99/Understanding-Logistic-Regression) ![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python)

Welcome to this repository! This project is designed to help you understand the core concepts of **Logistic Regression**, a fundamental machine learning algorithm used for binary classification (predicting one of two outcomes).

We'll use the **Pumpkin Seeds Dataset** to build and analyze our model. The goal is to classify pumpkin seeds into one of two varieties: **Çerçevelik** or **Ürgüp Sivrisi**, based on their physical characteristics.

This project follows a step-by-step learning flow, similar to how you might approach a real-world machine learning problem.

---

## 🚀 What is Logistic Regression?

Imagine you want to predict if an email is **"Spam"** or **"Not Spam"**. Logistic Regression helps you do this by estimating the probability that an input (the email) belongs to a specific class (Spam). It then uses a threshold (usually 0.5) to make a final decision (if the probability is > 0.5, classify as Spam).

Unlike Linear Regression (which predicts a continuous value), Logistic Regression uses a special function (the **logistic** or **sigmoid** function) to squash the output into a probability between 0 and 1.

---

## 📂 The Dataset

The dataset contains various measurements of pumpkin seeds. Each row represents a single seed, and the columns are features like **Area**, **Perimeter**, **Major Axis Length**, etc. The target variable is **Class**, which tells us the variety of the seed (`Çerçevelik` or `Ürgüp Sivrisi`).

---

## 🗂️ Project Structure and Learning Flow

The main logic is contained in the `main.py` script. It walks through the following steps:

1. **🔍 Data Loading & Initial EDA (Exploratory Data Analysis)**
   - Load the dataset and get familiar with it.
   - Sample rows, data types, summary statistics.
   - Distribution of the target variable and missing-value checks.

2. **🤖 Train and Evaluate the Initial Model (all features)**
   - Separate features (X) from target (y).
   - Split into training and testing sets.
   - Feature scaling with `StandardScaler`.
   - Train a `LogisticRegression` model on all features.
   - Evaluate performance at the default threshold (0.5):
     - **Accuracy**, **Classification Report** (Precision, Recall, F1-score), **Confusion Matrix**.

3. **📈 Interpret the Model Coefficients**
   - Examine intercept and feature coefficients (log-odds impact).
   - Positive coefficient → increases probability of `Ürgüp Sivrisi`.
   - Negative coefficient → decreases probability of `Ürgüp Sivrisi`.
   - Coefficients near zero suggest weak influence.

4. **✂️ Feature Selection: Drop Low-Impact Features**
   - Identify features with the smallest absolute coefficients (`Major_Axis_Length`, `Extent`).
   - Create a reduced dataset without these features.

5. **🔄 Train and Evaluate the Second Model (reduced features)**
   - Repeat splitting, scaling, training with reduced feature set.
   - Evaluate using the same metrics at threshold 0.5.

6. **🔍 Compare Performance of Both Models**
   - Compare confusion matrices side by side:
     - True Negatives (TN)
     - False Positives (FP)
     - False Negatives (FN)
     - True Positives (TP)
   - Analyze the impact of dropping features on classification counts.

7. **⚙️ Threshold Optimization (on the Reduced Model)**
   - Obtain predicted probabilities for `Ürgüp Sivrisi`.
   - Compute ROC AUC score.
   - Plot KDE of predicted probabilities for both classes.
   - Iterate through possible thresholds (0 to 1), computing an evaluation metric (e.g., F1-score).
   - Select and visualize the optimal threshold on the KDE plot.
   - Re-evaluate performance (confusion matrix, classification report) at the optimal threshold.

---

## ▶️ How to Run the Code

1. **Clone the repository**
   ```bash
   git clone https://github.com/abdulganiu99/Understanding-Logistic-Regression.git
   cd Understanding-Logistic-Regression
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Download the dataset**
   - Place `Pumpkin_Seeds_Dataset.csv` inside a folder named `data/` in the project root.

4. **Run the script**
   ```bash
   python main.py
   ```
   The script will print outputs to the console and save confusion matrix and KDE plots as `.png` files in the script's directory.

---

## 🧹 Keeping Notebooks Clean (Optional)

If you prototyped in Jupyter Notebooks, remove outputs before committing:

1. Install **nbstripout**:
   ```bash
   pip install nbstripout
   ```
2. Install Git hook in your repo:
   ```bash
   nbstripout --install
   ```
3. To clean existing notebooks:
   - **Windows** (Command Prompt):
     ```bat
     for /r . %i in (*.ipynb) do nbstripout --force "%i"
     ```
   - **Linux/macOS** (bash):
     ```bash
     find . -name '*.ipynb' -exec nbstripout --force {} \;
     ```

---

This project provides a hands-on way to learn about logistic regression, model evaluation, coefficient interpretation, feature selection, and threshold tuning. Feel free to explore the code, tweak parameters, and ask questions!

**Happy Learning!**
