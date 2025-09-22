# dishap81-titanic
Machine Learning project for the Titanic dataset, focusing on predicting passenger survival. Features data exploration, outlier removal, and model training.
# 🚢 Titanic Survival Prediction: A Machine Learning Project

## 🎯 Project Goal
The objective of this project is to build a machine learning model to predict whether a passenger survived the sinking of the Titanic, based on data provided by the Kaggle "Titanic - Machine Learning from Disaster" competition.

## 💾 Dataset
The project utilizes the **Titanic** dataset from Kaggle.
* **Source:** [Kaggle Titanic Competition Link](https://www.kaggle.com/c/titanic/data)
* **Key Features Used:** Pclass (Ticket Class), Sex, Age, SibSp (# of siblings/spouses), Parch (# of parents/children), and Fare.
* **Target Variable:** Survived (0 = No, 1 = Yes).

## 🚀 Key Steps & Methodology

The solution follows a standard data science workflow:

1.  **Exploratory Data Analysis (EDA):** Initial analysis and visualization (like the boxplot you generated) to understand data distributions and relationships with survival.
2.  **Data Preprocessing:**
    * Handling **Missing Values** (e.g., imputing missing Age/Fare).
    * **Feature Engineering** (e.g., creating a `Title` feature from the Name).
    * **Outlier Removal** (as noted in your boxplot title).
3.  **Model Training:** Training various classification algorithms.
4.  **Model Evaluation:** Assessing the model's accuracy and other metrics.

## 🛠️ Installation & Requirements
To run this project locally, you need Python and the following libraries installed.

### Dependencies:
You can install all dependencies using `pip`:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn eda
