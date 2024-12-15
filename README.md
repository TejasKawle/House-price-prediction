# House Price Detection System

This project is a **Machine Learning model** built to predict house prices based on various features in the dataset. The model uses the **XGBoost Regressor** for accurate predictions. The dataset used includes housing attributes like the number of rooms, area, and more.

---

## **Features**
- **Data Preprocessing**:
  - Handles missing values using mean imputation.
  - Calculates statistical measures for feature understanding.
- **Feature Correlation Analysis**:
  - Uses a heatmap to visualize correlations among the features.
- **Model Implementation**:
  - Employs the XGBoost Regressor for prediction.
- **Performance Evaluation**:
  - Metrics: R-squared error, Mean Absolute Error (MAE), and visualizations for performance analysis.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - XGBoost

---

## **Project Workflow**

### 1. Data Loading:
The dataset is loaded from a CSV file using Pandas.

### 2. Data Preprocessing:
- Missing values in the dataset are imputed with the mean of respective columns.
- Statistical measures are calculated to understand the dataset.

### 3. Correlation Analysis:
- A heatmap is constructed to analyze correlations between features.

### 4. Splitting the Data:
- The dataset is split into **training** and **testing** sets using an 80-20 split ratio.

### 5. Model Training:
- The `XGBRegressor` is used to train the model on the training data.
- Early stopping can be added to avoid overfitting.

### 6. Model Evaluation:
- The model’s performance is measured using:
  - **R-squared Error**
  - **Mean Absolute Error (MAE)**
  - **Root Mean Squared Error (RMSE)**
- A scatter plot visualizes the actual vs. predicted prices.

---

## **Setup and Usage**

### Prerequisites:
1. Python 3.7+
2. Required Libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```

### Steps to Run:
1. Clone the repository or download the code.
2. Ensure the dataset (`HousingData.csv`) is in the working directory.
3. Run the script:
   ```bash
   python house_price_detection.py
   ```
4. View the metrics and visualization outputs.

---

## **Dataset**
The dataset used for this project includes housing data with features like:
- CRIM: Per capita crime rate by town
- ZN: Proportion of residential land zoned for lots
- RM: Average number of rooms per dwelling
- MEDV: Median value of owner-occupied homes (target variable)

---

## **Improvements and Future Scope**
1. **Hyperparameter Tuning**: Use GridSearchCV to optimize model parameters.
2. **Feature Engineering**: Add or transform features to improve predictive power.
3. **Cross-Validation**: Implement k-fold cross-validation for robust performance evaluation.
4. **Deployment**: Deploy the model using Flask or FastAPI for real-world usage.

---

## **Acknowledgments**
- The dataset used is publicly available.
- Libraries and frameworks that made this project possible.

---

## **License**
This project is licensed under the MIT License.

