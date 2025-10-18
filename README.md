# Delivery Time Prediction

A machine learning project to predict delivery times for food delivery services using historical order data.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data](#data)
- [Evaluation](#evaluation)
- [Running the Project](#running-the-project)
- [Tests](#tests)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to build accurate delivery time prediction models to help food delivery platforms estimate arrival times, improve customer satisfaction, and optimize delivery operations.

### Features:

- **Data Preprocessing**: Cleaning and feature engineering for delivery data
- **Multiple Models**: Implementation of various ML algorithms for time prediction
- **Model Evaluation**: Comprehensive performance metrics and comparison
- **Feature Analysis**: Identification of key factors affecting delivery times
- **Scalable Pipeline**: End-to-end ML pipeline from data to predictions

## Installation

To get started with the project, follow the steps below:

1. **Clone the repository:**

   ```bash
   git clone git@github.com:codehass/delivery-time-prediction.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd delivery-time-prediction
   ```

3. **Install the required dependencies:**

   You can install the required Python libraries via `pip`. Make sure you have Python 3.6+ installed.

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, you can manually install the dependencies:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn pytest
   ```

## Data

The dataset includes features such as:

- Order_ID
- Distance_km
- Weather
- Traffic_Level
- Time_of_Day
- Time_of_Day
- Preparation_Time_min
- Courier_Experience_yrs
- Delivery_Time_min (target variable)

The data is stored in the `data/dataset.csv` file.

### Data Preprocessing

1. **Loading the Data:** The raw data is loaded using `pandas`.
2. **Feature Engineering:** The features are processed to clean the data, handle missing values, encode categorical variables, and scale numerical features.
3. **Train-Test Split:** The data is split into training and testing datasets using `train_test_split` from `sklearn`.

## Models Implemented

Three machine learning models are used for delivery time prediction:

1. **RandomForestRegressor**
2. **SVR**

### Model Training and Evaluation

Each model is trained on the processed dataset and evaluated based on:

- **MAE (Mean Absolute Error)**: Measures the average of the absolute differences between predicted and actual values, indicating the magnitude of errors.
- **MSE (Mean Squared Error)**: Measures the average of the squared differences between predicted and actual values, giving more weight to larger errors.
- **R² (R-Squared)**: Represents the proportion of variance in the target variable that is explained by the model, with higher values indicating better performance.

### Code Implementation

- **`pipeline.py`**: Contains functions for data preprocessing and splitting the dataset.
- **`data_exploration.ipynb/`**: Contains Jupyter notebooks for exploratory data analysis (EDA) and model experimentation.
- **`test_pipeline.py`**: Contains unit tests for various parts of the pipeline, ensuring that the data processing and modeling steps function correctly.

## Evaluation

The model's performance is evaluated based on several metrics:

- **Confusion Matrix**
- **Precision-Recall Curve**

The precision-recall curve is plotted for all three models, and the one with the best trade-off between precision and recall is selected as the final model for churn prediction.

## Running the Project

Open the eda_analysis.ipynb and run the cells in sequence. This notebook trains the three models: Logistic Regression, Random Forest, and K-Nearest Neighbors. The notebook will:

- **Load the dataset**

- **Preprocess the data**

- **Train each model**

Evaluate each model’s performance based on mae, mse, and F1-score

**2. View Results**

After running the training and evaluation steps, you will see evaluation metrics such as:

Confusion Matrix

These metrics will help you compare the performance of the models and choose the best one.

## Tests

This project includes unit tests to ensure the correctness of various parts of the code, including:

To run the tests:

```bash
pytest
```

This will run all the tests in the `test_pipeline.py` file.

## Contributing

Contributions are welcome! If you find a bug or want to improve the project, feel free to fork the repository and submit a pull request.

To contribute:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
