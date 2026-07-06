# Market Prices Prediction Using Regression Models

This project focuses on predicting market prices using various regression models. The goal is to evaluate the performance of different models and identify the best one for this task. The dataset includes features such as `output_own_price`, `output_comp_price`, and `output_own_profits`, among others.

## Dataset

The dataset used in this project is stored in `output_data.csv`. It contains the following key features:
- `output_date`: The date of the market data.
- `mkt_id`: The market identifier.
- `output_own_price`: The target variable representing the market price.
- `output_comp_price`: Competitor's price.
- `output_own_profits`: Own profits.

## Preprocessing

- **Encoding**: Categorical variables (`output_date` and `mkt_id`) are encoded using `LabelEncoder`.
- **Imputation**: Missing values are handled using `SimpleImputer` with a mean strategy.
- **Feature Selection**: Features are selected based on their importance using `SelectFromModel`.

## Models

The following regression models are implemented and evaluated:

### 1. Linear Regression
- **Mean Squared Error (MSE)**: 0.0663859081668457
- **Best Features**: `[output_own_cost, output_comp_price]`

### 2. Lasso Regression
- **Mean Squared Error (MSE)**: 0.0663859081668457
- **Best Features**: `[output_own_profits]`
- **Note**: The model shows signs of underfitting.

### 3. Ridge Regression
- **Mean Squared Error (MSE)**: 0.06638594717669895
- **Best Features**: `[output_own_cost, output_comp_price]`

### 4. Elastic Net Regression
- **Mean Squared Error (MSE)**: 0.12300046352838176
- **Best Features**: `[output_own_profits]`

### 5. Random Forest Regressor
- **Mean Squared Error (MSE)**: 0.014679371421792334
- **Best Features**: `[output_comp_price, output_X]`

### 6. Gradient Boosting Regressor
- **Mean Squared Error (MSE)**: 0.05191168906110088
- **Best Features**: `[output_comp_price, output_X]`

### 7. XGBoost Regressor
- **Mean Squared Error (MSE)**: 0.019841374133488583
- **Best Features**: `[output_comp_price]`

### 8. Neural Network Regressor
- **Mean Squared Error (MSE)**: 0.22620928336071672
- **Best Features**: `[output_comp_price, output_X, output_own_profits]`

## Results

The **Random Forest Regressor** achieved the best performance with the lowest Mean Squared Error (MSE) of **0.014679371421792334**. The **XGBoost Regressor** also performed well with an MSE of **0.019841374133488583**.

## Visualization

The script includes a function to plot actual vs. predicted values, providing a visual comparison of the model's performance. Below are the graphical representations of the models:

1. **Linear Regression**:
<img width="200" alt="linear" src="https://github.com/user-attachments/assets/0402f260-0c8f-4977-893e-f3c1beb61755" />

2. **Lasso Regression**:
<img width="200" alt="lasso" src="https://github.com/user-attachments/assets/08fd70da-1dd6-4610-8c10-f3afb63069bd" />

3. **Ridge Regression**:
<img width="200" alt="ridge" src="https://github.com/user-attachments/assets/d5128bf5-835d-4050-8d85-082eb4ee9fcc" />

4. **Elastic Net Regression**:
<img width="200" alt="elastic net" src="https://github.com/user-attachments/assets/bac4613c-b090-4dd4-88e1-abda0a28c989" />

5. **Random Forest Regressor**:
<img width="200" alt="random forest" src="https://github.com/user-attachments/assets/69b20987-e98a-4ccc-90ff-b7a7285e736a" />

6. **Gradient Boosting Regressor**:
<img width="200" alt="gradient boosting" src="https://github.com/user-attachments/assets/aae79331-8a72-4273-bd8f-0fff7d867826" />

7. **XGBoost Regressor**:
<img width="200" alt="XGB" src="https://github.com/user-attachments/assets/06b91c89-3936-4263-a40a-94cc113e9473" />

8. **Neural Network Regressor**:
<img width="200" alt="neural network" src="https://github.com/user-attachments/assets/0d3902cb-a849-4d01-bb47-acd85396a76a" />

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/MAHMOUD2ABDALLAH/furniture-Sales.git
   ```
2. Navigate to the project directory:
   ```bash
   cd furniture-Sales
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python main.py
   ```

## Dependencies

The project requires the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `xgboost`

You can install them using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

## Acknowledgments

- Thanks to the contributors of the `scikit-learn` and `xgboost` libraries for providing robust machine learning tools.
- Special thanks to Upwork for the freelancing opportunity.

---



### Key Points:
1. **Model Performance**: The README highlights the best-performing models (Random Forest and XGBoost) and their MSE scores.
2. **Visualization**: The graphs are linked to the corresponding models for easy reference.
3. **Usage Instructions**: Clear steps are provided for running the project.

<img width="100" height="100" alt="Upwork" src="https://github.com/user-attachments/assets/bc39e304-56ee-41a3-be6a-c8e7c6c9cc8e" />


