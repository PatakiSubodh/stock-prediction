# **stock prediction**

## how to run the file:
run the file **app.py** using the following command:
`streamlit run app.py`

_be sure to be inside the right directory_


# **_viva-questions_**
# Stock Trend Prediction Project Q&A

## Theoretical Questions

### Q1. Explain the purpose of using `MinMaxScaler` in this project.
**A:** `MinMaxScaler` is used to scale the data to a specific range, typically between 0 and 1. This normalization is important because it ensures that all features contribute equally to the model, improving the efficiency and performance of the linear regression algorithm.

### Q2. Why do we split the data into training and testing sets?
**A:** Splitting data into training and testing sets allows us to evaluate the performance of the model on unseen data. The training set is used to fit the model, and the testing set is used to assess its predictive accuracy. This helps in understanding how well the model generalizes to new data.

### Q3. What is the significance of the moving averages (100MA and 200MA) in stock trend analysis?
**A:** Moving averages smooth out price data to identify trends over a specific period. The 100-day and 200-day moving averages are commonly used to indicate medium-term and long-term trends, respectively. They help in identifying support and resistance levels, trend direction, and potential buy or sell signals.

## Practical Questions

### Q4. How did you ensure the CSV files are correctly loaded and processed in your project?
**A:** The CSV files are loaded using `pandas.read_csv()` and processed by assigning appropriate column names and converting the 'Date' column to datetime format. The dataframe is then set to use the 'Date' column as its index to facilitate time series analysis.

### Q5. Can you explain the steps involved in preparing the data for training the linear regression model?
**A:** The steps involved are:
1. Loading and preprocessing the data (handling dates and missing values).
2. Splitting the data into training and testing sets.
3. Normalizing the data using `MinMaxScaler`.
4. Creating sequences of 100 data points to use as input features (`x_train` and `x_test`).
5. Flattening the sequences for input into the linear regression model.

### Q6. How did you handle the potential overfitting of the linear regression model?
**A:** Overfitting can be mitigated by using cross-validation techniques and ensuring that the model is validated on a separate testing set. In this project, the data is split into training and testing sets, and the model's performance is evaluated on the testing set to check for overfitting.

## Technical Questions

### Q7. Describe the role of `LinearRegression` in your project. Why did you choose this model?
**A:** `LinearRegression` is used to predict future stock prices based on past data. It fits a linear model to the training data by minimizing the least squares error. This model was chosen because it is simple to implement, easy to interpret, and suitable for understanding the relationship between variables in a linear manner.

### Q8. How does the reshaping of `x_train` and `x_test` impact the linear regression model?
**A:** Reshaping `x_train` and `x_test` from sequences of 100 data points to a flat array is necessary because the `LinearRegression` model expects 2D input where each row represents an observation and each column represents a feature. Flattening the input allows the model to process the time series data correctly.

### Q9. What are the potential limitations of using a linear regression model for stock price prediction?
**A:** The main limitations include:
- **Assumption of Linearity:** Stock prices often follow non-linear patterns, and linear regression may not capture these complexities.
- **Sensitivity to Outliers:** Linear regression is sensitive to outliers, which can skew predictions.
- **Feature Engineering:** It may require extensive feature engineering to improve accuracy, as it does not handle complex relationships well.
- **Overfitting/Underfitting:** There is a risk of overfitting to the training data or underfitting if the model is too simple.

## Integration and Implementation Questions

### Q10. How did you integrate the machine learning model with the Streamlit application for visualization?
**A:** The machine learning model is integrated with Streamlit by:
1. Loading and preprocessing the data within the Streamlit app.
2. Training the linear regression model on the training data.
3. Using the trained model to make predictions on the testing data.
4. Creating visualizations using `matplotlib` and displaying them within the Streamlit app using `st.pyplot()`.

### Q11. Explain the process of saving and displaying plots in Streamlit.
**A:** Plots are created using `matplotlib` and saved to a `BytesIO` buffer. This buffer is then encoded to base64 and passed to the Streamlit template to display the plot. This approach allows embedding the plot directly within the HTML content served by Streamlit.

### Q12. How would you deploy this Streamlit application for wider usage?
**A:** To deploy the Streamlit application, you can:
1. Host the application on a cloud platform like AWS, Heroku, or Streamlit Sharing.
2. Ensure that the necessary dependencies are included in a requirements.txt file.
3. Use a web server to serve the application and set up appropriate configurations for scalability and security.
4. Provide the necessary CSV files and ensure the paths are correctly set for deployment.


---------------------------------- **OR** --------------------------------------
# Stock Trend Prediction Project

## Purpose
### Q13. Can you briefly explain the purpose of your project?

**A:** The purpose of this project is to predict stock price trends using historical stock price data. The project uses machine learning techniques to analyze past stock prices and generate predictions, which can help investors make informed decisions.

## Data Handling
### Q14. How do you handle missing values in your dataset?

**A:** In this project, we assume the provided dataset is clean. However, if there were missing values, we could handle them by methods such as forward filling, backward filling, or interpolation using pandas functions like `fillna()`. Ensuring data integrity is crucial before feeding it into the machine learning model.

## Feature Scaling
### Q15. Why did you use MinMaxScaler for scaling the data?

**A:** We used MinMaxScaler to scale the data to a range of 0 to 1 because linear regression models perform better when the data is normalized. Scaling ensures that all features contribute equally to the result, improving the model's performance and convergence speed.

## Model Selection
### Q16. Why did you choose Linear Regression for this project?

**A:** Linear Regression was chosen because it is a simple and interpretable model that works well for continuous output variables. Given the linear nature of stock price movements over short periods, Linear Regression can effectively capture the trend without overfitting the data.

## Moving Averages
### Q17. What is the significance of 100MA and 200MA in stock price analysis?

**A:** The 100-day and 200-day moving averages (100MA and 200MA) are commonly used in stock price analysis to identify long-term trends. The 100MA provides insight into the short to mid-term trend, while the 200MA offers a long-term perspective. They help investors understand the overall direction of the stock and identify potential buy or sell signals.

## Data Splitting
### Q18. How did you split your data into training and testing sets, and why?

**A:** The data was split into 70% for training and 30% for testing. This split ensures that the model has enough data to learn from while keeping a significant portion for validation to evaluate the model's performance on unseen data.

## Predictive Modeling
### Q19. How do you ensure that your predictions are based on previous actual prices rather than previous predictions?

**A:** In the project, predictions are based on the actual stock prices from the dataset. We use a sliding window approach where each prediction is made using the previous 100 actual stock prices, ensuring that each prediction is independent of previous predictions and based solely on historical data.

## Visualization
### Q20. Why is data visualization important in your project?

**A:** Data visualization is crucial because it helps to understand the trends and patterns in the stock prices, which are not easily discernible from raw data. It also provides a visual way to compare the actual and predicted prices, making it easier to assess the model's performance and communicate the results.

## Technical Implementation
### Q21. Can you explain the steps involved in preparing the data for the Linear Regression model?

**A:** The steps involved are:
1. **Loading the Data:** Reading the CSV file into a pandas DataFrame.
2. **Preprocessing:** Converting dates to datetime format and setting them as the index.
3. **Splitting Data:** Dividing the data into training and testing sets.
4. **Scaling Data:** Normalizing the data using MinMaxScaler.
5. **Creating Training Data:** Forming sequences of 100 past stock prices to predict the next price.
6. **Training the Model:** Fitting the Linear Regression model to the training data.
7. **Creating Test Data:** Using the last 100 days of the training set combined with the test set for predictions.
8. **Making Predictions:** Using the trained model to predict future stock prices.

## Model Evaluation
### Q22. How do you evaluate the performance of your model?

**A:** The performance of the model is evaluated by comparing the predicted prices to the actual prices in the test set. Metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) can be used to quantify the accuracy of the predictions. Additionally, visualizing the predicted prices against the actual prices provides a clear indication of how well the model is performing.

## Limitations
### Q23. What are the limitations of your current model?

**A:** The limitations of the current model include:
1. **Simplicity:** Linear Regression is a simple model and may not capture complex patterns in the stock prices.
2. **Assumption of Linearity:** The model assumes a linear relationship between past and future prices, which may not always be the case.
3. **Lack of Features:** The model only uses past prices and does not incorporate other influential factors like market sentiment, news, or economic indicators.
4. **Overfitting:** If not carefully managed, the model might overfit to the training data, reducing its generalizability to new data.

## Future Work
### Q24. What improvements or future work do you suggest for this project?

**A:** Future improvements could include:
1. **Using More Advanced Models:** Incorporating more sophisticated models like LSTM (Long Short-Term Memory) networks that are better suited for time series data.
2. **Feature Engineering:** Adding more features such as volume, technical indicators, or external factors like news sentiment.
3. **Hyperparameter Tuning:** Performing grid search or random search to find the optimal parameters for the model.
4. **Cross-Validation:** Using cross-validation techniques to ensure the model's robustness and generalizability.
5. **Real-Time Predictions:** Implementing a real-time prediction system that updates with new data.
