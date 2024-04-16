# Analyzing customer churn using Deep Learning

### Introduction

In the competitive telecommunication industry, retaining customers is as crucial as acquiring new ones. The cost of acquiring a new customer substantially surpasses that of retaining an existing one. With this in mind, telecom companies are increasingly turning to advanced analytics to understand consumer behaviors and predict churn—whether customers are likely to cancel their services. This report details our approach using deep learning to predict customer churn based on various service usage data from a telecom company.

### Dataset Overview

The dataset used in this analysis includes various attributes related to the services used by customers of a telecom company. These attributes are:
Link to dataset: https://www.kaggle.com/datasets/barun2104/telecom-churn/data

Churn: Indicates if a customer cancelled service (1) or not (0).
AccountWeeks: Number of weeks the customer has had an active account.
ContractRenewal: Indicates if a customer recently renewed their contract (1) or not (0).
DataPlan: Indicates if a customer has a data plan (1) or not (0).
DataUsage: Monthly data usage in gigabytes.
CustServCalls: Number of calls made to customer service.
DayMins: Average daytime minutes used per month.
DayCalls: Average number of daytime calls made per month.
MonthlyCharge: Average monthly bill.
OverageFee: Largest overage fee in the last 12 months.
The dataset does not contain any missing values, making it an ideal candidate for training predictive models without the need for initial data cleaning or imputation.

### Model Architecture and Training Process

Four different models were developed and trained to predict customer churn based on the features outlined. Each model was built using TensorFlow and Keras, employing a sequential layer architecture with variations in configuration to assess their predictive performance:

#### Model 1
Architecture: This model consists of three dense layers with 128, 128, and 64 units respectively, each followed by batch normalization and dropout (30%) to prevent overfitting. The activation function used was 'ReLU', except for the output layer which used a 'Sigmoid' activation.
Optimizer: Adam optimizer with a learning rate of 0.001.
Loss Function: Binary crossentropy.
Metrics: Accuracy, AUC, and custom F1 score.
Training: The model was trained over 100 epochs with a batch size of 32, using early stopping based on validation AUC to prevent overfitting.
#### Model 2
A simpler model with two dense layers of 10 and 15 units respectively, followed by a sigmoid output layer.
Trained for 50 epochs to assess the impact of a less complex architecture.
#### Model 3
An extensive model featuring larger dense layers and more dropout to test robustness against overfitting.
Included a dense layer with 1024 units to capture more complex patterns in the data.
#### Model 4
Similar architecture to Model 1 but trained for 500 epochs with early stopping, providing a deeper training regimen to analyze its effect on the learning outcome.

### Model Evaluation and Results

Post-training, each model's performance was evaluated on a test dataset. The key performance metrics considered were accuracy, AUC (Area Under the Curve), and the custom F1 score:

- Model 1: Demonstrated high effectiveness with an accuracy of 93.10%, AUC of 91.55%, and an F1 score of 75.53%.
- Model 2: Despite its simplicity, achieved an accuracy of 92.80%, AUC of 92.50%, but a lower F1 score of 71.76%.
- Model 3: Showed lower performance compared to other models, which could be due to overfitting despite the dropout layers, with an accuracy of 88.61%, AUC of 91.93%, and an F1 score of 39.68%.
- Model 4: Performed moderately well with an accuracy of 90.25%, AUC of 90.80%, and an F1 score of 56.95%.

### Conclusion

The models developed in this study highlight the potential of using deep learning techniques in predicting customer churn in the telecom industry. Model 1 emerged as the most effective, balancing complexity and training depth to achieve high accuracy and F1 score. This suggests that for datasets with similar feature spaces, a moderately complex neural network may deliver optimal predictive performance. Future studies could explore the integration of more diverse data sources, further hyperparameter tuning, and the use of ensemble methods to enhance model robustness and accuracy.
