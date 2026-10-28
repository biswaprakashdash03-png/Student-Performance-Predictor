# Project Documentation: Student Performance Predictor

## 1. Problem statement
This educational project demonstrates how regression can estimate a numeric outcome—an exam score—from several academic indicators.

## 2. Objective
Build an end-to-end ML workflow covering dataset preparation, training, evaluation, model persistence, and a simple web interface.

## 3. Inputs and output
Inputs: study hours per day, attendance percentage, previous score, and assignments completed.
Output: estimated exam score on a 0–100 scale.

## 4. Dataset
The included 500-row CSV is synthetic and reproducible. Its values are generated for demonstration, not collected from real students. Synthetic patterns can make metrics look better than performance on real data.

## 5. Algorithm
Linear Regression estimates a numeric target by fitting a linear relationship between input features and exam score. The training pipeline imputes missing values using the median and scales the features before fitting the model.

## 6. Evaluation
The data is split into 80% training and 20% testing with a fixed random seed.

- MAE: average absolute difference between predicted and actual test scores.
- RMSE: square root of mean squared prediction error; larger errors are penalized more heavily.
- R²: proportion of target variation explained by the model on the test split; it can be negative.

## 7. Limitations and responsible use
The demo data is synthetic, the relationship is simplified, and the model is not validated for real student outcomes. A prediction is not a guarantee. Do not use this app to grade students, decide admissions, or make other high-stakes educational decisions. Real use would require appropriate data governance, consent, bias analysis, validation, and human oversight.

## 8. Future scope
Try other regression algorithms, cross-validation, residual analysis, feature importance, data validation, automated tests, and deployment.
