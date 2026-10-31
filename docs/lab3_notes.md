# Lab 3 – Training Churn Models

## 1. Why do we split data into training and test sets?

We use training data to teach the model and test data to check how well it performs on unseen data.

The scaler should learn only from training data to prevent data leakage.

## 2. What does stratify=y do?

It maintains a similar churn rate in both training and test datasets.

## 3. Compare Random Forest at max_depth=6 and None.

At max_depth=6, training accuracy is 81.2% and test accuracy is 78.0%.

At max_depth=None, training accuracy is 100% and test accuracy is 76.6%.

This indicates overfitting when the tree depth is unlimited.

## 4. Why is 80% accuracy less impressive than it sounds?

The baseline accuracy is already 75.6% because most customers do not churn.

Therefore, 80% accuracy is only a small improvement over the baseline.

## 5. Which features are the strongest signs of customer churn?

The two strongest signs of customer churn are late payments (+0.43) and support tickets (+0.35).

Tenure months (-1.03) is the strongest sign of lower churn risk, meaning customers with longer tenure are less likely to leave.

The Random Forest model also identifies tenure months as its most important feature (0.338).

This is consistent with the importance of customer tenure and engagement in the Lab 2 analysis.