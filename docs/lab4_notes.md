# Lab 4 - Evaluation, Tuning and Saving Notes

## 1. Why can accuracy be misleading?

Accuracy measures the percentage of correct predictions.

Our model achieved 80.6% accuracy. However, accuracy alone can be misleading because most customers stay.

Recall was 50.8%, meaning the model caught only 50.8% of the customers who actually left.

## 2. False Positives and False Negatives

- False Positive: The model predicts that a customer will leave, but the customer actually stays.
- False Negative: The model predicts that a customer will stay, but the customer actually leaves.

For ShopSmart, false negatives can be costly because the business misses an opportunity to retain a customer.

## 3. What happens when we lower the threshold?

When we lower the decision threshold, more customers are flagged as likely to leave.

This generally increases recall but can reduce precision because more customers who would stay may also be flagged.

## 4. My chosen threshold

I selected a threshold of 0.4.

At this threshold:

- Precision: 0.626
- Recall: 0.508
- F1-score: 0.561
- Customers caught: 62 out of 122
- False alarms: 37

This threshold provides a balance between identifying customers who may leave and avoiding too many false alarms.

## 5. Why use cross-validation?

Cross-validation evaluates a model across multiple training-data splits.

We used 5-fold cross-validation on the training data to compare Logistic Regression and Random Forest.

The test data was kept separate and used for the final evaluation of the winning model.

This helps provide a fairer estimate of model performance.