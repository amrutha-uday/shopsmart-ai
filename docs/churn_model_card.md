# ShopSmart AI - Churn Model Card

## 1. Model Overview

This model predicts whether a customer is likely to leave ShopSmart.

The purpose is to help the business identify customers who may need retention offers.

## 2. Model Details

- Winning model: Logistic Regression
- Model selection method: 5-fold cross-validation
- Cross-validation ROC-AUC: 0.777
- Decision threshold: 0.4
- Random Forest best max_depth: 4

## 3. Final Model Performance

- Accuracy: 0.806
- Precision: 0.626
- Recall: 0.508
- F1-score: 0.561
- ROC-AUC: 0.783

The model caught 62 out of 122 customers who left, with 37 false alarms.

## 4. Intended Use

The model is intended to help ShopSmart prioritize customers for retention activities.

It should support business decisions, not make automatic decisions about customers.

## 5. Limitations

1. The model uses synthetic customer data.
2. It missed some customers who actually left.
3. Predictions on real customer data may be different.
4. The model's performance depends on the quality and relevance of input data.

## 6. Conclusion

Logistic Regression performed better than Random Forest during cross-validation.

The selected model achieved an ROC-AUC of 0.783 on the test data and was saved using joblib.