# metrics/_classification.py
# Here we are defining classification metrics.

def accuracy_score(y, y_pred):
    """
    Accuracy score is the ratio of correctly predicted observations to the total observations.
    It is a common metric for classification problems, but it can be misleading if the classes are imbalanced.
    """
    correct_predictions = (y == y_pred).sum()
    total_predictions = len(y)
    accuracy = correct_predictions / total_predictions
    return accuracy

def precision_score(y, y_pred):
    """
    Precision score is the ratio of correctly predicted positive observations to the total predicted positives.
    It is a useful metric when the cost of false positives is high.
    """
    true_positives = ((y == 1) & (y_pred == 1)).sum()
    predicted_positives = (y_pred == 1).sum()
    precision = (true_positives / predicted_positives) if predicted_positives > 0 else 0
    return precision

def recall_score(y, y_pred):
    """
    Recall score is the ratio of correctly predicted positive observations to the all observations in actual class.
    It is a useful metric when the cost of false negatives is high.
    """
    true_positives = ((y == 1) & (y_pred == 1)).sum()
    actual_positives = (y == 1).sum()
    recall = (true_positives / actual_positives) if actual_positives > 0 else 0
    return recall

def f1_score(y, y_pred):
    """
    F1 score is the weighted average of Precision and Recall. It is a useful metric when you need to balance precision and recall.
    """
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def classification_report(y, y_pred):
    return {
        "Accuracy Score": round(accuracy_score(y, y_pred), 6),
        "Precision Score": round(precision_score(y, y_pred), 6),
        "Recall Score": round(recall_score(y, y_pred), 6),
        "F1 Score": round(f1_score(y, y_pred), 6)
    }