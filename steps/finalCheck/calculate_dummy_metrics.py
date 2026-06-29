from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from zenml import step
from typing import Annotated

@step
def evaluate_dummy_baseline(X_train, X_test, y_train, y_test) -> Annotated[float, "dummy_accuracy"]:
    """
    Trains a trivial baseline model that always predicts the majority class.
    Returns only the baseline accuracy.
    """

    dummy_model = DummyClassifier(strategy="most_frequent")
    dummy_model.fit(X_train, y_train)

    y_pred_dummy = dummy_model.predict(X_test)

    return accuracy_score(y_test, y_pred_dummy)