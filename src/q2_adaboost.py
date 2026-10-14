
import json

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from experiment_utils import LOGS_DIR, load_dataset_and_split, make_masked_features

SEED = 26504
DEPTHS = [1, 2, 3, 4]
ROUND_OPTIONS = [10, 25, 50, 100]


class SAMME:
    """Multiclass AdaBoost using the discrete SAMME algorithm."""

    def __init__(self, n_estimators=50, max_depth=1, random_state=26504):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = []
        self.classes_ = None

    def fit(self, X, y):
        n_samples = len(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("SAMME requires at least two classes.")

        sample_weights = np.full(n_samples, 1.0 / n_samples)

        for iteration in range(self.n_estimators):
            stump = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state + iteration,
            )
            stump.fit(X, y, sample_weight=sample_weights)
            predictions = stump.predict(X)
            incorrect = predictions != y
            error = float(np.dot(sample_weights, incorrect))

            # A learner no better than random guessing cannot be added.
            if error >= 1.0 - 1.0 / n_classes:
                if not self.estimators_:
                    raise ValueError(
                        "Weak learner did not beat random guessing."
                    )
                break

            # A perfect learner needs no further boosting rounds.
            if error <= 1e-12:
                alpha = 1.0
                self.estimators_.append(stump)
                self.estimator_weights_.append(alpha)
                break

            alpha = np.log((1.0 - error) / error) + np.log(n_classes - 1)
            self.estimators_.append(stump)
            self.estimator_weights_.append(float(alpha))

            # SAMME reweights only incorrectly classified samples.
            sample_weights *= np.exp(alpha * incorrect)
            sample_weights /= sample_weights.sum()

        if not self.estimators_:
            raise RuntimeError("SAMME did not fit any weak learners.")

        return self

    def predict(self, X):
        scores = np.zeros((len(X), len(self.classes_)), dtype=float)

        for estimator, alpha in zip(
            self.estimators_, self.estimator_weights_
        ):
            predictions = estimator.predict(X)
            for class_index, class_label in enumerate(self.classes_):
                scores[:, class_index] += (
                    alpha * (predictions == class_label)
                )

        return self.classes_[np.argmax(scores, axis=1)]


def evaluate(model, X, y):
    return float(accuracy_score(y, model.predict(X)))


def main():
    LOGS_DIR.mkdir(exist_ok=True)
    X_train, y_train, X_val, y_val, X_test, y_test, *_ = (
        load_dataset_and_split()
    )

    results = []
    best_config = None
    best_validation_accuracy = -1.0

    # Tune on validation data only.
    for depth in DEPTHS:
        for rounds in ROUND_OPTIONS:
            model = SAMME(
                n_estimators=rounds,
                max_depth=depth,
                random_state=SEED,
            )
            model.fit(X_train, y_train)
            val_accuracy = evaluate(model, X_val, y_val)

            record = {
                "max_depth": depth,
                "requested_rounds": rounds,
                "actual_rounds": len(model.estimators_),
                "validation_accuracy": val_accuracy,
            }
            results.append(record)
            print(record)

            if val_accuracy > best_validation_accuracy:
                best_validation_accuracy = val_accuracy
                best_config = {
                    "max_depth": depth,
                    "n_estimators": rounds,
                }

    # Refit the chosen configuration on training data only.
    best_model = SAMME(
        n_estimators=best_config["n_estimators"],
        max_depth=best_config["max_depth"],
        random_state=SEED,
    )
    best_model.fit(X_train, y_train)

    final_results = {
        "seed": SEED,
        "tuning_metric": "validation_accuracy",
        "best_config": best_config,
        "best_validation_accuracy": best_validation_accuracy,
        "clean_test_accuracy": evaluate(best_model, X_test, y_test),
        "masked_test_accuracy": evaluate(
            best_model, make_masked_features(X_test), y_test
        ),
        "tuning_results": results,
    }

    output_path = LOGS_DIR / "q2_adaboost.json"
    with output_path.open("w") as f:
        json.dump(final_results, f, indent=2)

    print("\nBest configuration:", best_config)
    print(json.dumps(final_results, indent=2))
    print("Results saved to:", output_path)


if __name__ == "__main__":
    main()
