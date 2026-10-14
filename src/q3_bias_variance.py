
import json

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from experiment_utils import LOGS_DIR, load_dataset_and_split

SEED = 26504
N_BOOTSTRAPS = 30
DEPTHS = [1, 2, 3, 4, None]


def main():
    LOGS_DIR.mkdir(exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, *_ = (
        load_dataset_and_split()
    )

    rng = np.random.default_rng(SEED)
    classes = np.unique(y_train)
    results = []

    for depth in DEPTHS:
        # Each row stores predictions from one bootstrap-trained model.
        predictions = np.zeros(
            (N_BOOTSTRAPS, len(y_test)), dtype=y_test.dtype
        )
        bootstrap_accuracies = []

        for b in range(N_BOOTSTRAPS):
            bootstrap_idx = rng.integers(
                0, len(X_train), size=len(X_train)
            )

            model = DecisionTreeClassifier(
                max_depth=depth,
                random_state=SEED + b,
            )
            model.fit(X_train[bootstrap_idx], y_train[bootstrap_idx])

            predictions[b] = model.predict(X_test)
            bootstrap_accuracies.append(
                accuracy_score(y_test, predictions[b])
            )

        # Per-example probability that a bootstrap model predicts each class.
        class_probabilities = np.zeros(
            (len(y_test), len(classes)), dtype=float
        )
        for class_index, class_label in enumerate(classes):
            class_probabilities[:, class_index] = np.mean(
                predictions == class_label, axis=0
            )

        # Multiclass variance: average 1 - sum_c p(c|x)^2.
        variance = np.mean(
            1.0 - np.sum(class_probabilities**2, axis=1)
        )

        # Bias proxy: error of the aggregate majority-vote prediction.
        mean_class_prediction = classes[
            np.argmax(class_probabilities, axis=1)
        ]
        aggregate_error = 1.0 - accuracy_score(y_test, mean_class_prediction)

        results.append({
            "max_depth": depth if depth is not None else "unlimited",
            "n_bootstraps": N_BOOTSTRAPS,
            "mean_individual_test_accuracy": float(
                np.mean(bootstrap_accuracies)
            ),
            "std_individual_test_accuracy": float(
                np.std(bootstrap_accuracies)
            ),
            "aggregate_majority_vote_accuracy": float(
                accuracy_score(y_test, mean_class_prediction)
            ),
            "multiclass_variance": float(variance),
            "aggregate_error_bias_proxy": float(aggregate_error),
        })

        print(results[-1])

    output_path = LOGS_DIR / "q3_bias_variance.json"
    with output_path.open("w") as f:
        json.dump({
            "seed": SEED,
            "n_bootstraps": N_BOOTSTRAPS,
            "depths": results,
        }, f, indent=2)

    print("Results saved to:", output_path)


if __name__ == "__main__":
    main()
