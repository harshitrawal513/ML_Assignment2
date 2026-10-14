
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from experiment_utils import LOGS_DIR, load_dataset_and_split, make_masked_features

SEED = 26504
N_TREES = 100

LOGS_DIR.mkdir(exist_ok=True)

X_train, y_train, X_val, y_val, X_test, y_test, *_ = (
    load_dataset_and_split()
)


def predict_by_vote(models, X, classes):
    """Predict by majority vote; resolve ties using sorted class order."""
    predictions = np.array([model.predict(X) for model in models]).T
    votes = np.stack([(predictions == c).sum(axis=1) for c in classes], axis=1)
    return classes[np.argmax(votes, axis=1)], predictions


rng = np.random.default_rng(SEED)
classes = np.unique(y_train)
models = []
oob_votes = np.zeros((len(X_train), len(classes)), dtype=np.int32)
oob_counts = np.zeros(len(X_train), dtype=np.int32)

for tree_id in range(N_TREES):
    # Bootstrap sample drawn with replacement from the training set.
    bootstrap_idx = rng.integers(0, len(X_train), size=len(X_train))
    in_bag = np.zeros(len(X_train), dtype=bool)
    in_bag[bootstrap_idx] = True
    oob_idx = np.flatnonzero(~in_bag)

    tree = DecisionTreeClassifier(random_state=SEED + tree_id)
    tree.fit(X_train[bootstrap_idx], y_train[bootstrap_idx])
    models.append(tree)

    if len(oob_idx):
        oob_pred = tree.predict(X_train[oob_idx])
        for row, pred in zip(oob_idx, oob_pred):
            oob_votes[row, np.where(classes == pred)[0][0]] += 1
            oob_counts[row] += 1

# Baseline tree for comparison.
baseline = DecisionTreeClassifier(random_state=SEED)
baseline.fit(X_train, y_train)

val_pred, _ = predict_by_vote(models, X_val, classes)
test_pred, individual_test_preds = predict_by_vote(models, X_test, classes)
masked_pred, _ = predict_by_vote(
    models, make_masked_features(X_test), classes
)

valid_oob = oob_counts > 0
oob_pred = classes[np.argmax(oob_votes[valid_oob], axis=1)]
oob_accuracy = accuracy_score(y_train[valid_oob], oob_pred)

# Disagreement: mean fraction of tree pairs that disagree per test image.
if N_TREES > 1:
    disagreement = np.mean([
        np.mean(individual_test_preds[:, i] != individual_test_preds[:, j])
        for i in range(N_TREES)
        for j in range(i + 1, N_TREES)
    ])
else:
    disagreement = 0.0

# Vote margin: top vote share minus second-highest vote share.
test_vote_counts = np.stack(
    [(individual_test_preds == c).sum(axis=1) for c in classes], axis=1
)
sorted_votes = np.sort(test_vote_counts, axis=1)
vote_margins = (sorted_votes[:, -1] - sorted_votes[:, -2]) / N_TREES

results = {
    "seed": SEED,
    "n_trees": N_TREES,
    "baseline_validation_accuracy": float(accuracy_score(y_val, baseline.predict(X_val))),
    "baseline_clean_test_accuracy": float(accuracy_score(y_test, baseline.predict(X_test))),
    "baseline_masked_test_accuracy": float(accuracy_score(y_test, baseline.predict(make_masked_features(X_test)))),
    "bagging_validation_accuracy": float(accuracy_score(y_val, val_pred)),
    "bagging_clean_test_accuracy": float(accuracy_score(y_test, test_pred)),
    "bagging_masked_test_accuracy": float(accuracy_score(y_test, masked_pred)),
    "oob_accuracy": float(oob_accuracy),
    "oob_coverage": float(np.mean(valid_oob)),
    "mean_pairwise_test_disagreement": float(disagreement),
    "mean_test_vote_margin": float(np.mean(vote_margins)),
    "test_vote_margins": vote_margins.tolist(),
}

output_path = LOGS_DIR / "q1_bagging.json"
with output_path.open("w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print("Results saved to:", output_path)
