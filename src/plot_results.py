
import json

import matplotlib.pyplot as plt
import numpy as np

from experiment_utils import ROOT, LOGS_DIR, FIGURES_DIR

FIGURES_DIR.mkdir(exist_ok=True)


def load_json(filename):
    with (LOGS_DIR / filename).open() as f:
        return json.load(f)


# Q1: baseline versus bagging.
baseline = load_json("q1_baseline.json")
bagging = load_json("q1_bagging.json")

labels = ["Validation", "Clean test", "Masked test"]
single = [
    baseline["validation_accuracy"],
    baseline["clean_test_accuracy"],
    baseline["masked_test_accuracy"],
]
ensemble = [
    bagging["bagging_validation_accuracy"],
    bagging["bagging_clean_test_accuracy"],
    bagging["bagging_masked_test_accuracy"],
]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width / 2, single, width, label="Single tree")
ax.bar(x + width / 2, ensemble, width, label="Bagging (100 trees)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.set_title("Q1: Decision tree vs. bagging")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "q1_accuracy.png", dpi=200)
plt.close(fig)


# Q2: validation accuracy by depth and boosting rounds.
adaboost = load_json("q2_adaboost.json")
tuning = adaboost["tuning_results"]

fig, ax = plt.subplots(figsize=(8, 5))
for depth in sorted({r["max_depth"] for r in tuning}):
    subset = [r for r in tuning if r["max_depth"] == depth]
    subset.sort(key=lambda r: r["requested_rounds"])
    ax.plot(
        [r["requested_rounds"] for r in subset],
        [r["validation_accuracy"] for r in subset],
        marker="o",
        label=f"Depth {depth}",
    )

ax.set_xlabel("Requested boosting rounds")
ax.set_ylabel("Validation accuracy")
ax.set_title("Q2: SAMME validation tuning")
ax.set_xticks([10, 25, 50, 100])
ax.set_ylim(0, 1)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "q2_tuning.png", dpi=200)
plt.close(fig)


# Q3: individual and majority-vote accuracy by tree depth.
bias_variance = load_json("q3_bias_variance.json")
depth_results = bias_variance["depths"]

depth_labels = [str(r["max_depth"]) for r in depth_results]
individual = [r["mean_individual_test_accuracy"] for r in depth_results]
majority = [r["aggregate_majority_vote_accuracy"] for r in depth_results]
variance = [r["multiclass_variance"] for r in depth_results]

x = np.arange(len(depth_labels))
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, individual, marker="o", label="Mean individual accuracy")
ax.plot(x, majority, marker="s", label="Majority-vote accuracy")
ax.set_xticks(x)
ax.set_xticklabels(depth_labels)
ax.set_xlabel("Tree depth (unlimited means no depth limit)")
ax.set_ylabel("Test accuracy")
ax.set_ylim(0, 1)
ax.set_title("Q3: Accuracy across bootstrap-trained trees")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "q3_accuracy.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, variance, marker="o")
ax.set_xticks(x)
ax.set_xticklabels(depth_labels)
ax.set_xlabel("Tree depth (unlimited means no depth limit)")
ax.set_ylabel("Multiclass prediction variance")
ax.set_title("Q3: Prediction variance across bootstrap samples")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "q3_variance.png", dpi=200)
plt.close(fig)

print("Plots saved in:", FIGURES_DIR)
for path in sorted(FIGURES_DIR.glob("*.png")):
    print(path.name)
