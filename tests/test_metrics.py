"""Classification metrics. Owned by #4.

Every expected number below is worked out by hand in the docstrings, not taken
from scikit-learn: checking one library against another only proves that the
two agree, not that either is right.

Formulas follow ``docs/slides/Learning from data.pdf`` p31-p36, with the
confusion matrix laid out as the deck does it - true class as rows, predicted
class as columns:

    Precision = TP / (TP + FP)       column-wise: correct / everything predicted as the class
    Recall    = TP / (TP + FN)       row-wise:    correct / everything truly in the class
    F1        = 2 / (1/Precision + 1/Recall)

The tests drive the public ``evaluate()`` end to end, with a stub model whose
predictions are fixed in advance, so the argmax, the matrix, the per-class
arithmetic and the written ``metrics.json`` are all covered together.
"""

from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pr_edge.config import Config, ModelConfig
from pr_edge.data import CLASS_NAMES
from pr_edge.evaluate import evaluate
from pr_edge.schemas import validate_metrics


class FixedPredictionModel(nn.Module):
    """Predicts whatever class index it is fed.

    Each input sample is a single number holding the class the model should
    predict. The forward pass turns it into one-hot logits, so ``argmax``
    inside ``evaluate()`` recovers exactly that class.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.one_hot(x.long().view(-1), self.num_classes).float()


def run_evaluate(y_true: list[int], y_pred: list[int], num_classes: int, out_dir=None) -> dict:
    """Evaluate a fixed set of predictions against a fixed set of labels."""
    cfg = Config(run_id="test_metrics", model=ModelConfig(num_classes=num_classes))
    dataset = TensorDataset(torch.tensor(y_pred), torch.tensor(y_true))
    # A small batch size so the predictions arrive over several batches, as
    # they do on the real test set.
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    return evaluate(
        FixedPredictionModel(num_classes), loader, cfg, torch.device("cpu"), out_dir=out_dir
    )


# The worked example: 3 classes, 10 samples.
#
#                       Predicted
#                      c0   c1   c2  | row sum (support)
#     True   c0         3    1    0  |  4
#            c1         0    2    0  |  2
#            c2         2    1    1  |  4
#     -------------------------------+
#     column sum        5    4    1  | 10
#
# The matrix is deliberately not symmetric (c2 -> c0 happens twice, c0 -> c2
# never), and precision differs from recall in every class, so a transposed
# matrix would change every number below.
WORKED_TRUE = [0, 0, 0, 0, 1, 1, 2, 2, 2, 2]
WORKED_PRED = [0, 0, 0, 1, 1, 1, 0, 0, 1, 2]
WORKED_MATRIX = [
    [3, 1, 0],
    [0, 2, 0],
    [2, 1, 1],
]


def test_accuracy_matches_a_hand_computed_example():
    """Accuracy is the diagonal over the total.

    Correct predictions = 3 + 2 + 1 = 6 out of 10  ->  accuracy = 0.6
    """
    metrics = run_evaluate(WORKED_TRUE, WORKED_PRED, num_classes=3)

    assert metrics["num_samples"] == 10
    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["confusion_matrix"] == WORKED_MATRIX


def test_precision_and_recall_match_a_hand_computed_example():
    """Per-class precision, recall and F1, then their macro averages.

    Precision uses the column sums (everything predicted as the class):
        c0: 3/5 = 0.6        c1: 2/4 = 0.5        c2: 1/1 = 1.0
    Recall uses the row sums (everything truly in the class):
        c0: 3/4 = 0.75       c1: 2/2 = 1.0        c2: 1/4 = 0.25
    F1 = 2 / (1/P + 1/R):
        c0: 2 / (1/0.6 + 1/0.75) = 2 / (5/3 + 4/3) = 2/3
        c1: 2 / (1/0.5 + 1/1.0)  = 2 / (2 + 1)     = 2/3
        c2: 2 / (1/1.0 + 1/0.25) = 2 / (1 + 4)     = 0.4

    Macro averages weight every class equally:
        precision = (0.6 + 0.5 + 1.0) / 3   = 0.7
        recall    = (0.75 + 1.0 + 0.25) / 3 = 2/3
        F1        = (2/3 + 2/3 + 0.4) / 3   = 26/45
    """
    metrics = run_evaluate(WORKED_TRUE, WORKED_PRED, num_classes=3)
    per_class = metrics["per_class"]
    c0, c1, c2 = CLASS_NAMES[:3]

    expected = {
        c0: {"precision": 0.6, "recall": 0.75, "f1": 2 / 3, "support": 4},
        c1: {"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "support": 2},
        c2: {"precision": 1.0, "recall": 0.25, "f1": 0.4, "support": 4},
    }
    for name, values in expected.items():
        assert per_class[name]["precision"] == pytest.approx(values["precision"])
        assert per_class[name]["recall"] == pytest.approx(values["recall"])
        assert per_class[name]["f1"] == pytest.approx(values["f1"])
        assert per_class[name]["support"] == values["support"]

    assert metrics["macro_precision"] == pytest.approx(0.7)
    assert metrics["macro_recall"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx(26 / 45)


def test_confusion_matrix_rows_are_true_labels():
    """A one-directional mistake pins the layout.

    Two samples of class 0, both predicted as class 1, and no other mistakes:

        rows = true, cols = predicted   ->  [[0, 2], [0, 0]]   (what we want)
        transposed                      ->  [[0, 0], [2, 0]]

    Support is a row sum, so it must say class 0 has 2 samples and class 1
    has none. Class 0 is then missed entirely (recall 0) and class 1 is
    predicted twice but never correctly (precision 0).
    """
    metrics = run_evaluate([0, 0], [1, 1], num_classes=2)
    c0, c1 = CLASS_NAMES[:2]

    assert metrics["confusion_matrix"] == [[0, 2], [0, 0]]
    assert metrics["confusion_matrix_convention"].startswith("rows=true, cols=predicted")
    assert metrics["per_class"][c0]["support"] == 2
    assert metrics["per_class"][c1]["support"] == 0
    assert metrics["per_class"][c0]["recall"] == 0.0
    assert metrics["per_class"][c1]["precision"] == 0.0

    # The same check on the larger worked example: c2 -> c0 twice, c0 -> c2 never.
    worked = run_evaluate(WORKED_TRUE, WORKED_PRED, num_classes=3)
    assert worked["confusion_matrix"][2][0] == 2
    assert worked["confusion_matrix"][0][2] == 0


def test_never_predicted_class_reports_zero_not_nan(tmp_path):
    """Undefined precision must stay numeric so the JSON and tables hold.

    True labels [0, 1, 2, 2], predictions [0, 1, 1, 1]:

                       Predicted
                      c0   c1   c2
        True   c0      1    0    0
               c1      0    1    0
               c2      0    2    0

    Class 2 is never predicted, so its precision is 0 / 0. It is reported as
    0.0, and with precision and recall both 0 its F1 is 0.0 as well.
    Macro precision = (1 + 1/3 + 0) / 3 = 4/9, still a finite number.
    """
    metrics = run_evaluate([0, 1, 2, 2], [0, 1, 1, 1], num_classes=3, out_dir=tmp_path)
    c2 = CLASS_NAMES[2]

    assert metrics["per_class"][c2] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 2}
    assert metrics["macro_precision"] == pytest.approx(4 / 9)

    # The file on disk must be strict JSON (json.dumps writes NaN as a bare
    # token that strict parsers reject) and must pass the results schema.
    written = json.loads(
        (tmp_path / "metrics.json").read_text(encoding="utf-8"),
        parse_constant=lambda token: pytest.fail(f"metrics.json contains {token}"),
    )
    assert validate_metrics(written) == []
    for name, scores in written["per_class"].items():
        for key in ("precision", "recall", "f1"):
            assert math.isfinite(scores[key]), f"{name}.{key} is not finite"


def test_class_absent_from_the_labels_reports_zero_recall_not_nan():
    """The mirror case: a class with no true samples has recall 0 / 0.

    True labels [0, 0, 1], predictions [0, 2, 1]. Class 2 never occurs in
    the labels (support 0), so its recall is undefined and reported as 0.0.
    It was predicted once, wrongly, so its precision is 0/1 = 0.0 too.
    """
    metrics = run_evaluate([0, 0, 1], [0, 2, 1], num_classes=3)
    c2 = CLASS_NAMES[2]

    assert metrics["per_class"][c2] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    assert math.isfinite(metrics["macro_recall"])
