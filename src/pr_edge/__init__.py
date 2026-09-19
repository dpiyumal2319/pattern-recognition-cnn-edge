"""EN3150 Assignment 03 - resource-constrained CNN for edge image classification.

The frozen contract for every public interface in this package is
``docs/interfaces.md``. Read it before implementing an issue.

Run directory layout produced by every experiment:

    results/<run_id>/
      config.yaml    resolved config, including the seed
      history.json   per-epoch train/val loss, accuracy and timing
      metrics.json   test accuracy, confusion matrix, precision, recall
      profile.json   parameters, disk size, MACs, mean epoch seconds
      model.pt       checkpoint (gitignored)
      figures/*.png
"""

__version__ = "0.1.0"
