# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements Witches' Brew clean-label attack on Neural Networks.

| Paper link: https://arxiv.org/abs/2009.02276
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Tuple, TYPE_CHECKING, List

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import Attack
from art.attacks.poisoning.gradient_matching.gradient_matching import GradientMatchingMixin
from art.estimators import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.classification.tensorflow import TensorFlowV2Classifier

if TYPE_CHECKING:
    # pylint: disable=C0412
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class WitchesBrewAttack(GradientMatchingMixin, Attack):
    """
    Implementation of Witches' Brew Attack.

    | Paper link: https://arxiv.org/abs/2009.02276
    """

    attack_params = Attack.attack_params + [
        "classifier",
        "percent_poison",
        "epsilon",
        "max_trials",
        "max_epochs",
        "learning_rate_schedule",
        "batch_size",
        "clip_values",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        percent_poison: float,
        epsilon: float = 0.1,
        max_trials: int = 8,
        max_epochs: int = 250,
        learning_rate_schedule: Tuple[List[float], List[int]] = ([1e-1, 1e-2, 1e-3, 1e-4], [100, 150, 200, 220]),
        batch_size: int = 128,
        clip_values: Tuple[float, float] = (0, 1.0),
        verbose: int = 1,
    ):
        """
        Initialize a Gradient Matching Clean-Label poisoning attack (Witches' Brew).

        :param classifier: The proxy classifier used for the attack.
        :param percent_poison: The ratio of samples to poison among x_train, with range [0,1].
        :param epsilon: The L-inf perturbation budget.
        :param max_trials: The maximum number of restarts to optimize the poison.
        :param max_epochs: The maximum number of epochs to optimize the train per trial.
        :param learning_rate_schedule: The learning rate schedule to optimize the poison.
            A List of (learning rate, epoch) pairs. The learning rate is used
            if the current epoch is less than the specified epoch.
        :param batch_size: Batch size.
        :param clip_values: The range of the input features to the classifier.
        :param verbose: Show progress bars.
        """
        super().__init__(
            classifier=classifier,
            epsilon=epsilon,
            max_epochs=max_epochs,
            learning_rate_schedule=learning_rate_schedule,
            batch_size=batch_size,
            clip_values=clip_values,
            verbose=verbose,
        )
        self.percent_poison = percent_poison
        self.epsilon = epsilon
        self.learning_rate_schedule = learning_rate_schedule
        self.max_trials = max_trials
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.clip_values = clip_values
        self.initial_epoch = 0

        if verbose is True:
            verbose = 1
        self.verbose = verbose
        self._check_params()

    def poison(
        self, x_trigger: np.ndarray, y_trigger: np.ndarray, x_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimizes a portion of poisoned samples from x_train to make a model classify x_target
        as y_target by matching the gradients.

        :param x_trigger: A list of samples to use as triggers.
        :param y_trigger: A list of target classes to classify the triggers into.
        :param x_train: A list of training data to poison a portion of.
        :param y_train: A list of labels for x_train.
        :return: A list of poisoned samples, and y_train.
        """
        if isinstance(self.substitute_classifier, PyTorchClassifier):
            initializer = self._initialize_poison_pytorch
            poisoner = self._poison_pytorch
            finish_poisoning = self._finish_poison_pytorch
        elif isinstance(self.substitute_classifier, TensorFlowV2Classifier):
            initializer = self._initialize_poison_tensorflow
            poisoner = self._poison_tensorflow
            finish_poisoning = self._finish_poison_tensorflow
        else:
            raise NotImplementedError("WitchesBrewAttack is currently implemented only for PyTorch and TensorFlow V2.")

        # Choose samples to poison.
        x_train = np.copy(x_train)
        y_train = np.copy(y_train)
        if len(np.shape(y_trigger)) == 2:  # dense labels
            classes_target = set(np.argmax(y_trigger, axis=-1))
        else:  # sparse labels
            classes_target = set(y_trigger)
        num_poison_samples = int(self.percent_poison * len(x_train))

        # Try poisoning num_trials times and choose the best one.
        best_B = np.finfo(np.float32).max  # pylint: disable=C0103
        best_x_poisoned = None
        best_indices_poison = None

        if len(np.shape(y_train)) == 2:
            y_train_classes = np.argmax(y_train, axis=-1)
        else:
            y_train_classes = y_train
        for _ in trange(self.max_trials):
            indices_poison = np.random.permutation(np.where([y in classes_target for y in y_train_classes])[0])[
                :num_poison_samples
            ]
            x_poison = x_train[indices_poison]
            y_poison = y_train[indices_poison]
            initializer(x_trigger, y_trigger, x_poison, y_poison)
            x_poisoned, B_ = poisoner(x_poison, y_poison)  # pylint: disable=C0103
            finish_poisoning()
            B_ = np.mean(B_)  # Averaging B losses from multiple batches.  # pylint: disable=C0103
            if B_ < best_B:
                best_B = B_  # pylint: disable=C0103
                best_x_poisoned = x_poisoned
                best_indices_poison = indices_poison

        if self.verbose > 0:
            print("Best B-score:", best_B)
        x_train[best_indices_poison] = best_x_poisoned
        return x_train, y_train  # y_train has not been modified.

    def _check_params(self) -> None:
        if not isinstance(self.learning_rate_schedule, tuple) or len(self.learning_rate_schedule) != 2:
            raise ValueError("learning_rate_schedule must be a pair of a list of learning rates and a list of epochs")

        if self.percent_poison > 1 or self.percent_poison < 0:
            raise ValueError("percent_poison must be in [0, 1]")

        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")

        if self.max_trials < 1:
            raise ValueError("max_trials must be positive")

        if not isinstance(self.clip_values, tuple) or len(self.clip_values) != 2:
            raise ValueError("clip_values must be a pair (min, max) of floats")

        if self.epsilon <= 0:
            raise ValueError("epsilon must be nonnegative")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if (
            isinstance(self.verbose, int)
            and self.verbose < 0
            or not isinstance(self.verbose, int)
            and not isinstance(self.verbose, bool)
        ):
            raise ValueError("verbose must be nonnegative integer or Boolean")


# Type alias for backwards compatibility
GradientMatchingAttack = WitchesBrewAttack
