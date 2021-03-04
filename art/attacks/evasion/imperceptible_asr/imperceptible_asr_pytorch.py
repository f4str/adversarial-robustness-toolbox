# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
This module implements the imperceptible, robust, and targeted attack to generate adversarial examples for automatic
speech recognition models. This attack will be implemented specifically for DeepSpeech model and is framework dependent,
specifically for PyTorch.

| Paper link: https://arxiv.org/abs/1903.10346
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import scipy

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin, NeuralNetworkMixin
from art.estimators.pytorch import PyTorchEstimator
from art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class ImperceptibleASRPyTorch(EvasionAttack):
    """
    This class implements the imperceptible, robust, and targeted attack to generate adversarial examples for automatic
    speech recognition models. This attack will be implemented specifically for DeepSpeech model and is framework
    dependent, specifically for PyTorch.

    | Paper link: https://arxiv.org/abs/1903.10346
    """

    attack_params = EvasionAttack.attack_params + [
        "initial_eps",
        "max_iter_1st_stage",
        "max_iter_2nd_stage",
        "learning_rate_1st_stage",
        "learning_rate_2nd_stage",
        "optimizer_1st_stage",
        "optimizer_2nd_stage",
        "global_max_length",
        "initial_rescale",
        "rescale_factor",
        "num_iter_adjust_rescale",
        "initial_alpha",
        "increase_factor_alpha",
        "num_iter_increase_alpha",
        "decrease_factor_alpha",
        "num_iter_decrease_alpha",
        "batch_size",
        "use_amp",
        "opt_level",
    ]

    _estimator_requirements = (
        BaseEstimator,
        LossGradientsMixin,
        NeuralNetworkMixin,
        SpeechRecognizerMixin,
        PyTorchEstimator,
        PyTorchDeepSpeech,
    )

    def __init__(
        self,
        estimator: PyTorchDeepSpeech,
        initial_eps: float = 0.001,
        max_iter_1st_stage: int = 1000,
        max_iter_2nd_stage: int = 4000,
        learning_rate_1st_stage: float = 0.1,
        learning_rate_2nd_stage: float = 0.001,
        optimizer_1st_stage: Optional["torch.optim.Optimizer"] = None,
        optimizer_2nd_stage: Optional["torch.optim.Optimizer"] = None,
        global_max_length: int = 10000,
        initial_rescale: float = 1.0,
        rescale_factor: float = 0.8,
        num_iter_adjust_rescale: int = 10,
        initial_alpha: float = 0.05,
        increase_factor_alpha: float = 1.2,
        num_iter_increase_alpha: int = 20,
        decrease_factor_alpha: float = 0.8,
        num_iter_decrease_alpha: int = 50,
        batch_size: int = 32,
        use_amp: bool = False,
        opt_level: str = "O1",
    ):
        """
        Create a :class:`.ImperceptibleASRPyTorch` instance.

        :param estimator: A trained estimator.
        :param initial_eps: Initial maximum perturbation that the attacker can introduce.
        :param max_iter_1st_stage: The maximum number of iterations applied for the first stage of the optimization of
                                   the attack.
        :param max_iter_2nd_stage: The maximum number of iterations applied for the second stage of the optimization of
                                   the attack.
        :param learning_rate_1st_stage: The initial learning rate applied for the first stage of the optimization of
                                        the attack.
        :param learning_rate_2nd_stage: The initial learning rate applied for the second stage of the optimization of
                                        the attack.
        :param optimizer_1st_stage: The optimizer applied for the first stage of the optimization of the attack. If
                                    `None` attack will use `torch.optim.SGD`.
        :param optimizer_2nd_stage: The optimizer applied for the second stage of the optimization of the attack. If
                                    `None` attack will use `torch.optim.SGD`.
        :param global_max_length: The length of the longest audio signal allowed by this attack.
        :param initial_rescale: Initial rescale coefficient to speedup the decrease of the perturbation size during
                                the first stage of the optimization of the attack.
        :param rescale_factor: The factor to adjust the rescale coefficient during the first stage of the optimization
                               of the attack.
        :param num_iter_adjust_rescale: Number of iterations to adjust the rescale coefficient.
        :param initial_alpha: The initial value of the alpha coefficient used in the second stage of the optimization
                              of the attack.
        :param increase_factor_alpha: The factor to increase the alpha coefficient used in the second stage of the
                                      optimization of the attack.
        :param num_iter_increase_alpha: Number of iterations to increase alpha.
        :param decrease_factor_alpha: The factor to decrease the alpha coefficient used in the second stage of the
                                      optimization of the attack.
        :param num_iter_decrease_alpha: Number of iterations to decrease alpha.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param use_amp: Whether to use the automatic mixed precision tool to enable mixed precision training or
                        gradient computation, e.g. with loss gradient computation. When set to True, this option is
                        only triggered if there are GPUs available.
        :param opt_level: Specify a pure or mixed precision optimization level. Used when use_amp is True. Accepted
                          values are `O0`, `O1`, `O2`, and `O3`.
        """
        import torch  # lgtm [py/repeated-import]
        from torch.autograd import Variable

        super().__init__(estimator=estimator)

        # Set attack attributes
        self._targeted = True
        self.initial_eps = initial_eps
        self.max_iter_1st_stage = max_iter_1st_stage
        self.max_iter_2nd_stage = max_iter_2nd_stage
        self.learning_rate_1st_stage = learning_rate_1st_stage
        self.learning_rate_2nd_stage = learning_rate_2nd_stage
        self.global_max_length = global_max_length
        self.initial_rescale = initial_rescale
        self.rescale_factor = rescale_factor
        self.num_iter_adjust_rescale = num_iter_adjust_rescale
        self.initial_alpha = initial_alpha
        self.increase_factor_alpha = increase_factor_alpha
        self.num_iter_increase_alpha = num_iter_increase_alpha
        self.decrease_factor_alpha = decrease_factor_alpha
        self.num_iter_decrease_alpha = num_iter_decrease_alpha
        self.batch_size = batch_size
        self._use_amp = use_amp

        # Create the main variable to optimize
        if self.estimator.device.type == "cpu":
            self.global_optimal_delta = Variable(
                torch.zeros(self.batch_size, self.global_max_length).type(torch.FloatTensor), requires_grad=True
            )
        else:
            self.global_optimal_delta = Variable(
                torch.zeros(self.batch_size, self.global_max_length).type(torch.cuda.FloatTensor), requires_grad=True
            )

        self.global_optimal_delta.to(self.estimator.device)

        # Create the optimizers
        self._optimizer_1st_stage_arg = optimizer_1st_stage
        if optimizer_1st_stage is None:
            self.optimizer_1st_stage = torch.optim.SGD(
                params=[self.global_optimal_delta], lr=self.learning_rate_1st_stage
            )

        self._optimizer_2nd_stage_arg = optimizer_2nd_stage
        if optimizer_2nd_stage is None:
            self.optimizer_2nd_stage = torch.optim.SGD(
                params=[self.global_optimal_delta], lr=self.learning_rate_2nd_stage
            )

        # Setup for AMP use
        if self._use_amp:
            from apex import amp

            if self.estimator.device.type == "cpu":
                enabled = False
            else:
                enabled = True

            self.estimator._model, [self.optimizer_1st_stage, self.optimizer_2nd_stage] = amp.initialize(
                models=self.estimator._model,
                optimizers=[self.optimizer_1st_stage, self.optimizer_2nd_stage],
                enabled=enabled,
                opt_level=opt_level,
                loss_scale=1.0,
            )

        # Check validity of attack attributes
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`. Note that, this
                  class only supports targeted attack.
        :return: An array holding the adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        if y is None:
            raise ValueError(
                "`ImperceptibleASRPyTorch` is a targeted attack and requires the definition of target"
                "labels `y`. Currently `y` is set to `None`."
            )

        # Start to compute adversarial examples
        dtype = x.dtype

        # Cast to type float64 to avoid overflow
        if dtype.type == np.float64:
            adv_x = x.copy()
        else:
            adv_x = x.copy().astype(np.float64)

        # Put the estimator in the training mode, otherwise CUDA can't backpropagate through the model.
        # However, estimator uses batch norm layers which need to be frozen
        self.estimator.model.train()
        self.estimator.set_batchnorm(train=False)

        # Compute perturbation with batching
        num_batch = int(np.ceil(len(x) / float(self.batch_size)))

        for m in range(num_batch):
            # Batch indexes
            batch_index_1, batch_index_2 = (
                m * self.batch_size,
                min((m + 1) * self.batch_size, len(x)),
            )

            # First reset delta
            self.global_optimal_delta.data = torch.zeros(self.batch_size, self.global_max_length).type(torch.float64)

            # Next, reset optimizers
            if self._optimizer_1st_stage_arg is not None:
                self.optimizer_1st_stage = self._optimizer_1st_stage_arg(
                    params=[self.global_optimal_delta], lr=self.learning_rate_1st_stage
                )
            if self._optimizer_2nd_stage_arg is not None:
                self.optimizer_2nd_stage = self._optimizer_2nd_stage_arg(
                    params=[self.global_optimal_delta], lr=self.learning_rate_2nd_stage
                )

            # Then compute the batch
            adv_x_batch = self._generate_batch(adv_x[batch_index_1:batch_index_2], y[batch_index_1:batch_index_2])

            for i in range(len(adv_x_batch)):
                adv_x[batch_index_1 + i] = adv_x_batch[i, : len(adv_x[batch_index_1 + i])]

        # Unfreeze batch norm layers again
        self.estimator.set_batchnorm(train=True)

        # Recast to the original type if needed
        if dtype.type == np.float32:
            adv_x = adv_x.astype(dtype)

        return adv_x

    def _generate_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate a batch of adversarial samples and return them in an array.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`. Note that, this
                  class only supports targeted attack.
        :return: A batch of adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        # First stage of attack
        successful_adv_input_1st_stage, original_input = self._attack_1st_stage(x=x, y=y)
        successful_perturbation_1st_stage = successful_adv_input_1st_stage - torch.tensor(original_input).to(
            self.estimator.device
        )

        # Compute original masking threshold and maximum psd
        theta_batch = []
        original_max_psd_batch = []

        for i in range(len(x)):
            theta, original_max_psd = self._compute_masking_threshold(original_input[i])
            theta = theta.transpose(1, 0)
            theta_batch.append(theta)
            original_max_psd_batch.append(original_max_psd)

        theta_batch = np.array(theta_batch)
        original_max_psd_batch = np.array(original_max_psd_batch)

        # Reset delta with new result
        local_batch_shape = successful_adv_input_1st_stage.shape
        self.global_optimal_delta.data = torch.zeros(self.batch_size, self.global_max_length).type(torch.float64)
        self.global_optimal_delta.data[
            : local_batch_shape[0], : local_batch_shape[1]
        ] = successful_perturbation_1st_stage

        # Second stage of attack
        successful_adv_input_2nd_stage = self._attack_2nd_stage(
            x=x, y=y, theta_batch=theta_batch, original_max_psd_batch=original_max_psd_batch
        )

        results = successful_adv_input_2nd_stage.detach().cpu().numpy()

        return results

    def _attack_1st_stage(self, x: np.ndarray, y: np.ndarray) -> Tuple["torch.Tensor", np.ndarray]:
        """
        The first stage of the attack.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`. Note that, this
                  class only supports targeted attack.
        :return: A tuple of two tensors:
                    - A tensor holding the candidate adversarial examples.
                    - An array holding the original inputs.
        """
        import torch  # lgtm [py/repeated-import]

        # Compute local shape
        local_batch_size = len(x)
        real_lengths = np.array([x_.shape[0] for x_ in x])
        local_max_length = np.max(real_lengths)

        # Initialize rescale
        rescale = np.ones([local_batch_size, local_max_length], dtype=np.float64) * self.initial_rescale

        # Reformat input
        input_mask = np.zeros([local_batch_size, local_max_length], dtype=np.float64)
        original_input = np.zeros([local_batch_size, local_max_length], dtype=np.float64)

        for local_batch_size_idx in range(local_batch_size):
            input_mask[local_batch_size_idx, : len(x[local_batch_size_idx])] = 1
            original_input[local_batch_size_idx, : len(x[local_batch_size_idx])] = x[local_batch_size_idx]

        # Optimization loop
        successful_adv_input = [None] * local_batch_size
        trans = [None] * local_batch_size

        for iter_1st_stage_idx in range(self.max_iter_1st_stage):
            # Zero the parameter gradients
            self.optimizer_1st_stage.zero_grad()

            # Call to forward pass
            loss, local_delta, decoded_output, masked_adv_input, _ = self._forward_1st_stage(
                original_input=original_input,
                original_output=y,
                local_batch_size=local_batch_size,
                local_max_length=local_max_length,
                rescale=rescale,
                input_mask=input_mask,
                real_lengths=real_lengths,
            )

            # Actual training
            if self._use_amp:
                from apex import amp

                with amp.scale_loss(loss, self.optimizer_1st_stage) as scaled_loss:
                    scaled_loss.backward()

            else:
                loss.backward()

            # Get sign of the gradients
            self.global_optimal_delta.grad = torch.sign(self.global_optimal_delta.grad)

            # Do optimization
            self.optimizer_1st_stage.step()

            # Save the best adversarial example and adjust the rescale coefficient if successful
            if iter_1st_stage_idx % self.num_iter_adjust_rescale == 0:
                for local_batch_size_idx in range(local_batch_size):
                    if decoded_output[local_batch_size_idx] == y[local_batch_size_idx]:
                        # Adjust the rescale coefficient
                        max_local_delta = np.max(np.abs(local_delta[local_batch_size_idx].detach().numpy()))

                        if rescale[local_batch_size_idx][0] * self.initial_eps > max_local_delta:
                            rescale[local_batch_size_idx] = max_local_delta / self.initial_eps
                        rescale[local_batch_size_idx] *= self.rescale_factor

                        # Save the best adversarial example
                        successful_adv_input[local_batch_size_idx] = masked_adv_input[local_batch_size_idx]
                        trans[local_batch_size_idx] = decoded_output[local_batch_size_idx]

            # If attack is unsuccessful
            if iter_1st_stage_idx == self.max_iter_1st_stage - 1:
                for local_batch_size_idx in range(local_batch_size):
                    if successful_adv_input[local_batch_size_idx] is None:
                        successful_adv_input[local_batch_size_idx] = masked_adv_input[local_batch_size_idx]
                        trans[local_batch_size_idx] = decoded_output[local_batch_size_idx]

        result = torch.stack(successful_adv_input)

        return result, original_input

    def _forward_1st_stage(
        self,
        original_input: np.ndarray,
        original_output: np.ndarray,
        local_batch_size: int,
        local_max_length: int,
        rescale: np.ndarray,
        input_mask: np.ndarray,
        real_lengths: np.ndarray,
    ) -> Tuple["torch.Tensor", "torch.Tensor", np.ndarray, "torch.Tensor", "torch.Tensor"]:
        """
        The forward pass of the first stage of the attack.

        :param original_input: Samples of shape (nb_samples, seq_length). Note that, sequences in the batch must have
                               equal lengths. A possible example of `original_input` could be:
                               `original_input = np.array([np.array([0.1, 0.2, 0.1]), np.array([0.3, 0.1, 0.0])])`.
        :param original_output: Target values of shape (nb_samples). Each sample in `original_output` is a string and
                                it may possess different lengths. A possible example of `original_output` could be:
                                `original_output = np.array(['SIXTY ONE', 'HELLO'])`.
        :param local_batch_size: Current batch size.
        :param local_max_length: Max length of the current batch.
        :param rescale: Current rescale coefficients.
        :param input_mask: Masks of true inputs.
        :param real_lengths: Real lengths of original sequences.
        :return: A tuple of (loss, local_delta, decoded_output, masked_adv_input)
                    - loss: The loss tensor of the first stage of the attack.
                    - local_delta: The delta of the current batch.
                    - decoded_output: Transcription output.
                    - masked_adv_input: Perturbed inputs.
        """
        import torch  # lgtm [py/repeated-import]
        from warpctc_pytorch import CTCLoss

        # Compute perturbed inputs
        local_delta = self.global_optimal_delta[:local_batch_size, :local_max_length]
        local_delta_rescale = torch.clamp(local_delta, -self.initial_eps, self.initial_eps).to(self.estimator.device)
        local_delta_rescale *= torch.tensor(rescale).to(self.estimator.device)
        adv_input = local_delta_rescale + torch.tensor(original_input).to(self.estimator.device)
        masked_adv_input = adv_input * torch.tensor(input_mask).to(self.estimator.device)

        # Transform data into the model input space
        inputs, targets, input_rates, target_sizes, batch_idx = self.estimator.preprocess_transform_model_input(
            x=masked_adv_input.to(self.estimator.device),
            y=original_output,
            real_lengths=real_lengths,
        )

        # Compute real input sizes
        input_sizes = input_rates.mul_(inputs.size()[-1]).int()

        # Call to DeepSpeech model for prediction
        outputs, output_sizes = self.estimator.model(
            inputs.to(self.estimator.device), input_sizes.to(self.estimator.device)
        )
        outputs_ = outputs.transpose(0, 1)
        float_outputs = outputs_.float()

        # Loss function
        criterion = CTCLoss()
        loss = criterion(float_outputs, targets, output_sizes, target_sizes).to(self.estimator.device)
        loss = loss / inputs.size(0)

        # Compute transcription
        decoded_output, _ = self.estimator.decoder.decode(outputs, output_sizes)
        decoded_output = [do[0] for do in decoded_output]
        decoded_output = np.array(decoded_output)

        # Rearrange to the original order
        decoded_output_ = decoded_output.copy()
        decoded_output[batch_idx] = decoded_output_

        return loss, local_delta, decoded_output, masked_adv_input, local_delta_rescale

    def _attack_2nd_stage(
        self, x: np.ndarray, y: np.ndarray, theta_batch: np.ndarray, original_max_psd_batch: np.ndarray
    ) -> "torch.Tensor":
        """
        The second stage of the attack.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`. Note that, this
                  class only supports targeted attack.
        :param theta_batch: Original thresholds.
        :param original_max_psd_batch: Original maximum psd.
        :return: An array holding the candidate adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        # Compute local shape
        local_batch_size = len(x)
        real_lengths = np.array([x_.shape[0] for x_ in x])
        local_max_length = np.max(real_lengths)

        # Initialize alpha and rescale
        alpha = np.array([self.initial_alpha] * local_batch_size, dtype=np.float64)
        rescale = np.ones([local_batch_size, local_max_length], dtype=np.float64) * self.initial_rescale

        # Reformat input
        input_mask = np.zeros([local_batch_size, local_max_length], dtype=np.float64)
        original_input = np.zeros([local_batch_size, local_max_length], dtype=np.float64)

        for local_batch_size_idx in range(local_batch_size):
            input_mask[local_batch_size_idx, : len(x[local_batch_size_idx])] = 1
            original_input[local_batch_size_idx, : len(x[local_batch_size_idx])] = x[local_batch_size_idx]

        # Optimization loop
        successful_adv_input = [None] * local_batch_size
        best_loss_2nd_stage = [np.inf] * local_batch_size
        trans = [None] * local_batch_size

        for iter_2nd_stage_idx in range(self.max_iter_2nd_stage):
            # Zero the parameter gradients
            self.optimizer_2nd_stage.zero_grad()

            # Call to forward pass of the first stage
            loss_1st_stage, _, decoded_output, masked_adv_input, local_delta_rescale = self._forward_1st_stage(
                original_input=original_input,
                original_output=y,
                local_batch_size=local_batch_size,
                local_max_length=local_max_length,
                rescale=rescale,
                input_mask=input_mask,
                real_lengths=real_lengths,
            )

            # Call to forward pass of the first stage
            loss_2nd_stage = self._forward_2nd_stage(
                local_delta_rescale=local_delta_rescale,
                theta_batch=theta_batch,
                original_max_psd_batch=original_max_psd_batch,
            )

            # Total loss
            loss = loss_1st_stage + torch.tensor(alpha).to(self.estimator.device) * loss_2nd_stage
            loss = torch.mean(loss)

            # Actual training
            if self._use_amp:
                from apex import amp

                with amp.scale_loss(loss, self.optimizer_2nd_stage) as scaled_loss:
                    scaled_loss.backward()

            else:
                loss.backward()

            # Do optimization
            self.optimizer_2nd_stage.step()

            # Save the best adversarial example and adjust the alpha coefficient
            for local_batch_size_idx in range(local_batch_size):
                if decoded_output[local_batch_size_idx] == y[local_batch_size_idx]:
                    if loss_2nd_stage[local_batch_size_idx] < best_loss_2nd_stage[local_batch_size_idx]:
                        # Update best loss at 2nd stage
                        best_loss_2nd_stage[local_batch_size_idx] = loss_2nd_stage[local_batch_size_idx]

                        # Save the best adversarial example
                        successful_adv_input[local_batch_size_idx] = masked_adv_input[local_batch_size_idx]
                        trans[local_batch_size_idx] = decoded_output[local_batch_size_idx]

                    # Adjust to increase the alpha coefficient
                    if iter_2nd_stage_idx % self.num_iter_increase_alpha == 0:
                        alpha[local_batch_size_idx] *= self.increase_factor_alpha

                # Adjust to decrease the alpha coefficient
                elif iter_2nd_stage_idx % self.num_iter_decrease_alpha == 0:
                    alpha[local_batch_size_idx] *= self.decrease_factor_alpha
                    alpha[local_batch_size_idx] = max(alpha[local_batch_size_idx], 0.0005)

            # If attack is unsuccessful
            if iter_2nd_stage_idx == self.max_iter_2nd_stage - 1:
                for local_batch_size_idx in range(local_batch_size):
                    if successful_adv_input[local_batch_size_idx] is None:
                        successful_adv_input[local_batch_size_idx] = masked_adv_input[local_batch_size_idx]
                        trans[local_batch_size_idx] = decoded_output[local_batch_size_idx]

        result = torch.stack(successful_adv_input)

        return result

    def _forward_2nd_stage(
        self,
        local_delta_rescale: "torch.Tensor",
        theta_batch: np.ndarray,
        original_max_psd_batch: np.ndarray,
    ) -> "torch.Tensor":
        """
        The forward pass of the second stage of the attack.

        :param local_delta_rescale: Local delta after rescaled.
        :param theta_batch: Original thresholds.
        :param original_max_psd_batch: Original maximum psd.
        :return: The loss tensor of the second stage of the attack.
        """
        import torch  # lgtm [py/repeated-import]

        # Compute loss for masking threshold
        losses = []
        relu = torch.nn.ReLU()

        for i in range(len(theta_batch)):
            psd_transform_delta = self._psd_transform(
                delta=local_delta_rescale[i, :], original_max_psd=original_max_psd_batch[i]
            )

            loss = torch.mean(relu(psd_transform_delta - torch.tensor(theta_batch[i]).to(self.estimator.device)))
            losses.append(loss)

        losses = torch.stack(losses)

        return losses

    def _compute_masking_threshold(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the masking threshold and the maximum psd of the original audio.

        :param x: Samples of shape (seq_length,).
        :return: A tuple of the masking threshold and the maximum psd.
        """
        import librosa

        # First compute the psd matrix
        # These parameters are needed for the transformation
        sample_rate = self.estimator.model.audio_conf.sample_rate
        window_size = self.estimator.model.audio_conf.window_size
        window_stride = self.estimator.model.audio_conf.window_stride

        n_fft = int(sample_rate * window_size)
        hop_length = int(sample_rate * window_stride)
        win_length = n_fft

        window = self.estimator.model.audio_conf.window.value

        if window == "hamming":
            window_fn = scipy.signal.windows.hamming
        elif window == "hann":
            window_fn = scipy.signal.windows.hann
        elif window == "blackman":
            window_fn = scipy.signal.windows.blackman
        elif window == "bartlett":
            window_fn = scipy.signal.windows.bartlett
        else:
            raise NotImplementedError("Spectrogram window %s not supported." % window)

        transformed_x = librosa.core.stft(
            y=x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window_fn, center=False
        )
        transformed_x *= np.sqrt(8.0 / 3.0)

        psd = abs(transformed_x / win_length)
        original_max_psd = np.max(psd * psd)
        psd = 10 * np.log10(psd * psd + 10e-20)
        psd = 96 - np.max(psd) + psd

        # Compute freqs and barks
        freqs = librosa.core.fft_frequencies(sample_rate, win_length)
        barks = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan(pow(freqs / 7500.0, 2))

        # Compute quiet threshold
        ath = np.zeros(len(barks), dtype=np.float64) - np.inf
        bark_idx = np.argmax(barks > 1)
        ath[bark_idx:] = (
            3.64 * pow(freqs[bark_idx:] * 0.001, -0.8)
            - 6.5 * np.exp(-0.6 * pow(0.001 * freqs[bark_idx:] - 3.3, 2))
            + 0.001 * pow(0.001 * freqs[bark_idx:], 4)
            - 12
        )

        # Compute the global masking threshold theta
        theta = []

        for i in range(psd.shape[1]):
            # Compute masker index
            masker_idx = scipy.signal.argrelextrema(psd[:, i], np.greater)[0]

            if 0 in masker_idx:
                masker_idx = np.delete(masker_idx, 0)

            if len(psd[:, i]) - 1 in masker_idx:
                masker_idx = np.delete(masker_idx, len(psd[:, i]) - 1)

            barks_psd = np.zeros([len(masker_idx), 3], dtype=np.float64)
            barks_psd[:, 0] = barks[masker_idx]
            barks_psd[:, 1] = 10 * np.log10(
                pow(10, psd[:, i][masker_idx - 1] / 10.0)
                + pow(10, psd[:, i][masker_idx] / 10.0)
                + pow(10, psd[:, i][masker_idx + 1] / 10.0)
            )
            barks_psd[:, 2] = masker_idx

            for j in range(len(masker_idx)):
                if barks_psd.shape[0] <= j + 1:
                    break

                while barks_psd[j + 1, 0] - barks_psd[j, 0] < 0.5:
                    quiet_threshold = (
                        3.64 * pow(freqs[int(barks_psd[j, 2])] * 0.001, -0.8)
                        - 6.5 * np.exp(-0.6 * pow(0.001 * freqs[int(barks_psd[j, 2])] - 3.3, 2))
                        + 0.001 * pow(0.001 * freqs[int(barks_psd[j, 2])], 4)
                        - 12
                    )
                    if barks_psd[j, 1] < quiet_threshold:
                        barks_psd = np.delete(barks_psd, j, axis=0)

                    if barks_psd.shape[0] == j + 1:
                        break

                    if barks_psd[j, 1] < barks_psd[j + 1, 1]:
                        barks_psd = np.delete(barks_psd, j, axis=0)
                    else:
                        barks_psd = np.delete(barks_psd, j + 1, axis=0)

                    if barks_psd.shape[0] == j + 1:
                        break

            # Compute the global masking threshold
            delta = 1 * (-6.025 - 0.275 * barks_psd[:, 0])

            t_s = []

            for m in range(barks_psd.shape[0]):
                d_z = barks - barks_psd[m, 0]
                zero_idx = np.argmax(d_z > 0)
                s_f = np.zeros(len(d_z), dtype=np.float64)
                s_f[:zero_idx] = 27 * d_z[:zero_idx]
                s_f[zero_idx:] = (-27 + 0.37 * max(barks_psd[m, 1] - 40, 0)) * d_z[zero_idx:]
                t_s.append(barks_psd[m, 1] + delta[m] + s_f)

            t_s = np.array(t_s)

            theta.append(np.sum(pow(10, t_s / 10.0), axis=0) + pow(10, ath / 10.0))

        theta = np.array(theta)

        return theta, original_max_psd

    def _psd_transform(self, delta: "torch.Tensor", original_max_psd: "torch.Tensor") -> "torch.Tensor":
        """
        Compute the psd matrix of the perturbation.

        :param delta: The perturbation.
        :param original_max_psd: The maximum psd of the original audio.
        :return: The psd matrix.
        """
        import torch  # lgtm [py/repeated-import]

        # These parameters are needed for the transformation
        sample_rate = self.estimator.model.audio_conf.sample_rate
        window_size = self.estimator.model.audio_conf.window_size
        window_stride = self.estimator.model.audio_conf.window_stride

        n_fft = int(sample_rate * window_size)
        hop_length = int(sample_rate * window_stride)
        win_length = n_fft

        window = self.estimator.model.audio_conf.window.value

        if window == "hamming":
            window_fn = torch.hamming_window
        elif window == "hann":
            window_fn = torch.hann_window
        elif window == "blackman":
            window_fn = torch.blackman_window
        elif window == "bartlett":
            window_fn = torch.bartlett_window
        else:
            raise NotImplementedError("Spectrogram window %s not supported." % window)

        # Return STFT of delta
        delta_stft = torch.stft(
            delta,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=False,
            window=window_fn(win_length).to(self.estimator.device),
        ).to(self.estimator.device)

        # Take abs of complex STFT results
        transformed_delta = torch.sqrt(torch.sum(torch.square(delta_stft), -1))

        # Compute the psd matrix
        psd = (8.0 / 3.0) * transformed_delta / win_length
        psd = psd ** 2
        psd = (
            torch.pow(torch.tensor(10.0), torch.tensor(9.6)).to(self.estimator.device)
            / torch.reshape(torch.tensor(original_max_psd).to(self.estimator.device), [-1, 1, 1])
            * psd
        )

        return psd

    def _check_params(self) -> None:
        """
        Apply attack-specific checks.
        """
        if self.initial_eps <= 0:
            raise ValueError("The perturbation size `initial_eps` has to be positive.")

        if not isinstance(self.max_iter_1st_stage, int):
            raise ValueError("The maximum number of iterations must be of type int.")
        if not self.max_iter_1st_stage > 0:
            raise ValueError("The maximum number of iterations must be greater than 0.")

        if not isinstance(self.max_iter_2nd_stage, int):
            raise ValueError("The maximum number of iterations must be of type int.")
        if not self.max_iter_2nd_stage > 0:
            raise ValueError("The maximum number of iterations must be greater than 0.")

        if not isinstance(self.learning_rate_1st_stage, float):
            raise ValueError("The learning rate must be of type float.")
        if not self.learning_rate_1st_stage > 0.0:
            raise ValueError("The learning rate must be greater than 0.0.")

        if not isinstance(self.learning_rate_2nd_stage, float):
            raise ValueError("The learning rate must be of type float.")
        if not self.learning_rate_2nd_stage > 0.0:
            raise ValueError("The learning rate must be greater than 0.0.")

        if not isinstance(self.global_max_length, int):
            raise ValueError("The length of the longest audio signal must be of type int.")
        if not self.global_max_length > 0:
            raise ValueError("The length of the longest audio signal must be greater than 0.")

        if not isinstance(self.initial_rescale, float):
            raise ValueError("The initial rescale coefficient must be of type float.")
        if not self.initial_rescale > 0.0:
            raise ValueError("The initial rescale coefficient must be greater than 0.0.")

        if not isinstance(self.rescale_factor, float):
            raise ValueError("The rescale factor must be of type float.")
        if not self.rescale_factor > 0.0:
            raise ValueError("The rescale factor must be greater than 0.0.")

        if not isinstance(self.num_iter_adjust_rescale, int):
            raise ValueError("The number of iterations must be of type int.")
        if not self.num_iter_adjust_rescale > 0:
            raise ValueError("The number of iterations must be greater than 0.")

        if not isinstance(self.initial_alpha, float):
            raise ValueError("The initial alpha must be of type float.")
        if not self.initial_alpha > 0.0:
            raise ValueError("The initial alpha must be greater than 0.0.")

        if not isinstance(self.increase_factor_alpha, float):
            raise ValueError("The factor to increase alpha must be of type float.")
        if not self.increase_factor_alpha > 0.0:
            raise ValueError("The factor to increase alpha must be greater than 0.0.")

        if not isinstance(self.num_iter_increase_alpha, int):
            raise ValueError("The number of iterations must be of type int.")
        if not self.num_iter_increase_alpha > 0:
            raise ValueError("The number of iterations must be greater than 0.")

        if not isinstance(self.decrease_factor_alpha, float):
            raise ValueError("The factor to decrease alpha must be of type float.")
        if not self.decrease_factor_alpha > 0.0:
            raise ValueError("The factor to decrease alpha must be greater than 0.0.")

        if not isinstance(self.num_iter_decrease_alpha, int):
            raise ValueError("The number of iterations must be of type int.")
        if not self.num_iter_decrease_alpha > 0:
            raise ValueError("The number of iterations must be greater than 0.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")
