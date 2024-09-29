from typing import Any

import torch

import ans


class LinearSoftmaxModel:

    def __init__(self, in_size: int, out_size: int, weight_scale: float = 1e-3) -> None:
        ########################################
        # TODO: implement

        self.weight = ...
        self.bias = ...

        ########################################

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 1e-3
    ) -> tuple[float, torch.Tensor]:
        """
        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
            learning_rate: gradient descent step size
        Returns:
            loss: average loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return loss, logits

    def val_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
        """
        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
        Returns:
            loss: average loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################
        
        return loss, logits


class LinearSVMModel(LinearSoftmaxModel):
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 1e-3
    ) -> tuple[float, torch.Tensor]:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return loss, logits

    def val_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################
        
        return loss, logits


def accuracy(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Args:
        scores: output linear scores (logits or probabilities); shape (num_samples, num_classes)
        targets: vector of class indicies (integers); shape (num_samples,)
    Returns:
        acc: averare accuracy on the batch; tensor containing single number (scalar), e.g. "tensor(0.364)"
    """
    
    ########################################
    # TODO: implement
    
    raise NotImplementedError
    
    # ENDTODO
    ########################################
    
    return acc


def train_epoch(
    model: LinearSoftmaxModel,
    loader: ans.data.BatchLoader,
    **train_step_kwargs
) -> Any:
    """
    Trains `model` on the dataset `loader` for one epoch.

    Args:
        model: Model to train. Must have method `train_step`.
        loader: loader of the training dataset
    Returns:
        return whatever is needed
    """
    ########################################
    # TODO: implement

    raise NotImplementedError

    # ENDTODO
    ########################################


def validate(
    model: LinearSoftmaxModel,
    loader: ans.data.BatchLoader,
) -> tuple[float, float]:
    """
    Validates `model` on the dataset `loader`.

    Args:
        model: Model to be validated. Must have method `val_step`.
        loader: loader of the training dataset
    Returns:
        mean_loss: average loss achieved on the dataset during training
        mean_acc: average accuracy achieved on the dataset during model training
    """
    ########################################
    # TODO: implement

    raise NotImplementedError

    # ENDTODO
    ########################################
