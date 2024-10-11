from numbers import Number
from typing import Callable, Optional, Self, TypeAlias, Union

import torch


# Variable can be added, subtracted, multiplied, and divided with the following
BinOpOtherType: TypeAlias = Union[Number, torch.Tensor, Self]


class Variable:

    def __init__(
            self,
            data: torch.Tensor,
            parents: tuple[Self, ...] = (),
            grad_fn: Optional[Callable[[torch.Tensor], tuple[torch.Tensor, ...]]] = None,
            name: Optional[str] = None
    ) -> None:
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        self.data = data
        self.grad: Optional[torch.Tensor] = None
        self.parents = parents
        self.grad_fn = grad_fn
        self.name = name

    def __repr__(self):
        if hasattr(self.grad_fn, 'func'):
            grad_fn_repr = self.grad_fn.func.__qualname__
        elif self.grad_fn is not None:
            grad_fn_repr = self.grad_fn.__qualname__
        else:
            grad_fn_repr = 'None'
        if self.data.ndim == 0 or (self.data.shape[-1] == self.data.numel() and self.data.numel() < 5):
            return f"{self.__class__.__name__}({self.data}, grad_fn={grad_fn_repr})"
        else:
            return f"{self.__class__.__name__}(shape={tuple(self.data.shape)}, grad_fn={grad_fn_repr})"

    def __add__(self, other: BinOpOtherType) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################
    
    def __radd__(self, other: BinOpOtherType) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

    def __sub__(self, other: BinOpOtherType) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################
    
    def __rsub__(self, other: BinOpOtherType) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

    def __mul__(self, other: BinOpOtherType) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################
    
    def __rmul__(self, other: BinOpOtherType) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################
    
    def __truediv__(self, other: BinOpOtherType) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################
    
    def __rtruediv__(self, other: BinOpOtherType) -> Self:
        ########################################
        # TODO: implement
        
        raise NotImplementedError

        # ENDTODO
        ########################################

    def backprop(self, dout: Optional[torch.Tensor] = None) -> None:
        """
        Runs full backpropagation starting from self. Fills the grad attribute with dself/dpredecessor for all
        predecessors of self.

        Args:
            dout: Incoming gradient on self; if None, then set to tensor of ones with proper shape and dtype
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################
