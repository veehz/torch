# This is a very rough draft.

# assume js_torch is available
from pyodide.ffi import JsProxy

class Tensor:
    # dealing with when .grad is None
    def __new__(cls, data, requires_grad=False):
        if(data is None):
            return None
        return super().__new__(cls)

    def __init__(self, data, requires_grad=False):
        # print(type(data))
        # self.data = js_torch.tensor(data)
        if isinstance(data, JsProxy):
            self.data = data
        else:
            print("creating tensor from data", data)
            self.data = torch_utils.create_tensor_from_python_data(data, requires_grad)
            print("created tensor", self.tolist())
            # print("created tensor", self.data._data)

    def __repr__(self):
        return f"Tensor(data={self.tolist()}{', requires_grad=True' if self.get_tensor().requires_grad else ''})"

    def tolist(self):
        return torch_utils.get_data_from_tensor(self.data).to_py()

    def get_tensor(self):
        return self.data

    def backward(self):
        self.get_tensor().backward()

    @property
    def grad(self):
        return Tensor(self.get_tensor().grad)

    @property
    def shape(self):
        return self.get_tensor().shape.to_py()

    def item(self):
        return self.get_tensor().item()

    def sum(self, dim=-1, keepdim=False):
        return Tensor(self.get_tensor().sum(dim, keepdim))

    def zero_grad(self):
        self.get_tensor().zero_grad()

    def __try_op__(self, op, other, func):
        other_data = other.get_tensor() if isinstance(other, Tensor) else other
        return Tensor(func(self.get_tensor(), other_data))

    def __add__(self, other):
        return self.__try_op__("add", other, lambda a, b: a.add(b))

    def __radd__(self, other):
        return self.__try_op__("radd", other, lambda a, b: a.add(b))

    def __mul__(self, other):
        return self.__try_op__("mul", other, lambda a, b: a.mul(b))

    def __rmul__(self, other):
        return self.__try_op__("rmul", other, lambda a, b: a.mul(b))

    def __truediv__(self, other):
        return self.__try_op__("truediv", other, lambda a, b: a.div(b))

    def __sub__(self, other):
        return self.__try_op__("sub", other, lambda a, b: a.sub(b))

    def __pow__(self, other):
        return self.__try_op__("pow", other, lambda a, b: a.pow(b))

    def __matmul__(self, other):
        return self.__try_op__("matmul", other, lambda a, b: a.matmul(b))

    # For when doing e.g. {a:.5f}
    def __format__(self, format_spec):
        return format(self.item(), format_spec)

    def reshape(self, *args):
        return Tensor(self.get_tensor().reshape(
            torch_utils.to_js_list(args)
        ))
    
    def allclose(self, other, rtol=1e-5, atol=1e-8, equal_nan=False):
        return self.get_tensor().allclose(other.get_tensor(), rtol, atol, equal_nan)

    # Make iterable so that can unpack
    def __iter__(self):
        return iter(self.tolist())

    def __getattr__(self, name):
        return lambda *args, **kwargs: Tensor(self.get_tensor().__getattribute__(name)(
            *_transform_args(args),
            *_transform_args(kwargs)
        ))

class _NoGrad:
    def __enter__(self):
        self.enabled = js_torch.getGradientTrackingEnabled()
        js_torch.setGradientTrackingEnabled(False)

    def __exit__(self, type, value, traceback):
        js_torch.setGradientTrackingEnabled(self.enabled)

class _Torch:
    _operations = [
        "sin", "mean", "square", "abs", "relu", "ones_like", "linspace", "ones",
    ]

    no_grad = _NoGrad

    @property
    def tensor(self):
        return Tensor

    def __getattr__(self, name):
        return lambda *args, **kwargs: Tensor(js_torch.__getattribute__(name)(
            *_transform_args(args),
            *_transform_args(kwargs)
        ))

    def Size(self, shape):
        return shape

    def cat(self, tensors, dim=0):
        return Tensor(js_torch.cat(torch_utils.to_js_list(list(map(lambda x: x.get_tensor(), tensors))), dim))


# torch = lambda: None
torch = _Torch()