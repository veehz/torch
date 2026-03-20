const _torch = window.torch;

export const {
  Tensor,
  no_grad,
  enable_no_grad,
  disable_no_grad,
  is_grad_enabled,
  TorchFunction,
  AccumulateGrad,
  nn,
  optim,
  eventBus,
  events,
  export_,
  ExportedProgram,
  add,
  matmul,
  relu,
  mul,
  randn,
  allclose,
  __right_index__,
  __left_index__,
  // etc...
} = window.torch;
