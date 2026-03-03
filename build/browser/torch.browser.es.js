var Qr = Object.defineProperty;
var o = (t, e) => Qr(t, "name", { value: e, configurable: !0 });
function K(t) {
  return Array.isArray(t[0]) ? t[0] : t;
}
o(K, "get_shape_from_args");
function lt(...t) {
  const e = K(t), r = new c(Array(e.reduce((s, n) => s * n, 1)).fill(Math.random()));
  return r.shape = e, r;
}
o(lt, "randn");
function Dr(...t) {
  const e = K(t), r = new c(Array(e.reduce((s, n) => s * n, 1)).fill(Math.random()));
  return r.shape = e, r;
}
o(Dr, "rand");
function ft(t, e, r) {
  const s = new c(
    Array(r.reduce((n, a) => n * a, 1)).fill(Math.floor(Math.random() * (e - t) + t))
  );
  return s.shape = r, s;
}
o(ft, "randint");
function Xr(...t) {
  const e = K(t), r = new c(Array(e.reduce((s, n) => s * n, 1)).fill(1));
  return r.shape = e, r;
}
o(Xr, "ones");
function Yr(...t) {
  const e = K(t), r = new c(Array(e.reduce((s, n) => s * n, 1)).fill(0));
  return r.shape = e, r;
}
o(Yr, "zeros");
function pt(t) {
  return Xr(t.shape);
}
o(pt, "ones_like");
function T(t) {
  return Yr(t.shape);
}
o(T, "zeros_like");
function gt(t, e, r) {
  const s = [], n = (e - t) / (r - 1);
  for (let a = 0; a < r - 1; a++)
    s.push(t + a * n);
  return s.push(e), new c(s);
}
o(gt, "linspace");
function wt(t, e = void 0, r = 1) {
  const s = [];
  for (let n = t; n < e; n += r)
    s.push(n);
  return new c(s);
}
o(wt, "arange");
let Zr = 0;
const Wr = /* @__PURE__ */ o(() => Zr++, "getNextId"), F = new EventTarget(), E = {
  TENSOR_BEFORE_BACKWARD: "tensor.beforeBackward",
  TENSOR_AFTER_BACKWARD: "tensor.afterBackward",
  OPERATION_BEFORE_FORWARD: "operation.beforeForward",
  OPERATION_AFTER_FORWARD: "operation.afterForward",
  OPERATION_BEFORE_BACKWARD: "operation.beforeBackward",
  OPERATION_AFTER_BACKWARD: "operation.afterBackward",
  OPERATION_BEFORE_ACCUMULATE_GRAD: "operation.beforeAccumulateGrad",
  OPERATION_AFTER_ACCUMULATE_GRAD: "operation.afterAccumulateGrad"
};
function Sr(...t) {
  for (const e of t)
    if (e instanceof c && e.requires_grad)
      return !0;
  return !1;
}
o(Sr, "resultRequiresGrad");
const Ie = class Ie {
  id = Wr();
  next_functions = [];
  saved_tensors = [];
  _retained_tensors = [];
  forward(...e) {
    const r = Sr(...e);
    F.dispatchEvent(new CustomEvent(E.OPERATION_BEFORE_FORWARD, {
      detail: {
        operation: this,
        requires_grad: r,
        args: e
      }
    }));
    const s = this._forward(...e);
    return F.dispatchEvent(new CustomEvent(E.OPERATION_AFTER_FORWARD, {
      detail: {
        operation: this,
        requires_grad: r,
        args: e,
        result: s
      }
    })), s;
  }
  backward(e) {
    F.dispatchEvent(new CustomEvent(E.OPERATION_BEFORE_BACKWARD, { detail: { operation: this, dz: e } }));
    for (const r of this._retained_tensors)
      r.grad || (r.grad = new c(new Array(r.dataLength()).fill(0))), r.grad = r.grad.add(e);
    this._backward(e), F.dispatchEvent(new CustomEvent(E.OPERATION_AFTER_BACKWARD, { detail: { operation: this, dz: e } }));
  }
};
o(Ie, "Operation");
let y = Ie;
const We = class We extends y {
  _forward(...e) {
    throw new Error("NullOp should not be called");
  }
  _backward(e) {
  }
};
o(We, "NullOp");
let G = We;
const h = new G(), Ne = class Ne extends y {
};
o(Ne, "UnaryOperation");
let w = Ne;
const Pe = class Pe extends y {
};
o(Pe, "BinaryOperation");
let g = Pe;
const je = class je extends w {
  variable;
  _forward(e) {
    return this.variable = e, e;
  }
  _backward(e) {
    this.variable.grad || (this.variable.grad = T(this.variable)), F.dispatchEvent(new CustomEvent(E.OPERATION_BEFORE_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } })), this.variable.grad = this.variable.grad.add(e), F.dispatchEvent(new CustomEvent(E.OPERATION_AFTER_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } }));
  }
};
o(je, "AccumulateGrad");
let U = je;
const Nr = /* @__PURE__ */ new Map(), V = /* @__PURE__ */ new Map();
function l(t, e) {
  Nr.set(t, e);
}
o(l, "registerOperation");
function R(t) {
  const e = Nr.get(t);
  if (!e)
    throw new Error(`Operation '${t}' is not registered.`);
  return e;
}
o(R, "getOperation");
function Ir(t) {
  const e = V.get(t);
  return e || (V.set(t, new (R(t))()), V.get(t));
}
o(Ir, "getOperationCache");
function Lr(t) {
  if (ArrayBuffer.isView(t))
    return [t.length];
  const e = [];
  for (; Array.isArray(t); )
    e.push(t.length), t = t[0];
  return e;
}
o(Lr, "_get_shape");
function Pr(t) {
  return Array.isArray(t) ? t.flatMap((e) => Pr(e)) : ArrayBuffer.isView(t) ? Array.from(t) : [t];
}
o(Pr, "_flatten");
const M = class M {
  id = Wr();
  data;
  _shape;
  grad_fn = null;
  grad = null;
  requires_grad;
  constructor(e, r = {}, s = {}) {
    if (this.data = Pr(e), this.requires_grad = r.requires_grad ?? !1, this._shape = s.shape ?? Lr(e), this.grad_fn = s.operation ?? null, this.requires_grad && !this.grad_fn) {
      const n = new U();
      n.variable = this, this.grad_fn = n;
    }
  }
  // TODO: Somehow having a shape of [] will have a weird error:
  // TypeError: Cannot read properties of undefined (reading 'length')
  // when running kernel (something to do with constants?)
  // so a little hack to return [1] when the shape is []
  get shape() {
    return this._shape.length === 0 ? [1] : this._shape;
  }
  toArray_() {
  }
  toArray() {
    return this.data;
  }
  dataLength() {
    return this.data.length;
  }
  set shape(e) {
    this._shape = e;
  }
  _executeUnaryOp(e) {
    return (this.requires_grad ? new (R(e))() : Ir(e)).forward(this);
  }
  _executeBinaryOp(e, r) {
    return typeof r == "number" && (r = new M(r)), (this.requires_grad || r.requires_grad ? new (R(e))() : Ir(e)).forward(this, r);
  }
  _executeOpRaw(e, ...r) {
    return new (R(e))().forward(this, ...r);
  }
  item() {
    if (this.dataLength() !== 1)
      throw new Error("Tensor.item() is only valid for scalars");
    return this.toArray()[0];
  }
  detach() {
    return new M(this.data, { requires_grad: !1 }, { shape: this.shape });
  }
  detach_() {
    this.requires_grad = !1, this.grad = null, this.grad_fn = null;
  }
  zero_() {
    this.data = Array(this.dataLength()).fill(0);
  }
  is_retain_grad = !1;
  retain_grad() {
    this.grad_fn instanceof U || this.is_retain_grad || (this.is_retain_grad = !0, this.grad_fn._retained_tensors.push(this));
  }
  backward(e) {
    if (this.requires_grad) {
      if (e)
        e.toArray_();
      else {
        if (this.dataLength() !== 1)
          throw new Error("Gradient is required for non-scalar tensors");
        e = new M(1);
      }
      this.grad_fn && (F.dispatchEvent(new CustomEvent(E.TENSOR_BEFORE_BACKWARD, { detail: { tensor: this } })), this.grad_fn.backward(e), F.dispatchEvent(new CustomEvent(E.TENSOR_AFTER_BACKWARD, { detail: { tensor: this } })));
    }
  }
  // operations
  // binary pointwise
  add(e) {
    return this._executeBinaryOp("add", e);
  }
  sub(e) {
    return this._executeBinaryOp("sub", e);
  }
  mul(e) {
    return this._executeBinaryOp("mul", e);
  }
  div(e) {
    return this._executeBinaryOp("div", e);
  }
  pow(e) {
    return typeof e == "number" && e % 1 === 0 ? this._executeOpRaw("powint", e) : this._executeBinaryOp("pow", e);
  }
  fmod(e) {
    return this._executeBinaryOp("fmod", e);
  }
  maximum(e) {
    return this._executeBinaryOp("maximum", e);
  }
  minimum(e) {
    return this._executeBinaryOp("minimum", e);
  }
  // unary pointwise
  log() {
    return this._executeUnaryOp("log");
  }
  sqrt() {
    return this._executeUnaryOp("sqrt");
  }
  exp() {
    return this._executeUnaryOp("exp");
  }
  square() {
    return this._executeUnaryOp("square");
  }
  abs() {
    return this._executeUnaryOp("abs");
  }
  sign() {
    return this._executeUnaryOp("sign");
  }
  neg() {
    return this._executeUnaryOp("neg");
  }
  reciprocal() {
    return this._executeUnaryOp("reciprocal");
  }
  reshape(e) {
    return this._executeOpRaw("reshape", e);
  }
  unsqueeze(e) {
    return this._executeOpRaw("unsqueeze", e);
  }
  // trigonometric
  sin() {
    return this._executeUnaryOp("sin");
  }
  cos() {
    return this._executeUnaryOp("cos");
  }
  tan() {
    return this._executeUnaryOp("tan");
  }
  // reduction
  sum() {
    return this._executeUnaryOp("sum");
  }
  mean() {
    return this._executeUnaryOp("mean");
  }
  // linalg
  transpose(e, r) {
    return this._executeOpRaw("transpose", e, r);
  }
  matmul(e) {
    return this._executeBinaryOp("matmul", e);
  }
  // comparison
  lt(e) {
    return this._executeBinaryOp("lt", e);
  }
  gt(e) {
    return this._executeBinaryOp("gt", e);
  }
  le(e) {
    return this._executeBinaryOp("le", e);
  }
  ge(e) {
    return this._executeBinaryOp("ge", e);
  }
  eq(e) {
    return this._executeBinaryOp("eq", e);
  }
  ne(e) {
    return this._executeBinaryOp("ne", e);
  }
};
o(M, "Tensor");
let c = M;
function x(t, e) {
  const r = Math.max(t.length, e.length), s = [...Array(r - t.length).fill(1), ...t], n = [...Array(r - e.length).fill(1), ...e], a = [];
  for (let d = 0; d < r; d++) {
    if (s[d] !== n[d] && s[d] !== 1 && n[d] !== 1)
      throw new Error(`Shape mismatch: ${t} and ${e}`);
    a.push(Math.max(s[d], n[d]));
  }
  return a;
}
o(x, "_broadcast_shape");
function f(t, e) {
  return t.length >= e.length ? t : [...Array(e.length - t.length).fill(1), ...t];
}
o(f, "_pad_shape");
function p(t, e, r) {
  let s = 0, n = 1, a = r;
  for (let d = t.length - 1; d >= 0; d--) {
    if (t[d] > 1) {
      const i = a % e[d];
      s = s + i * n;
    }
    n *= t[d], a = Math.floor(a / e[d]);
  }
  return s;
}
o(p, "_get_original_index");
function De(t) {
  return (...e) => new (R(t))().forward(...e);
}
o(De, "generate_function$1");
function q(t) {
  return (e) => (typeof e == "number" && (e = new c(e)), new (R(t))().forward(e));
}
o(q, "generate_unary_function$1");
function m(t) {
  return (e, r) => (typeof e == "number" && (e = new c(e)), typeof r == "number" && (r = new c(r)), new (R(t))().forward(e, r));
}
o(m, "generate_binary_function$1");
const xt = m("__left_index__"), mt = m("__right_index__"), qt = m("add"), kt = m("sub"), vt = m("mul"), yt = m("div"), At = m("pow"), Ot = m("fmod"), bt = m("maximum"), Ft = m("minimum"), Et = q("log"), Rt = q("sqrt"), Bt = q("exp"), Mt = q("square"), Tt = q("abs"), es = q("sign"), Ct = q("neg"), Ut = q("reciprocal"), zt = De("reshape"), Dt = De("unsqueeze"), It = q("sin"), Wt = q("cos"), Nt = q("tan"), Pt = q("sum"), jt = q("mean"), Kt = De("transpose"), $t = m("matmul"), Vt = m("lt"), Gt = m("gt"), Ht = m("le"), Jt = m("ge"), Qt = m("eq"), Xt = m("ne"), rs = /* @__PURE__ */ o(function(t, e, r, s, n, a) {
  const d = Array(a);
  for (let i = 0; i < a; i++) {
    const _ = p(e, n, i), u = p(s, n, i);
    d[i] = _;
  }
  return d;
}, "___left_index___kernel");
function ss(t, e, r = null) {
  const s = x(t.shape, e.shape), n = f(t.shape, s), a = f(e.shape, s), d = rs, i = s.reduce((_, u) => _ * u, 1);
  return new c(
    d(t.data, n, e.data, a, s, i),
    { requires_grad: t.requires_grad || e.requires_grad },
    { operation: r, shape: s }
  );
}
o(ss, "___left_index___tensor");
const Ke = class Ke extends g {
  _forward(e, r) {
    return (e.requires_grad || r.requires_grad) && (this.saved_tensors = [e, r]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), this.next_functions.push(r.grad_fn ? r.grad_fn : h), ss(e, r, e.requires_grad || r.requires_grad ? this : null);
  }
  _backward(e) {
    const [r, s] = this.saved_tensors, [n, a] = this.next_functions;
  }
};
o(Ke, "__Left_index__");
let H = Ke;
l("__left_index__", H);
const ts = /* @__PURE__ */ o(function(t, e, r, s, n, a) {
  const d = Array(a);
  for (let i = 0; i < a; i++) {
    const _ = p(e, n, i), u = p(s, n, i);
    d[i] = u;
  }
  return d;
}, "___right_index___kernel");
function ns(t, e, r = null) {
  const s = x(t.shape, e.shape), n = f(t.shape, s), a = f(e.shape, s), d = ts, i = s.reduce((_, u) => _ * u, 1);
  return new c(
    d(t.data, n, e.data, a, s, i),
    { requires_grad: t.requires_grad || e.requires_grad },
    { operation: r, shape: s }
  );
}
o(ns, "___right_index___tensor");
const $e = class $e extends g {
  _forward(e, r) {
    return (e.requires_grad || r.requires_grad) && (this.saved_tensors = [e, r]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), this.next_functions.push(r.grad_fn ? r.grad_fn : h), ns(e, r, e.requires_grad || r.requires_grad ? this : null);
  }
  _backward(e) {
    const [r, s] = this.saved_tensors, [n, a] = this.next_functions;
  }
};
o($e, "__Right_index__");
let J = $e;
l("__right_index__", J);
const as = /* @__PURE__ */ o(function(t, e, r, s, n, a) {
  const d = Array(a);
  for (let i = 0; i < a; i++) {
    const _ = p(e, n, i), u = p(s, n, i);
    d[i] = t[_] + r[u];
  }
  return d;
}, "_add_kernel");
function is(t, e, r = null) {
  const s = x(t.shape, e.shape), n = f(t.shape, s), a = f(e.shape, s), d = as, i = s.reduce((_, u) => _ * u, 1);
  return new c(
    d(t.data, n, e.data, a, s, i),
    { requires_grad: t.requires_grad || e.requires_grad },
    { operation: r, shape: s }
  );
}
o(is, "_add_tensor");
const Ve = class Ve extends g {
  _forward(e, r) {
    return (e.requires_grad || r.requires_grad) && (this.saved_tensors = [e, r]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), this.next_functions.push(r.grad_fn ? r.grad_fn : h), is(e, r, e.requires_grad || r.requires_grad ? this : null);
  }
  _backward(e) {
    const [r, s] = this.saved_tensors, [n, a] = this.next_functions;
    n.backward(e), a.backward(e);
  }
};
o(Ve, "Add");
let Q = Ve;
l("add", Q);
const os = /* @__PURE__ */ o(function(t, e, r, s, n, a) {
  const d = Array(a);
  for (let i = 0; i < a; i++) {
    const _ = p(e, n, i), u = p(s, n, i);
    d[i] = t[_] - r[u];
  }
  return d;
}, "_sub_kernel");
function ds(t, e, r = null) {
  const s = x(t.shape, e.shape), n = f(t.shape, s), a = f(e.shape, s), d = os, i = s.reduce((_, u) => _ * u, 1);
  return new c(
    d(t.data, n, e.data, a, s, i),
    { requires_grad: t.requires_grad || e.requires_grad },
    { operation: r, shape: s }
  );
}
o(ds, "_sub_tensor");
const Ge = class Ge extends g {
  _forward(e, r) {
    return (e.requires_grad || r.requires_grad) && (this.saved_tensors = [e, r]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), this.next_functions.push(r.grad_fn ? r.grad_fn : h), ds(e, r, e.requires_grad || r.requires_grad ? this : null);
  }
  _backward(e) {
    const [r, s] = this.saved_tensors, [n, a] = this.next_functions;
    n.backward(e), a.backward(e.mul(new c(-1)));
  }
};
o(Ge, "Sub");
let X = Ge;
l("sub", X);
const us = /* @__PURE__ */ o(function(t, e, r, s, n, a) {
  const d = Array(a);
  for (let i = 0; i < a; i++) {
    const _ = p(e, n, i), u = p(s, n, i);
    d[i] = t[_] * r[u];
  }
  return d;
}, "_mul_kernel");
function _s(t, e, r = null) {
  const s = x(t.shape, e.shape), n = f(t.shape, s), a = f(e.shape, s), d = us, i = s.reduce((_, u) => _ * u, 1);
  return new c(
    d(t.data, n, e.data, a, s, i),
    { requires_grad: t.requires_grad || e.requires_grad },
    { operation: r, shape: s }
  );
}
o(_s, "_mul_tensor");
const He = class He extends g {
  _forward(e, r) {
    return (e.requires_grad || r.requires_grad) && (this.saved_tensors = [e, r]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), this.next_functions.push(r.grad_fn ? r.grad_fn : h), _s(e, r, e.requires_grad || r.requires_grad ? this : null);
  }
  _backward(e) {
    const [r, s] = this.saved_tensors, [n, a] = this.next_functions;
    n.backward(e.mul(s)), a.backward(e.mul(r));
  }
};
o(He, "Mul");
let Y = He;
l("mul", Y);
const cs = /* @__PURE__ */ o(function(t, e, r, s, n, a) {
  const d = Array(a);
  for (let i = 0; i < a; i++) {
    const _ = p(e, n, i), u = p(s, n, i);
    d[i] = t[_] / r[u];
  }
  return d;
}, "_div_kernel");
function hs(t, e, r = null) {
  const s = x(t.shape, e.shape), n = f(t.shape, s), a = f(e.shape, s), d = cs, i = s.reduce((_, u) => _ * u, 1);
  return new c(
    d(t.data, n, e.data, a, s, i),
    { requires_grad: t.requires_grad || e.requires_grad },
    { operation: r, shape: s }
  );
}
o(hs, "_div_tensor");
const Je = class Je extends g {
  _forward(e, r) {
    return (e.requires_grad || r.requires_grad) && (this.saved_tensors = [e, r]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), this.next_functions.push(r.grad_fn ? r.grad_fn : h), hs(e, r, e.requires_grad || r.requires_grad ? this : null);
  }
  _backward(e) {
    const [r, s] = this.saved_tensors, [n, a] = this.next_functions;
    n.backward(e.div(s)), a.backward(e.mul(r).mul(new c(-1)).div(s).div(s));
  }
};
o(Je, "Div");
let Z = Je;
l("div", Z);
const ls = /* @__PURE__ */ o(function(t, e, r, s, n, a) {
  const d = Array(a);
  for (let i = 0; i < a; i++) {
    const _ = p(e, n, i), u = p(s, n, i);
    d[i] = Math.pow(t[_], r[u]);
  }
  return d;
}, "_pow_kernel");
function fs(t, e, r = null) {
  const s = x(t.shape, e.shape), n = f(t.shape, s), a = f(e.shape, s), d = ls, i = s.reduce((_, u) => _ * u, 1);
  return new c(
    d(t.data, n, e.data, a, s, i),
    { requires_grad: t.requires_grad || e.requires_grad },
    { operation: r, shape: s }
  );
}
o(fs, "_pow_tensor");
const Qe = class Qe extends g {
  _forward(e, r) {
    return (e.requires_grad || r.requires_grad) && (this.saved_tensors = [e, r]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), this.next_functions.push(r.grad_fn ? r.grad_fn : h), fs(e, r, e.requires_grad || r.requires_grad ? this : null);
  }
  _backward(e) {
    const [r, s] = this.saved_tensors, [n, a] = this.next_functions;
    n.backward(e.mul(s).mul(r.pow(s.sub(new c(1))))), a.backward(e.mul(r.pow(s)).mul(r.log()));
  }
};
o(Qe, "Pow");
let S = Qe;
l("pow", S);
const ps = /* @__PURE__ */ o(function(t, e, r, s, n, a) {
  const d = Array(a);
  for (let i = 0; i < a; i++) {
    const _ = p(e, n, i), u = p(s, n, i);
    d[i] = t[_] % r[u];
  }
  return d;
}, "_fmod_kernel");
function gs(t, e, r = null) {
  const s = x(t.shape, e.shape), n = f(t.shape, s), a = f(e.shape, s), d = ps, i = s.reduce((_, u) => _ * u, 1);
  return new c(
    d(t.data, n, e.data, a, s, i),
    { requires_grad: t.requires_grad || e.requires_grad },
    { operation: r, shape: s }
  );
}
o(gs, "_fmod_tensor");
const Xe = class Xe extends g {
  _forward(e, r) {
    return (e.requires_grad || r.requires_grad) && (this.saved_tensors = [e, r]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), this.next_functions.push(r.grad_fn ? r.grad_fn : h), gs(e, r, e.requires_grad || r.requires_grad ? this : null);
  }
  _backward(e) {
    const [r, s] = this.saved_tensors, [n, a] = this.next_functions;
    n.backward(e);
  }
};
o(Xe, "Fmod");
let L = Xe;
l("fmod", L);
const ws = /* @__PURE__ */ o(function(t, e, r, s, n, a) {
  const d = Array(a);
  for (let i = 0; i < a; i++) {
    const _ = p(e, n, i), u = p(s, n, i);
    d[i] = Math.max(t[_], r[u]);
  }
  return d;
}, "_maximum_kernel");
function xs(t, e, r = null) {
  const s = x(t.shape, e.shape), n = f(t.shape, s), a = f(e.shape, s), d = ws, i = s.reduce((_, u) => _ * u, 1);
  return new c(
    d(t.data, n, e.data, a, s, i),
    { requires_grad: t.requires_grad || e.requires_grad },
    { operation: r, shape: s }
  );
}
o(xs, "_maximum_tensor");
const Ye = class Ye extends g {
  _forward(e, r) {
    return (e.requires_grad || r.requires_grad) && (this.saved_tensors = [e, r]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), this.next_functions.push(r.grad_fn ? r.grad_fn : h), xs(e, r, e.requires_grad || r.requires_grad ? this : null);
  }
  _backward(e) {
    const [r, s] = this.saved_tensors, [n, a] = this.next_functions;
    n.backward(e.mul(r.ge(s))), a.backward(e.mul(s.gt(r)));
  }
};
o(Ye, "Maximum");
let ee = Ye;
l("maximum", ee);
const ms = /* @__PURE__ */ o(function(t, e, r, s, n, a) {
  const d = Array(a);
  for (let i = 0; i < a; i++) {
    const _ = p(e, n, i), u = p(s, n, i);
    d[i] = Math.min(t[_], r[u]);
  }
  return d;
}, "_minimum_kernel");
function qs(t, e, r = null) {
  const s = x(t.shape, e.shape), n = f(t.shape, s), a = f(e.shape, s), d = ms, i = s.reduce((_, u) => _ * u, 1);
  return new c(
    d(t.data, n, e.data, a, s, i),
    { requires_grad: t.requires_grad || e.requires_grad },
    { operation: r, shape: s }
  );
}
o(qs, "_minimum_tensor");
const Ze = class Ze extends g {
  _forward(e, r) {
    return (e.requires_grad || r.requires_grad) && (this.saved_tensors = [e, r]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), this.next_functions.push(r.grad_fn ? r.grad_fn : h), qs(e, r, e.requires_grad || r.requires_grad ? this : null);
  }
  _backward(e) {
    const [r, s] = this.saved_tensors, [n, a] = this.next_functions;
    n.backward(e.mul(r.le(s))), a.backward(e.mul(s.lt(r)));
  }
};
o(Ze, "Minimum");
let re = Ze;
l("minimum", re);
function ks(t, e, r = null) {
  const s = new Array(t.dataLength());
  for (let n = 0; n < s.length; n++)
    s[n] = Math.pow(t.data[n], e);
  return new c(
    s,
    { requires_grad: t.requires_grad },
    { operation: r, shape: t.shape }
  );
}
o(ks, "_powint_tensor");
const Se = class Se extends y {
  n;
  _forward(e, r) {
    return e.requires_grad && (this.saved_tensors = [e], this.n = r), this.next_functions.push(e.grad_fn ? e.grad_fn : h), ks(e, r, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [r] = this.saved_tensors, s = this.n, [n] = this.next_functions;
    n.backward(e.mul(s).mul(r.pow(s - 1)));
  }
};
o(Se, "PowInt");
let se = Se;
l("powint", se);
const vs = /* @__PURE__ */ o(function(t, e) {
  const r = new Array(e);
  for (let s = 0; s < e; s++)
    r[s] = Math.log(t[s]);
  return r;
}, "_log_kernel");
function ys(t, e = null) {
  const r = vs, s = t.shape.reduce((n, a) => n * a, 1);
  return new c(
    r(t.data, s),
    { requires_grad: t.requires_grad },
    { operation: e, shape: t.shape }
  );
}
o(ys, "_log_tensor");
const Le = class Le extends w {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), ys(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [r] = this.saved_tensors, [s] = this.next_functions;
    s.backward(new c(1).div(r));
  }
};
o(Le, "Log");
let te = Le;
l("log", te);
const As = /* @__PURE__ */ o(function(t, e) {
  const r = new Array(e);
  for (let s = 0; s < e; s++)
    r[s] = Math.sqrt(t[s]);
  return r;
}, "_sqrt_kernel");
function Os(t, e = null) {
  const r = As, s = t.shape.reduce((n, a) => n * a, 1);
  return new c(
    r(t.data, s),
    { requires_grad: t.requires_grad },
    { operation: e, shape: t.shape }
  );
}
o(Os, "_sqrt_tensor");
const er = class er extends w {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), Os(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [r] = this.saved_tensors, [s] = this.next_functions;
    s.backward(new c(1).div(r.sqrt()).div(2));
  }
};
o(er, "Sqrt");
let ne = er;
l("sqrt", ne);
const bs = /* @__PURE__ */ o(function(t, e) {
  const r = new Array(e);
  for (let s = 0; s < e; s++)
    r[s] = Math.exp(t[s]);
  return r;
}, "_exp_kernel");
function Fs(t, e = null) {
  const r = bs, s = t.shape.reduce((n, a) => n * a, 1);
  return new c(
    r(t.data, s),
    { requires_grad: t.requires_grad },
    { operation: e, shape: t.shape }
  );
}
o(Fs, "_exp_tensor");
const rr = class rr extends w {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), Fs(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [r] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.mul(r.exp()));
  }
};
o(rr, "Exp");
let ae = rr;
l("exp", ae);
const Es = /* @__PURE__ */ o(function(t, e) {
  const r = new Array(e);
  for (let s = 0; s < e; s++)
    r[s] = t[s] * t[s];
  return r;
}, "_square_kernel");
function Rs(t, e = null) {
  const r = Es, s = t.shape.reduce((n, a) => n * a, 1);
  return new c(
    r(t.data, s),
    { requires_grad: t.requires_grad },
    { operation: e, shape: t.shape }
  );
}
o(Rs, "_square_tensor");
const sr = class sr extends w {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), Rs(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [r] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.mul(r).mul(new c(2)));
  }
};
o(sr, "Square");
let ie = sr;
l("square", ie);
const Bs = /* @__PURE__ */ o(function(t, e) {
  const r = new Array(e);
  for (let s = 0; s < e; s++)
    r[s] = Math.abs(t[s]);
  return r;
}, "_abs_kernel");
function Ms(t, e = null) {
  const r = Bs, s = t.shape.reduce((n, a) => n * a, 1);
  return new c(
    r(t.data, s),
    { requires_grad: t.requires_grad },
    { operation: e, shape: t.shape }
  );
}
o(Ms, "_abs_tensor");
const tr = class tr extends w {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), Ms(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [r] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.mul(es(r)));
  }
};
o(tr, "Abs");
let oe = tr;
l("abs", oe);
const Ts = /* @__PURE__ */ o(function(t, e) {
  const r = new Array(e);
  for (let s = 0; s < e; s++)
    r[s] = Math.sign(t[s]);
  return r;
}, "_sign_kernel");
function Cs(t, e = null) {
  const r = Ts, s = t.shape.reduce((n, a) => n * a, 1);
  return new c(
    r(t.data, s),
    { requires_grad: t.requires_grad },
    { operation: e, shape: t.shape }
  );
}
o(Cs, "_sign_tensor");
const nr = class nr extends w {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), Cs(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [r] = this.saved_tensors, [s] = this.next_functions;
  }
};
o(nr, "Sign");
let de = nr;
l("sign", de);
const Us = /* @__PURE__ */ o(function(t, e) {
  const r = new Array(e);
  for (let s = 0; s < e; s++)
    r[s] = -t[s];
  return r;
}, "_neg_kernel");
function zs(t, e = null) {
  const r = Us, s = t.shape.reduce((n, a) => n * a, 1);
  return new c(
    r(t.data, s),
    { requires_grad: t.requires_grad },
    { operation: e, shape: t.shape }
  );
}
o(zs, "_neg_tensor");
const ar = class ar extends w {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), zs(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [r] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.mul(new c(-1)));
  }
};
o(ar, "Neg");
let ue = ar;
l("neg", ue);
const Ds = /* @__PURE__ */ o(function(t, e) {
  const r = new Array(e);
  for (let s = 0; s < e; s++)
    r[s] = 1 / t[s];
  return r;
}, "_reciprocal_kernel");
function Is(t, e = null) {
  const r = Ds, s = t.shape.reduce((n, a) => n * a, 1);
  return new c(
    r(t.data, s),
    { requires_grad: t.requires_grad },
    { operation: e, shape: t.shape }
  );
}
o(Is, "_reciprocal_tensor");
const ir = class ir extends w {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), Is(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [r] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.mul(r.pow(-2)));
  }
};
o(ir, "Reciprocal");
let _e = ir;
l("reciprocal", _e);
const or = class or extends y {
  _forward(e, r) {
    const s = e.dataLength(), n = r.reduce((a, d) => a * d, 1);
    if (s !== n)
      throw new Error("Shape mismatch: " + e.shape + " and " + r);
    if (e.requires_grad && (this.saved_tensors = [e]), e.grad_fn)
      this.next_functions.push(e.grad_fn);
    else if (e.requires_grad) {
      const a = new U();
      a.variable = e, this.next_functions.push(a);
    } else
      this.next_functions.push(h);
    return new c(
      e.data,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: r }
    );
  }
  _backward(e) {
    const [r] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.reshape(r.shape));
  }
};
o(or, "Reshape");
let ce = or;
l("reshape", ce);
const dr = class dr extends y {
  _forward(e, r) {
    if (e.requires_grad && (this.saved_tensors = [e]), e.grad_fn)
      this.next_functions.push(e.grad_fn);
    else if (e.requires_grad) {
      const n = new U();
      n.variable = e, this.next_functions.push(n);
    } else
      this.next_functions.push(h);
    r < 0 && (r += e.shape.length + 1);
    const s = [...e.shape];
    return s.splice(r, 0, 1), new c(
      e.data,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: s }
    );
  }
  _backward(e) {
    const [r] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.reshape(r.shape));
  }
};
o(dr, "Unsqueeze");
let he = dr;
l("unsqueeze", he);
const Ws = /* @__PURE__ */ o(function(t, e) {
  const r = new Array(e);
  for (let s = 0; s < e; s++)
    r[s] = Math.sin(t[s]);
  return r;
}, "_sin_kernel");
function Ns(t, e = null) {
  const r = Ws, s = t.shape.reduce((n, a) => n * a, 1);
  return new c(
    r(t.data, s),
    { requires_grad: t.requires_grad },
    { operation: e, shape: t.shape }
  );
}
o(Ns, "_sin_tensor");
const ur = class ur extends w {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), Ns(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [r] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.mul(r.cos()));
  }
};
o(ur, "Sin");
let le = ur;
l("sin", le);
const Ps = /* @__PURE__ */ o(function(t, e) {
  const r = new Array(e);
  for (let s = 0; s < e; s++)
    r[s] = Math.cos(t[s]);
  return r;
}, "_cos_kernel");
function js(t, e = null) {
  const r = Ps, s = t.shape.reduce((n, a) => n * a, 1);
  return new c(
    r(t.data, s),
    { requires_grad: t.requires_grad },
    { operation: e, shape: t.shape }
  );
}
o(js, "_cos_tensor");
const _r = class _r extends w {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), js(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [r] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.mul(r.sin().neg()));
  }
};
o(_r, "Cos");
let fe = _r;
l("cos", fe);
const Ks = /* @__PURE__ */ o(function(t, e) {
  const r = new Array(e);
  for (let s = 0; s < e; s++)
    r[s] = Math.tan(t[s]);
  return r;
}, "_tan_kernel");
function $s(t, e = null) {
  const r = Ks, s = t.shape.reduce((n, a) => n * a, 1);
  return new c(
    r(t.data, s),
    { requires_grad: t.requires_grad },
    { operation: e, shape: t.shape }
  );
}
o($s, "_tan_tensor");
const cr = class cr extends w {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), $s(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [r] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.mul(r.cos().pow(-2)));
  }
};
o(cr, "Tan");
let pe = cr;
l("tan", pe);
function Vs(t, e = null) {
  return new c(
    t.toArray().reduce((r, s) => r + s, 0),
    { requires_grad: t.requires_grad },
    { operation: e }
  );
}
o(Vs, "_sum_tensor");
const hr = class hr extends w {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), Vs(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [r] = this.saved_tensors, [s] = this.next_functions;
    s.backward(T(r).add(e.item()));
  }
};
o(hr, "Sum");
let ge = hr;
l("sum", ge);
function Gs(t, e = null) {
  return new c(
    t.toArray().reduce((r, s) => r + s, 0) / t.dataLength(),
    { requires_grad: t.requires_grad },
    { operation: e }
  );
}
o(Gs, "_mean_tensor");
const lr = class lr extends w {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), Gs(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [r] = this.saved_tensors, [s] = this.next_functions;
    s.backward(T(r).add(e.item() / r.dataLength()));
  }
};
o(lr, "Mean");
let we = lr;
l("mean", we);
function Hs(t, e, r, s = null) {
  e = e < 0 ? t.shape.length + e : e, r = r < 0 ? t.shape.length + r : r;
  const n = [...t.shape];
  [n[e], n[r]] = [n[r], n[e]];
  const a = t.dataLength(), d = new Array(a), i = new Array(t.shape.length), _ = new Array(n.length);
  for (let u = t.shape.length - 1, k = 1; u >= 0; u--)
    i[u] = k, k *= t.shape[u];
  for (let u = n.length - 1, k = 1; u >= 0; u--)
    _[u] = k, k *= n[u];
  for (let u = 0; u < a; u++) {
    let k = u, W = 0;
    for (let A = 0; A < n.length; A++) {
      const O = _[A], $ = Math.floor(k / O);
      k %= O;
      let v = A;
      A === e ? v = r : A === r && (v = e), W += $ * i[v];
    }
    d[u] = t.data[W];
  }
  return new c(
    d,
    { requires_grad: t.requires_grad },
    { operation: s, shape: n }
  );
}
o(Hs, "_transpose_tensor");
const fr = class fr extends y {
  dim0;
  dim1;
  _forward(e, r, s) {
    return e.requires_grad && (this.saved_tensors = [e], this.dim0 = r, this.dim1 = s), this.next_functions.push(e.grad_fn ? e.grad_fn : h), Hs(e, r, s, this);
  }
  _backward(e) {
    const [r] = this.saved_tensors, s = this.dim0, n = this.dim1, [a] = this.next_functions;
    a.backward(e.transpose(s, n));
  }
};
o(fr, "Transpose");
let xe = fr;
l("transpose", xe);
function Js(t, e, r = null) {
  if (t.shape.length == 1 && e.shape.length == 1)
    return t.mul(e).sum();
  const s = t.shape.length == 1, n = e.shape.length == 1, a = s ? [1, t.shape[0]] : t.shape, d = n ? [e.shape[0], 1] : e.shape;
  if (a[a.length - 1] != d[d.length - 2])
    throw new Error("Shape mismatch: " + t.shape + " and " + e.shape);
  const i = x(a.slice(0, -2), d.slice(0, -2)).concat([
    a[a.length - 2],
    d[d.length - 1]
  ]), _ = i.reduce((b, N) => b * N, 1), u = new Array(_).fill(0), k = f(a, i), W = f(d, i), A = i[i.length - 2], O = i[i.length - 1], $ = a[a.length - 1];
  for (let b = 0; b < _; b++) {
    const N = b % (A * O), Vr = Math.floor(N / O), Gr = N % O;
    let Hr = p(k, i, b - Gr), Jr = p(W, i, b - Vr * O), zr = 0;
    for (let P = 0; P < $; P++)
      zr += t.data[Hr + P] * e.data[Jr + P * O];
    u[b] = zr;
  }
  let v = [...i];
  return s && (v = v.slice(0, -2).concat([i[i.length - 1]])), n && (v = v.slice(0, -1)), new c(
    u,
    { requires_grad: t.requires_grad || e.requires_grad },
    { operation: r, shape: v }
  );
}
o(Js, "_matmul_tensor");
const pr = class pr extends g {
  _forward(e, r) {
    return (e.requires_grad || r.requires_grad) && (this.saved_tensors = [e, r]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), this.next_functions.push(r.grad_fn ? r.grad_fn : h), Js(e, r, e.requires_grad || r.requires_grad ? this : null);
  }
  _backward(e) {
    const [r, s] = this.saved_tensors, [n, a] = this.next_functions;
    n.backward(e.matmul(s.transpose(-2, -1))), a.backward(r.transpose(-2, -1).matmul(e));
  }
};
o(pr, "Matmul");
let me = pr;
l("matmul", me);
const Qs = /* @__PURE__ */ o(function(t, e, r, s, n, a) {
  const d = Array(a);
  for (let i = 0; i < a; i++) {
    const _ = p(e, n, i), u = p(s, n, i);
    d[i] = t[_] < r[u] ? 1 : 0;
  }
  return d;
}, "_lt_kernel");
function Xs(t, e, r = null) {
  const s = x(t.shape, e.shape), n = f(t.shape, s), a = f(e.shape, s), d = Qs, i = s.reduce((_, u) => _ * u, 1);
  return new c(
    d(t.data, n, e.data, a, s, i),
    { requires_grad: t.requires_grad || e.requires_grad },
    { operation: r, shape: s }
  );
}
o(Xs, "_lt_tensor");
const gr = class gr extends g {
  _forward(e, r) {
    return (e.requires_grad || r.requires_grad) && (this.saved_tensors = [e, r]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), this.next_functions.push(r.grad_fn ? r.grad_fn : h), Xs(e, r, e.requires_grad || r.requires_grad ? this : null);
  }
  _backward(e) {
    const [r, s] = this.saved_tensors, [n, a] = this.next_functions;
  }
};
o(gr, "Lt");
let qe = gr;
l("lt", qe);
const Ys = /* @__PURE__ */ o(function(t, e, r, s, n, a) {
  const d = Array(a);
  for (let i = 0; i < a; i++) {
    const _ = p(e, n, i), u = p(s, n, i);
    d[i] = t[_] > r[u] ? 1 : 0;
  }
  return d;
}, "_gt_kernel");
function Zs(t, e, r = null) {
  const s = x(t.shape, e.shape), n = f(t.shape, s), a = f(e.shape, s), d = Ys, i = s.reduce((_, u) => _ * u, 1);
  return new c(
    d(t.data, n, e.data, a, s, i),
    { requires_grad: t.requires_grad || e.requires_grad },
    { operation: r, shape: s }
  );
}
o(Zs, "_gt_tensor");
const wr = class wr extends g {
  _forward(e, r) {
    return (e.requires_grad || r.requires_grad) && (this.saved_tensors = [e, r]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), this.next_functions.push(r.grad_fn ? r.grad_fn : h), Zs(e, r, e.requires_grad || r.requires_grad ? this : null);
  }
  _backward(e) {
    const [r, s] = this.saved_tensors, [n, a] = this.next_functions;
  }
};
o(wr, "Gt");
let ke = wr;
l("gt", ke);
const Ss = /* @__PURE__ */ o(function(t, e, r, s, n, a) {
  const d = Array(a);
  for (let i = 0; i < a; i++) {
    const _ = p(e, n, i), u = p(s, n, i);
    d[i] = t[_] <= r[u] ? 1 : 0;
  }
  return d;
}, "_le_kernel");
function Ls(t, e, r = null) {
  const s = x(t.shape, e.shape), n = f(t.shape, s), a = f(e.shape, s), d = Ss, i = s.reduce((_, u) => _ * u, 1);
  return new c(
    d(t.data, n, e.data, a, s, i),
    { requires_grad: t.requires_grad || e.requires_grad },
    { operation: r, shape: s }
  );
}
o(Ls, "_le_tensor");
const xr = class xr extends g {
  _forward(e, r) {
    return (e.requires_grad || r.requires_grad) && (this.saved_tensors = [e, r]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), this.next_functions.push(r.grad_fn ? r.grad_fn : h), Ls(e, r, e.requires_grad || r.requires_grad ? this : null);
  }
  _backward(e) {
    const [r, s] = this.saved_tensors, [n, a] = this.next_functions;
  }
};
o(xr, "Le");
let ve = xr;
l("le", ve);
const et = /* @__PURE__ */ o(function(t, e, r, s, n, a) {
  const d = Array(a);
  for (let i = 0; i < a; i++) {
    const _ = p(e, n, i), u = p(s, n, i);
    d[i] = t[_] >= r[u] ? 1 : 0;
  }
  return d;
}, "_ge_kernel");
function rt(t, e, r = null) {
  const s = x(t.shape, e.shape), n = f(t.shape, s), a = f(e.shape, s), d = et, i = s.reduce((_, u) => _ * u, 1);
  return new c(
    d(t.data, n, e.data, a, s, i),
    { requires_grad: t.requires_grad || e.requires_grad },
    { operation: r, shape: s }
  );
}
o(rt, "_ge_tensor");
const mr = class mr extends g {
  _forward(e, r) {
    return (e.requires_grad || r.requires_grad) && (this.saved_tensors = [e, r]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), this.next_functions.push(r.grad_fn ? r.grad_fn : h), rt(e, r, e.requires_grad || r.requires_grad ? this : null);
  }
  _backward(e) {
    const [r, s] = this.saved_tensors, [n, a] = this.next_functions;
  }
};
o(mr, "Ge");
let ye = mr;
l("ge", ye);
const st = /* @__PURE__ */ o(function(t, e, r, s, n, a) {
  const d = Array(a);
  for (let i = 0; i < a; i++) {
    const _ = p(e, n, i), u = p(s, n, i);
    d[i] = t[_] == r[u] ? 1 : 0;
  }
  return d;
}, "_eq_kernel");
function tt(t, e, r = null) {
  const s = x(t.shape, e.shape), n = f(t.shape, s), a = f(e.shape, s), d = st, i = s.reduce((_, u) => _ * u, 1);
  return new c(
    d(t.data, n, e.data, a, s, i),
    { requires_grad: t.requires_grad || e.requires_grad },
    { operation: r, shape: s }
  );
}
o(tt, "_eq_tensor");
const qr = class qr extends g {
  _forward(e, r) {
    return (e.requires_grad || r.requires_grad) && (this.saved_tensors = [e, r]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), this.next_functions.push(r.grad_fn ? r.grad_fn : h), tt(e, r, e.requires_grad || r.requires_grad ? this : null);
  }
  _backward(e) {
    const [r, s] = this.saved_tensors, [n, a] = this.next_functions;
  }
};
o(qr, "Eq");
let Ae = qr;
l("eq", Ae);
const nt = /* @__PURE__ */ o(function(t, e, r, s, n, a) {
  const d = Array(a);
  for (let i = 0; i < a; i++) {
    const _ = p(e, n, i), u = p(s, n, i);
    d[i] = t[_] != r[u] ? 1 : 0;
  }
  return d;
}, "_ne_kernel");
function at(t, e, r = null) {
  const s = x(t.shape, e.shape), n = f(t.shape, s), a = f(e.shape, s), d = nt, i = s.reduce((_, u) => _ * u, 1);
  return new c(
    d(t.data, n, e.data, a, s, i),
    { requires_grad: t.requires_grad || e.requires_grad },
    { operation: r, shape: s }
  );
}
o(at, "_ne_tensor");
const kr = class kr extends g {
  _forward(e, r) {
    return (e.requires_grad || r.requires_grad) && (this.saved_tensors = [e, r]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), this.next_functions.push(r.grad_fn ? r.grad_fn : h), at(e, r, e.requires_grad || r.requires_grad ? this : null);
  }
  _backward(e) {
    const [r, s] = this.saved_tensors, [n, a] = this.next_functions;
  }
};
o(kr, "Ne");
let Oe = kr;
l("ne", Oe);
const it = /* @__PURE__ */ o(function(t, e) {
  const r = new Array(e);
  for (let s = 0; s < e; s++)
    r[s] = Math.max(t[s], 0);
  return r;
}, "_relu_kernel");
function ot(t, e = null) {
  const r = it, s = t.shape.reduce((n, a) => n * a, 1);
  return new c(
    r(t.data, s),
    { requires_grad: t.requires_grad },
    { operation: e, shape: t.shape }
  );
}
o(ot, "_relu_tensor");
const vr = class vr extends w {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), ot(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [r] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.mul(r.gt(0)));
  }
};
o(vr, "Relu");
let be = vr;
l("relu", be);
const dt = /* @__PURE__ */ o(function(t, e) {
  const r = new Array(e);
  for (let s = 0; s < e; s++)
    r[s] = 1 / (1 + Math.exp(-t[s]));
  return r;
}, "_sigmoid_kernel");
function ut(t, e = null) {
  const r = dt, s = t.shape.reduce((n, a) => n * a, 1);
  return new c(
    r(t.data, s),
    { requires_grad: t.requires_grad },
    { operation: e, shape: t.shape }
  );
}
o(ut, "_sigmoid_tensor");
var C;
let _t = (C = class extends w {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : h), ut(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [r] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.mul(r.exp().add(1).pow(-2).reciprocal().mul(r.exp()).mul(-1)));
  }
}, o(C, "Sigmoid"), C);
l("sigmoid", _t);
const j = class j extends c {
  constructor(e, r = {
    requires_grad: !0
  }, s = {}) {
    e instanceof c ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : e instanceof j ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : super(e, r, s);
  }
};
o(j, "Parameter");
let z = j;
const yr = class yr {
  _modules;
  _parameters;
  constructor() {
    this._parameters = {}, this._modules = {};
  }
  register_parameter(e, r) {
    this._parameters[e] = r;
  }
  register_module(e, r) {
    this._modules[e] = r;
  }
  register(e, r) {
    r instanceof z ? this.register_parameter(e, r) : this.register_module(e, r);
  }
  parameters() {
    let e = Object.values(this._parameters);
    for (const r of Object.values(this._modules))
      e = e.concat(r.parameters());
    return e;
  }
};
o(yr, "Module");
let B = yr;
const Ar = class Ar extends B {
  weight;
  bias;
  constructor(e, r) {
    super();
    const s = Math.sqrt(1 / e);
    this.weight = new z(
      Dr([r, e]).mul(2 * s).sub(s)
    ), this.bias = new z(
      Dr([r]).mul(2 * s).sub(s)
    ), this.register("weight", this.weight), this.register("bias", this.bias);
  }
  forward(e) {
    return e.matmul(this.weight.transpose(0, 1)).add(this.bias);
  }
};
o(Ar, "Linear");
let Fe = Ar;
const Or = class Or extends B {
  constructor() {
    super();
  }
  forward(e) {
    return Kr(e);
  }
};
o(Or, "ReLU");
let Ee = Or;
const br = class br extends B {
  constructor() {
    super();
  }
  forward(e) {
    return $r(e);
  }
};
o(br, "Sigmoid");
let Re = br;
const Fr = class Fr extends B {
  _modulesArr;
  constructor(...e) {
    super(), this._modulesArr = e;
    for (let r = 0; r < e.length; r++)
      this.register(r.toString(), e[r]);
  }
  append(e) {
    return this.register(this._modulesArr.length.toString(), e), this._modulesArr.push(e), this;
  }
  extend(e) {
    for (const r of e._modulesArr)
      this.append(r);
    return this;
  }
  insert(e, r) {
    this._modulesArr.splice(e, 0, r);
    for (let s = e; s < this._modulesArr.length; s++)
      this.register(s.toString(), this._modulesArr[s]);
    return this;
  }
  forward(e) {
    let r = e;
    for (const s of this._modulesArr)
      r = s.forward(r);
    return r;
  }
};
o(Fr, "Sequential");
let Be = Fr;
const Er = class Er {
};
o(Er, "Loss");
let D = Er;
const Rr = class Rr extends D {
  constructor() {
    super();
  }
  forward(e, r) {
    return e.sub(r).pow(2).mean();
  }
};
o(Rr, "MSELoss");
let Me = Rr;
const Br = class Br extends D {
  constructor() {
    super();
  }
  forward(e, r) {
    return e.sub(r).abs().mean();
  }
};
o(Br, "L1Loss");
let Te = Br;
const Mr = class Mr extends D {
  weight;
  constructor(e = null) {
    super(), this.weight = e;
  }
  forward(e, r) {
    const s = r.mul(e.log()), n = r.neg().add(1).mul(e.neg().add(1).log()), a = s.add(n).neg().mean();
    return this.weight ? a.mul(this.weight) : a;
  }
};
o(Mr, "BCELoss");
let Ce = Mr;
function jr(t) {
  return (e) => (typeof e == "number" && (e = new c(e)), new (R(t))().forward(e));
}
o(jr, "generate_unary_function");
const Kr = jr("relu"), $r = jr("sigmoid"), ct = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  relu: Kr,
  sigmoid: $r
}, Symbol.toStringTag, { value: "Module" })), Yt = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  BCELoss: Ce,
  L1Loss: Te,
  Linear: Fe,
  MSELoss: Me,
  Module: B,
  Parameter: z,
  ReLU: Ee,
  Sequential: Be,
  Sigmoid: Re,
  functional: ct
}, Symbol.toStringTag, { value: "Module" })), Tr = class Tr {
  params;
  defaults;
  constructor(e, r) {
    this.params = e, this.defaults = r;
  }
  zero_grad() {
    for (const e of this.params)
      e.grad = null;
  }
};
o(Tr, "Optimizer");
let I = Tr;
const Cr = class Cr extends I {
  state = /* @__PURE__ */ new Map();
  lr;
  momentum;
  dampening;
  weight_decay;
  nesterov;
  maximize;
  constructor(e, r = 1e-3, s = 0, n = 0, a = 0, d = !1, i = !1) {
    super(e, {}), this.lr = r, this.momentum = s, this.dampening = n, this.weight_decay = a, this.nesterov = d, this.maximize = i;
  }
  step() {
    for (const e of this.params) {
      let r = this.maximize ? e.grad.mul(-1) : e.grad;
      if (this.weight_decay !== 0 && (r = r.add(e.mul(this.weight_decay))), this.momentum !== 0) {
        if (this.state.has(e)) {
          let a = this.state.get(e).velocity;
          a = a.mul(this.momentum), a = a.add(r.mul(1 - this.dampening)), this.state.set(e, { velocity: a });
        } else
          this.state.set(e, { velocity: r });
        let n = this.state.get(e).velocity;
        this.nesterov ? r = r.add(n.mul(this.momentum)) : r = n, this.state.set(e, { velocity: n });
      }
      const s = e.sub(r.mul(this.lr));
      e.data = s.data;
    }
  }
};
o(Cr, "SGD");
let Ue = Cr;
const Ur = class Ur extends I {
  state = /* @__PURE__ */ new Map();
  step_count = 0;
  lr;
  beta1;
  beta2;
  eps;
  weight_decay;
  amsgrad;
  maximize;
  constructor(e, r = 1e-3, s = [0.9, 0.999], n = 1e-8, a = 0, d = !1, i = !1) {
    super(e, {}), this.lr = r, this.beta1 = s[0], this.beta2 = s[1], this.eps = n, this.weight_decay = a, this.amsgrad = d, this.maximize = i;
  }
  step() {
    this.step_count += 1;
    for (const e of this.params) {
      let r = this.maximize ? e.grad.mul(-1) : e.grad;
      this.weight_decay !== 0 && (r = r.add(e.mul(this.weight_decay))), this.state.has(e) || this.state.set(e, {
        m: T(e),
        v: T(e),
        vmax: T(e)
      });
      const s = this.state.get(e);
      s.m = s.m.mul(this.beta1).add(r.mul(1 - this.beta1)), s.v = s.v.mul(this.beta2).add(r.mul(r).mul(1 - this.beta2));
      const n = 1 - Math.pow(this.beta1, this.step_count), a = 1 - Math.pow(this.beta2, this.step_count);
      let d;
      const i = s.m.div(n);
      this.amsgrad ? (s.vmax = s.vmax.maximum(s.v), d = s.vmax.div(a)) : d = s.v.div(a);
      const _ = i.div(d.sqrt().add(this.eps)).mul(this.lr), u = e.sub(_);
      e.data = u.data;
    }
  }
};
o(Ur, "Adam");
let ze = Ur;
const Zt = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Adam: ze,
  Optimizer: I,
  SGD: Ue
}, Symbol.toStringTag, { value: "Module" }));
export {
  oe as Abs,
  U as AccumulateGrad,
  Q as Add,
  fe as Cos,
  Z as Div,
  Ae as Eq,
  ae as Exp,
  L as Fmod,
  ye as Ge,
  ke as Gt,
  ve as Le,
  te as Log,
  qe as Lt,
  me as Matmul,
  ee as Maximum,
  we as Mean,
  re as Minimum,
  Y as Mul,
  Oe as Ne,
  ue as Neg,
  y as Operation,
  S as Pow,
  se as PowInt,
  _e as Reciprocal,
  ce as Reshape,
  de as Sign,
  le as Sin,
  ne as Sqrt,
  ie as Square,
  X as Sub,
  ge as Sum,
  pe as Tan,
  c as Tensor,
  xe as Transpose,
  he as Unsqueeze,
  H as __Left_index__,
  J as __Right_index__,
  xt as __left_index__,
  mt as __right_index__,
  Tt as abs,
  qt as add,
  wt as arange,
  Wt as cos,
  yt as div,
  Qt as eq,
  F as eventBus,
  E as events,
  Bt as exp,
  Ot as fmod,
  Jt as ge,
  Gt as gt,
  Ht as le,
  gt as linspace,
  Et as log,
  Vt as lt,
  $t as matmul,
  bt as maximum,
  jt as mean,
  Ft as minimum,
  vt as mul,
  Xt as ne,
  Ct as neg,
  Yt as nn,
  Xr as ones,
  pt as ones_like,
  Zt as optim,
  At as pow,
  Dr as rand,
  ft as randint,
  lt as randn,
  Ut as reciprocal,
  zt as reshape,
  es as sign,
  It as sin,
  Rt as sqrt,
  Mt as square,
  kt as sub,
  Pt as sum,
  Nt as tan,
  Kt as transpose,
  Dt as unsqueeze,
  Yr as zeros,
  T as zeros_like
};
//# sourceMappingURL=torch.browser.es.js.map
