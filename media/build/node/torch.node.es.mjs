function $(r, e) {
  const t = Math.max(r.length, e.length), s = [...Array(t - r.length).fill(1), ...r], n = [...Array(t - e.length).fill(1), ...e], i = [];
  for (let a = 0; a < t; a++) {
    if (s[a] !== n[a] && s[a] !== 1 && n[a] !== 1)
      throw new Error(`Shape mismatch: ${r} and ${e}`);
    i.push(Math.max(s[a], n[a]));
  }
  return i;
}
function V(r, e, t) {
  const s = M(e, r), n = new Array(e.reduce((i, a) => i * a, 1)).fill(0);
  for (let i = 0; i < t.length; i++)
    n[B(s, r, i)] += t[i];
  return n;
}
function M(r, e) {
  return r.length >= e.length ? r : [...Array(e.length - r.length).fill(1), ...r];
}
function B(r, e, t) {
  let s = 0, n = 1, i = t;
  for (let a = r.length - 1; a >= 0; a--) {
    if (r[a] > 1) {
      const o = i % e[a];
      s = s + o * n;
    }
    n *= r[a], i = Math.floor(i / e[a]);
  }
  return s;
}
function D(r) {
  return Array.isArray(r[0]) ? r[0] : r;
}
function Le(...r) {
  const e = D(r), t = new c(Array(e.reduce((s, n) => s * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
function K(...r) {
  const e = D(r), t = new c(Array(e.reduce((s, n) => s * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
function Se(r, e, t) {
  const s = new c(
    Array(t.reduce((n, i) => n * i, 1)).fill(Math.floor(Math.random() * (e - r) + r))
  );
  return s.shape = t, s;
}
function ae(...r) {
  const e = D(r), t = new c(Array(e.reduce((s, n) => s * n, 1)).fill(1));
  return t.shape = e, t;
}
function ie(...r) {
  const e = D(r), t = new c(Array(e.reduce((s, n) => s * n, 1)).fill(0));
  return t.shape = e, t;
}
function De(r) {
  return ae(r.shape);
}
function R(r) {
  return ie(r.shape);
}
function Pe(r, e, t) {
  const s = [], n = (e - r) / (t - 1);
  for (let i = 0; i < t - 1; i++)
    s.push(r + i * n);
  return s.push(e), new c(s);
}
function Ie(r, e = void 0, t = 1) {
  const s = [];
  for (let n = r; n < e; n += t)
    s.push(n);
  return new c(s);
}
let oe = 0;
const H = () => oe++, A = new EventTarget(), v = {
  TENSOR_BEFORE_BACKWARD: "tensor.beforeBackward",
  TENSOR_AFTER_BACKWARD: "tensor.afterBackward",
  OPERATION_BEFORE_FORWARD: "operation.beforeForward",
  OPERATION_AFTER_FORWARD: "operation.afterForward",
  OPERATION_BEFORE_BACKWARD: "operation.beforeBackward",
  OPERATION_AFTER_BACKWARD: "operation.afterBackward",
  OPERATION_BEFORE_ACCUMULATE_GRAD: "operation.beforeAccumulateGrad",
  OPERATION_AFTER_ACCUMULATE_GRAD: "operation.afterAccumulateGrad"
};
function ue(...r) {
  for (const e of r)
    if (e instanceof c && e.requires_grad)
      return !0;
  return !1;
}
class E {
  id = H();
  next_functions = [];
  saved_tensors = [];
  _retained_tensors = [];
  forward(...e) {
    const t = ue(...e);
    A.dispatchEvent(new CustomEvent(v.OPERATION_BEFORE_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e
      }
    }));
    const s = this._forward(...e);
    return A.dispatchEvent(new CustomEvent(v.OPERATION_AFTER_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e,
        result: s
      }
    })), s;
  }
  backward(e) {
    A.dispatchEvent(new CustomEvent(v.OPERATION_BEFORE_BACKWARD, { detail: { operation: this, dz: e } }));
    for (const t of this._retained_tensors)
      t.grad || (t.grad = new c(new Array(t.dataLength()).fill(0))), t.grad = t.grad.add(e);
    this._backward(e), A.dispatchEvent(new CustomEvent(v.OPERATION_AFTER_BACKWARD, { detail: { operation: this, dz: e } }));
  }
}
class ce extends E {
  _forward(...e) {
    throw new Error("NullOp should not be called");
  }
  _backward(e) {
  }
}
const x = new ce();
class P extends E {
}
class J extends E {
}
class T extends P {
  variable;
  _forward(e) {
    return this.variable = e, e;
  }
  _backward(e) {
    if (this.variable.grad || (this.variable.grad = R(this.variable)), A.dispatchEvent(new CustomEvent(v.OPERATION_BEFORE_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } })), typeof e == "number")
      this.variable.grad = this.variable.grad.add(e);
    else {
      const t = V(e.shape, this.variable.shape, e.data);
      this.variable.grad = this.variable.grad.add(new c(t, {}, { shape: this.variable.shape }));
    }
    A.dispatchEvent(new CustomEvent(v.OPERATION_AFTER_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } }));
  }
}
const Q = /* @__PURE__ */ new Map(), I = /* @__PURE__ */ new Map();
function q(r, e) {
  Q.set(r, e);
}
function O(r) {
  const e = Q.get(r);
  if (!e)
    throw new Error(`Operation '${r}' is not registered.`);
  return e;
}
function j(r) {
  const e = I.get(r);
  return e || (I.set(r, new (O(r))()), I.get(r));
}
function he(r) {
  if (ArrayBuffer.isView(r))
    return [r.length];
  const e = [];
  for (; Array.isArray(r); )
    e.push(r.length), r = r[0];
  return e;
}
function X(r) {
  return Array.isArray(r) ? r.flatMap((e) => X(e)) : ArrayBuffer.isView(r) ? Array.from(r) : [r];
}
class c {
  // Auto-generated ID
  id = H();
  // Optional user-defined name
  name = null;
  data;
  _shape;
  grad_fn = null;
  grad = null;
  requires_grad;
  constructor(e, t = {}, s = {}) {
    if (this.data = X(e), this.requires_grad = t.requires_grad ?? !1, t.name && (this.name = t.name), this._shape = s.shape ?? he(e), this.grad_fn = s.operation ?? null, this.requires_grad && !this.grad_fn) {
      const n = new T();
      n.variable = this, this.grad_fn = n;
    }
  }
  get shape() {
    return this._shape;
  }
  toArray_() {
  }
  toFlatArray() {
    return this.data;
  }
  toArray() {
    if (this.shape.length === 0)
      return this.data[0];
    let e = 0;
    const t = this.data, s = (n) => {
      const i = this.shape[n], a = new Array(i), o = n === this.shape.length - 1;
      for (let h = 0; h < i; h++)
        o ? a[h] = t[e++] : a[h] = s(n + 1);
      return a;
    };
    return s(0);
  }
  dataLength() {
    return this.data.length;
  }
  set shape(e) {
    this._shape = e;
  }
  _executeUnaryOp(e) {
    return (this.requires_grad ? new (O(e))() : j(e)).forward(this);
  }
  _executeBinaryOp(e, t) {
    return typeof t == "number" && (t = new c(t)), (this.requires_grad || t.requires_grad ? new (O(e))() : j(e)).forward(this, t);
  }
  _executeOpRaw(e, ...t) {
    return new (O(e))().forward(this, ...t);
  }
  item() {
    if (this.dataLength() !== 1)
      throw new Error("Tensor.item() is only valid for scalars");
    return this.data[0];
  }
  detach() {
    return new c(this.data, { requires_grad: !1 }, { shape: this.shape });
  }
  detach_() {
    this.requires_grad = !1, this.grad = null, this.grad_fn = null;
  }
  zero_() {
    this.data = Array(this.dataLength()).fill(0);
  }
  is_retain_grad = !1;
  retain_grad() {
    this.grad_fn instanceof T || this.is_retain_grad || (this.is_retain_grad = !0, this.grad_fn._retained_tensors.push(this));
  }
  backward(e) {
    if (this.requires_grad) {
      if (e)
        e.toArray_();
      else {
        if (this.dataLength() !== 1)
          throw new Error("Gradient is required for non-scalar tensors");
        e = new c(1);
      }
      this.grad_fn && (A.dispatchEvent(new CustomEvent(v.TENSOR_BEFORE_BACKWARD, { detail: { tensor: this } })), this.grad_fn.backward(e), A.dispatchEvent(new CustomEvent(v.TENSOR_AFTER_BACKWARD, { detail: { tensor: this } })));
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
  squeeze(e) {
    return this._executeOpRaw("squeeze", e);
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
  transpose(e, t) {
    return this._executeOpRaw("transpose", e, t);
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
  // other
  sigmoid() {
    return this._executeUnaryOp("sigmoid");
  }
}
function N(r) {
  return (...e) => new (O(r))().forward(...e);
}
function m(r) {
  return (e) => (typeof e == "number" && (e = new c(e)), new (O(r))().forward(e));
}
function _(r) {
  return (e, t) => (typeof e == "number" && (e = new c(e)), typeof t == "number" && (t = new c(t)), new (O(r))().forward(e, t));
}
const Ne = _("__left_index__"), We = _("__right_index__"), ze = _("add"), Ge = _("sub"), Ke = _("mul"), je = _("div"), $e = _("pow"), Ve = _("fmod"), He = _("maximum"), Je = _("minimum"), Qe = m("log"), Xe = m("sqrt"), Ye = m("exp"), Ze = m("square"), et = m("abs"), de = m("sign"), tt = m("neg"), rt = m("reciprocal"), st = N("reshape"), nt = N("unsqueeze"), at = m("sin"), it = m("cos"), ot = m("tan"), ut = m("sum"), ct = m("mean"), ht = N("transpose"), dt = _("matmul"), lt = _("lt"), _t = _("gt"), pt = _("le"), ft = _("ge"), gt = _("eq"), mt = _("ne");
function f(r, e, t = null) {
  const s = (a, o, h, u, d, l) => {
    const g = Array(l);
    for (let p = 0; p < l; p++) {
      const k = B(o, d, p), b = B(u, d, p);
      g[p] = r(a, h, k, b);
    }
    return g;
  }, n = (a, o, h = null) => {
    const u = $(a.shape, o.shape), d = M(a.shape, u), l = M(o.shape, u), g = u.reduce((p, k) => p * k, 1);
    return new c(
      s(a.data, d, o.data, l, u, g),
      { requires_grad: a.requires_grad || o.requires_grad },
      { operation: h, shape: u }
    );
  }, i = class extends J {
    _forward(a, o) {
      return (a.requires_grad || o.requires_grad) && (this.saved_tensors = [a, o]), this.next_functions.push(a.grad_fn ? a.grad_fn : x), this.next_functions.push(o.grad_fn ? o.grad_fn : x), n(a, o, a.requires_grad || o.requires_grad ? this : null);
    }
    _backward(a) {
      const [o, h] = this.saved_tensors, [u, d] = this.next_functions;
      e(o, h, u, d, a);
    }
  };
  return t && q(t, i), i;
}
function w(r, e, t = null) {
  const s = (a, o) => {
    const h = Array(o);
    for (let u = 0; u < o; u++)
      h[u] = r(a, u);
    return h;
  }, n = (a, o = null) => {
    const h = a.dataLength();
    return new c(
      s(a.data, h),
      { requires_grad: a.requires_grad },
      { operation: o, shape: a.shape }
    );
  }, i = class extends P {
    _forward(a) {
      return a.requires_grad && (this.saved_tensors = [a]), this.next_functions.push(a.grad_fn ? a.grad_fn : x), n(a, a.requires_grad ? this : null);
    }
    _backward(a) {
      const [o] = this.saved_tensors, [h] = this.next_functions;
      e(o, h, a);
    }
  };
  return t && q(t, i), i;
}
function S(r, e) {
  const t = V(r.shape, e, r.data);
  return new c(t, { requires_grad: r.requires_grad }, { shape: e });
}
f(
  (r, e, t, s) => t,
  (r, e, t, s, n) => {
  },
  "__left_index__"
);
f(
  (r, e, t, s) => s,
  (r, e, t, s, n) => {
  },
  "__right_index__"
);
f(
  (r, e, t, s) => r[t] + e[s],
  (r, e, t, s, n) => {
    t.backward(n), s.backward(n);
  },
  "add"
);
f(
  (r, e, t, s) => r[t] - e[s],
  (r, e, t, s, n) => {
    t.backward(n), s.backward(n.mul(new c(-1)));
  },
  "sub"
);
f(
  (r, e, t, s) => r[t] * e[s],
  (r, e, t, s, n) => {
    t.backward(n.mul(e)), s.backward(n.mul(r));
  },
  "mul"
);
f(
  (r, e, t, s) => r[t] / e[s],
  (r, e, t, s, n) => {
    t.backward(n.div(e)), s.backward(n.mul(r).mul(new c(-1)).div(e).div(e));
  },
  "div"
);
f(
  (r, e, t, s) => Math.pow(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(e).mul(r.pow(e.sub(new c(1))))), s.backward(n.mul(r.pow(e)).mul(r.log()));
  },
  "pow"
);
f(
  (r, e, t, s) => r[t] % e[s],
  (r, e, t, s, n) => {
    t.backward(n);
  },
  "fmod"
);
f(
  (r, e, t, s) => Math.max(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(r.ge(e))), s.backward(n.mul(e.gt(r)));
  },
  "maximum"
);
f(
  (r, e, t, s) => Math.min(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(r.le(e))), s.backward(n.mul(e.lt(r)));
  },
  "minimum"
);
function le(r, e, t = null) {
  const s = new Array(r.dataLength());
  for (let n = 0; n < s.length; n++)
    s[n] = Math.pow(r.data[n], e);
  return new c(
    s,
    { requires_grad: r.requires_grad },
    { operation: t, shape: r.shape }
  );
}
class _e extends E {
  n;
  _forward(e, t) {
    return e.requires_grad && (this.saved_tensors = [e], this.n = t), this.next_functions.push(e.grad_fn ? e.grad_fn : x), le(e, t, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, s = this.n, [n] = this.next_functions;
    n.backward(e.mul(s).mul(t.pow(s - 1)));
  }
}
q("powint", _e);
w(
  (r, e) => Math.log(r[e]),
  (r, e, t) => {
    e.backward(t.mul(new c(1).div(r)));
  },
  "log"
);
w(
  (r, e) => Math.sqrt(r[e]),
  (r, e, t) => {
    e.backward(t.mul(new c(1).div(r.sqrt()).div(2)));
  },
  "sqrt"
);
w(
  (r, e) => Math.exp(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.exp())));
  },
  "exp"
);
w(
  (r, e) => r[e] * r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(r).mul(new c(2))));
  },
  "square"
);
w(
  (r, e) => Math.abs(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(de(r))));
  },
  "abs"
);
w(
  (r, e) => Math.sign(r[e]),
  (r, e, t) => {
    e.backward(0);
  },
  "sign"
);
w(
  (r, e) => -r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(new c(-1))));
  },
  "neg"
);
w(
  (r, e) => 1 / r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.pow(-2))).neg());
  },
  "reciprocal"
);
class pe extends E {
  _forward(e, t) {
    const s = e.dataLength(), n = t.reduce((i, a) => i * a, 1);
    if (s !== n)
      throw new Error("Shape mismatch: " + e.shape + " and " + t);
    if (e.requires_grad && (this.saved_tensors = [e]), e.grad_fn)
      this.next_functions.push(e.grad_fn);
    else if (e.requires_grad) {
      const i = new T();
      i.variable = e, this.next_functions.push(i);
    } else
      this.next_functions.push(x);
    return new c(
      e.data,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: t }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.reshape(t.shape));
  }
}
q("reshape", pe);
class fe extends E {
  _forward(e, t) {
    if (e.requires_grad && (this.saved_tensors = [e]), e.grad_fn)
      this.next_functions.push(e.grad_fn);
    else if (e.requires_grad) {
      const n = new T();
      n.variable = e, this.next_functions.push(n);
    } else
      this.next_functions.push(x);
    let s = [...e.shape];
    return t !== void 0 ? (t < 0 && (t += e.shape.length), s[t] === 1 && s.splice(t, 1)) : s = s.filter((n) => n !== 1), new c(
      e.data,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: s }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.reshape(t.shape));
  }
}
q("squeeze", fe);
class ge extends E {
  _forward(e, t) {
    if (e.requires_grad && (this.saved_tensors = [e]), e.grad_fn)
      this.next_functions.push(e.grad_fn);
    else if (e.requires_grad) {
      const n = new T();
      n.variable = e, this.next_functions.push(n);
    } else
      this.next_functions.push(x);
    t < 0 && (t += e.shape.length + 1);
    const s = [...e.shape];
    return s.splice(t, 0, 1), new c(
      e.data,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: s }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.reshape(t.shape));
  }
}
q("unsqueeze", ge);
w(
  (r, e) => Math.sin(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.cos())));
  },
  "sin"
);
w(
  (r, e) => Math.cos(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.sin().neg())));
  },
  "cos"
);
w(
  (r, e) => Math.tan(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.cos().pow(-2))));
  },
  "tan"
);
function me(r, e = null) {
  return new c(
    r.toFlatArray().reduce((t, s) => t + s, 0),
    { requires_grad: r.requires_grad },
    { operation: e }
  );
}
class we extends P {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : x), me(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(R(t).add(e.item()));
  }
}
q("sum", we);
function be(r, e = null) {
  return new c(
    r.toFlatArray().reduce((t, s) => t + s, 0) / r.dataLength(),
    { requires_grad: r.requires_grad },
    { operation: e }
  );
}
class xe extends P {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : x), be(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(R(t).add(e.item() / t.dataLength()));
  }
}
q("mean", xe);
function qe(r, e, t, s = null) {
  if (r.shape.length + e < 0 || r.shape.length + t < 0)
    throw new Error(`Transpose: Dimension out of range (${e} and ${t})`);
  e = e < 0 ? r.shape.length + e : e, t = t < 0 ? r.shape.length + t : t;
  const n = [...r.shape];
  [n[e], n[t]] = [n[t], n[e]];
  const i = r.dataLength(), a = new Array(i), o = new Array(r.shape.length), h = new Array(n.length);
  for (let u = r.shape.length - 1, d = 1; u >= 0; u--)
    o[u] = d, d *= r.shape[u];
  for (let u = n.length - 1, d = 1; u >= 0; u--)
    h[u] = d, d *= n[u];
  for (let u = 0; u < i; u++) {
    let d = u, l = 0;
    for (let g = 0; g < n.length; g++) {
      const p = h[g], k = Math.floor(d / p);
      d %= p;
      let b = g;
      g === e ? b = t : g === t && (b = e), l += k * o[b];
    }
    a[u] = r.data[l];
  }
  return new c(
    a,
    { requires_grad: r.requires_grad },
    { operation: s, shape: n }
  );
}
class ye extends E {
  dim0;
  dim1;
  _forward(e, t, s) {
    return e.requires_grad && (this.saved_tensors = [e], this.dim0 = t, this.dim1 = s), this.next_functions.push(e.grad_fn ? e.grad_fn : x), qe(e, t, s, this);
  }
  _backward(e) {
    const [t] = this.saved_tensors, s = this.dim0, n = this.dim1, [i] = this.next_functions;
    i.backward(e.transpose(s, n));
  }
}
q("transpose", ye);
function Ae(r, e, t = null) {
  if (r.shape.length == 1 && e.shape.length == 1)
    return [r.mul(e).sum(), []];
  const s = r.shape.length == 1, n = e.shape.length == 1, i = s ? [1, r.shape[0]] : r.shape, a = n ? [e.shape[0], 1] : e.shape;
  if (i[i.length - 1] != a[a.length - 2])
    throw new Error("Shape mismatch: " + r.shape + " and " + e.shape);
  const o = $(i.slice(0, -2), a.slice(0, -2)).concat([
    i[i.length - 2],
    a[a.length - 1]
  ]), h = o.reduce((y, U) => y * U, 1), u = new Array(h).fill(0), d = M(i, o), l = M(a, o), g = o[o.length - 2], p = o[o.length - 1], k = i[i.length - 1];
  for (let y = 0; y < h; y++) {
    const U = y % (g * p), te = Math.floor(U / p), re = U % p;
    let se = B(d, o, y - re), ne = B(l, o, y - te * p), G = 0;
    for (let L = 0; L < k; L++)
      G += r.data[se + L] * e.data[ne + L * p];
    u[y] = G;
  }
  let b = [...o];
  return s && (b = b.slice(0, -2).concat([o[o.length - 1]])), n && (b = b.slice(0, -1)), [new c(
    u,
    { requires_grad: r.requires_grad || e.requires_grad },
    { operation: t, shape: b }
  ), b];
}
class ve extends J {
  shape;
  _forward(e, t) {
    (e.requires_grad || t.requires_grad) && (this.saved_tensors = [e, t]), this.next_functions.push(e.grad_fn ? e.grad_fn : x), this.next_functions.push(t.grad_fn ? t.grad_fn : x);
    const s = Ae(e, t, e.requires_grad || t.requires_grad ? this : null);
    return this.shape = s[1], s[0];
  }
  _backward(e) {
    const [t, s] = this.saved_tensors, [n, i] = this.next_functions;
    if (t.shape.length === 1 && s.shape.length === 1) {
      n.backward(e.mul(s)), i.backward(e.mul(t));
      return;
    }
    if (t.shape.length === 1) {
      const h = e.unsqueeze(-2), u = t.unsqueeze(-2);
      let d = h.matmul(s.transpose(-2, -1)), l = u.transpose(-2, -1).matmul(h);
      d = d.squeeze(-2), l = S(l, s.shape), n.backward(d), i.backward(l);
      return;
    }
    if (s.shape.length === 1) {
      const h = e.unsqueeze(-1), u = s.unsqueeze(-1);
      let d = h.matmul(u.transpose(-2, -1)), l = t.transpose(-2, -1).matmul(h);
      d = S(d, t.shape), l = l.squeeze(-1), n.backward(d), i.backward(l);
      return;
    }
    let a = e.matmul(s.transpose(-2, -1)), o = t.transpose(-2, -1).matmul(e);
    a = S(a, t.shape), o = S(o, s.shape), n.backward(a), i.backward(o);
  }
}
q("matmul", ve);
f(
  (r, e, t, s) => r[t] < e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "lt"
);
f(
  (r, e, t, s) => r[t] > e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "gt"
);
f(
  (r, e, t, s) => r[t] <= e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "le"
);
f(
  (r, e, t, s) => r[t] >= e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "ge"
);
f(
  (r, e, t, s) => r[t] == e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "eq"
);
f(
  (r, e, t, s) => r[t] != e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "ne"
);
w(
  (r, e) => Math.max(r[e], 0),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.gt(0))));
  },
  "relu"
);
w(
  (r, e) => 1 / (1 + Math.exp(-r[e])),
  (r, e, t) => {
    const s = r.sigmoid();
    e.backward(s.mul(s.mul(-1).add(1)).mul(t));
  },
  "sigmoid"
);
class F extends c {
  constructor(e, t = {
    requires_grad: !0
  }, s = {}) {
    e instanceof c ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : e instanceof F ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : super(e, t, s);
  }
}
class C {
  _modules;
  _parameters;
  constructor() {
    this._parameters = {}, this._modules = {};
  }
  register_parameter(e, t) {
    this._parameters[e] = t;
  }
  register_module(e, t) {
    this._modules[e] = t;
  }
  register(e, t) {
    t instanceof F ? this.register_parameter(e, t) : this.register_module(e, t);
  }
  parameters() {
    let e = Object.values(this._parameters);
    for (const t of Object.values(this._modules))
      e = e.concat(t.parameters());
    return e;
  }
}
class Oe extends C {
  weight;
  bias;
  constructor(e, t) {
    super();
    const s = Math.sqrt(1 / e);
    this.weight = new F(
      K([t, e]).mul(2 * s).sub(s)
    ), this.bias = new F(
      K([t]).mul(2 * s).sub(s)
    ), this.register("weight", this.weight), this.register("bias", this.bias);
  }
  forward(e) {
    return e.matmul(this.weight.transpose(0, 1)).add(this.bias);
  }
}
class Ee extends C {
  constructor() {
    super();
  }
  forward(e) {
    return Z(e);
  }
}
class ke extends C {
  constructor() {
    super();
  }
  forward(e) {
    return ee(e);
  }
}
class Re extends C {
  _modulesArr;
  constructor(...e) {
    super(), this._modulesArr = e;
    for (let t = 0; t < e.length; t++)
      this.register(t.toString(), e[t]);
  }
  append(e) {
    return this.register(this._modulesArr.length.toString(), e), this._modulesArr.push(e), this;
  }
  extend(e) {
    for (const t of e._modulesArr)
      this.append(t);
    return this;
  }
  insert(e, t) {
    this._modulesArr.splice(e, 0, t);
    for (let s = e; s < this._modulesArr.length; s++)
      this.register(s.toString(), this._modulesArr[s]);
    return this;
  }
  forward(e) {
    let t = e;
    for (const s of this._modulesArr)
      t = s.forward(t);
    return t;
  }
}
class W {
}
class Fe extends W {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).pow(2).mean();
  }
}
class Me extends W {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).abs().mean();
  }
}
class Be extends W {
  weight;
  constructor(e = null) {
    super(), this.weight = e;
  }
  forward(e, t) {
    const s = t.mul(e.log()), n = t.neg().add(1).mul(e.neg().add(1).log()), i = s.add(n).neg().mean();
    return this.weight ? i.mul(this.weight) : i;
  }
}
function Y(r) {
  return (e) => (typeof e == "number" && (e = new c(e)), new (O(r))().forward(e));
}
const Z = Y("relu"), ee = Y("sigmoid"), Te = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  relu: Z,
  sigmoid: ee
}, Symbol.toStringTag, { value: "Module" })), wt = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  BCELoss: Be,
  L1Loss: Me,
  Linear: Oe,
  MSELoss: Fe,
  Module: C,
  Parameter: F,
  ReLU: Ee,
  Sequential: Re,
  Sigmoid: ke,
  functional: Te
}, Symbol.toStringTag, { value: "Module" }));
class z {
  params;
  defaults;
  constructor(e, t) {
    this.params = e, this.defaults = t;
  }
  zero_grad() {
    for (const e of this.params)
      e.grad = null;
  }
}
class Ce extends z {
  state = /* @__PURE__ */ new Map();
  lr;
  momentum;
  dampening;
  weight_decay;
  nesterov;
  maximize;
  constructor(e, t = 1e-3, s = 0, n = 0, i = 0, a = !1, o = !1) {
    super(e, {}), this.lr = t, this.momentum = s, this.dampening = n, this.weight_decay = i, this.nesterov = a, this.maximize = o;
  }
  step() {
    for (const e of this.params) {
      let t = this.maximize ? e.grad.mul(-1) : e.grad;
      if (this.weight_decay !== 0 && (t = t.add(e.mul(this.weight_decay))), this.momentum !== 0) {
        if (this.state.has(e)) {
          let i = this.state.get(e).velocity;
          i = i.mul(this.momentum), i = i.add(t.mul(1 - this.dampening)), this.state.set(e, { velocity: i });
        } else
          this.state.set(e, { velocity: t });
        let n = this.state.get(e).velocity;
        this.nesterov ? t = t.add(n.mul(this.momentum)) : t = n, this.state.set(e, { velocity: n });
      }
      const s = e.sub(t.mul(this.lr));
      e.data = s.data;
    }
  }
}
class Ue extends z {
  state = /* @__PURE__ */ new Map();
  step_count = 0;
  lr;
  beta1;
  beta2;
  eps;
  weight_decay;
  amsgrad;
  maximize;
  constructor(e, t = 1e-3, s = [0.9, 0.999], n = 1e-8, i = 0, a = !1, o = !1) {
    super(e, {}), this.lr = t, this.beta1 = s[0], this.beta2 = s[1], this.eps = n, this.weight_decay = i, this.amsgrad = a, this.maximize = o;
  }
  step() {
    this.step_count += 1;
    for (const e of this.params) {
      let t = this.maximize ? e.grad.mul(-1) : e.grad;
      this.weight_decay !== 0 && (t = t.add(e.mul(this.weight_decay))), this.state.has(e) || this.state.set(e, {
        m: R(e),
        v: R(e),
        vmax: R(e)
      });
      const s = this.state.get(e);
      s.m = s.m.mul(this.beta1).add(t.mul(1 - this.beta1)), s.v = s.v.mul(this.beta2).add(t.mul(t).mul(1 - this.beta2));
      const n = 1 - Math.pow(this.beta1, this.step_count), i = 1 - Math.pow(this.beta2, this.step_count);
      let a;
      const o = s.m.div(n);
      this.amsgrad ? (s.vmax = s.vmax.maximum(s.v), a = s.vmax.div(i)) : a = s.v.div(i);
      const h = o.div(a.sqrt().add(this.eps)).mul(this.lr), u = e.sub(h);
      e.data = u.data;
    }
  }
}
const bt = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Adam: Ue,
  Optimizer: z,
  SGD: Ce
}, Symbol.toStringTag, { value: "Module" }));
export {
  T as AccumulateGrad,
  ve as Matmul,
  xe as Mean,
  _e as PowInt,
  pe as Reshape,
  fe as Squeeze,
  we as Sum,
  c as Tensor,
  E as TorchFunction,
  ye as Transpose,
  ge as Unsqueeze,
  Ne as __left_index__,
  We as __right_index__,
  et as abs,
  ze as add,
  Ie as arange,
  it as cos,
  je as div,
  gt as eq,
  A as eventBus,
  v as events,
  Ye as exp,
  Ve as fmod,
  ft as ge,
  _t as gt,
  pt as le,
  Pe as linspace,
  Qe as log,
  lt,
  dt as matmul,
  He as maximum,
  ct as mean,
  Je as minimum,
  Ke as mul,
  mt as ne,
  tt as neg,
  wt as nn,
  ae as ones,
  De as ones_like,
  bt as optim,
  $e as pow,
  K as rand,
  Se as randint,
  Le as randn,
  rt as reciprocal,
  st as reshape,
  de as sign,
  at as sin,
  Xe as sqrt,
  Ze as square,
  Ge as sub,
  ut as sum,
  ot as tan,
  ht as transpose,
  nt as unsqueeze,
  ie as zeros,
  R as zeros_like
};
