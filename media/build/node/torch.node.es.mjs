function Y(s, e) {
  const t = Math.max(s.length, e.length), r = [...Array(t - s.length).fill(1), ...s], n = [...Array(t - e.length).fill(1), ...e], i = [];
  for (let a = 0; a < t; a++) {
    if (r[a] !== n[a] && r[a] !== 1 && n[a] !== 1)
      throw new Error(`Shape mismatch: ${s} and ${e}`);
    i.push(Math.max(r[a], n[a]));
  }
  return i;
}
function Z(s, e, t) {
  const r = D(e, s), n = new Array(e.reduce((i, a) => i * a, 1)).fill(0);
  for (let i = 0; i < t.length; i++)
    n[I(r, s, i)] += t[i];
  return n;
}
function D(s, e) {
  return s.length >= e.length ? s : [...Array(e.length - s.length).fill(1), ...s];
}
function I(s, e, t) {
  let r = 0, n = 1, i = t;
  for (let a = s.length - 1; a >= 0; a--) {
    if (s[a] > 1) {
      const o = i % e[a];
      r = r + o * n;
    }
    n *= s[a], i = Math.floor(i / e[a]);
  }
  return r;
}
function W(s) {
  return Array.isArray(s[0]) ? s[0] : s;
}
function Ie(...s) {
  const e = W(s), t = new h(Array(e.reduce((r, n) => r * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
function H(...s) {
  const e = W(s), t = new h(Array(e.reduce((r, n) => r * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
function Pe(s, e, t) {
  const r = new h(
    Array(t.reduce((n, i) => n * i, 1)).fill(Math.floor(Math.random() * (e - s) + s))
  );
  return r.shape = t, r;
}
function ee(...s) {
  const e = W(s), t = new h(Array(e.reduce((r, n) => r * n, 1)).fill(1));
  return t.shape = e, t;
}
function ce(...s) {
  const e = W(s), t = new h(Array(e.reduce((r, n) => r * n, 1)).fill(0));
  return t.shape = e, t;
}
function ze(s) {
  return ee(s.shape);
}
function N(s) {
  return ce(s.shape);
}
function Ne(s, e, t) {
  const r = [], n = (e - s) / (t - 1);
  for (let i = 0; i < t - 1; i++)
    r.push(s + i * n);
  return r.push(e), new h(r);
}
function We(s, e = void 0, t = 1) {
  const r = [];
  for (let n = s; n < e; n += t)
    r.push(n);
  return new h(r);
}
let he = 0;
const te = () => he++, k = new EventTarget(), R = {
  TENSOR_BEFORE_BACKWARD: "tensor.beforeBackward",
  TENSOR_AFTER_BACKWARD: "tensor.afterBackward",
  OPERATION_BEFORE_FORWARD: "operation.beforeForward",
  OPERATION_AFTER_FORWARD: "operation.afterForward",
  OPERATION_BEFORE_BACKWARD: "operation.beforeBackward",
  OPERATION_AFTER_BACKWARD: "operation.afterBackward",
  OPERATION_BEFORE_ACCUMULATE_GRAD: "operation.beforeAccumulateGrad",
  OPERATION_AFTER_ACCUMULATE_GRAD: "operation.afterAccumulateGrad"
};
function de(...s) {
  for (const e of s)
    if (e instanceof h && e.requires_grad)
      return !0;
  return !1;
}
class A {
  id = te();
  next_functions = [];
  saved_tensors = [];
  _retained_tensors = [];
  forward(...e) {
    const t = de(...e);
    k.dispatchEvent(new CustomEvent(R.OPERATION_BEFORE_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e
      }
    }));
    const r = this._forward(...e);
    return k.dispatchEvent(new CustomEvent(R.OPERATION_AFTER_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e,
        result: r
      }
    })), r;
  }
  backward(e) {
    k.dispatchEvent(new CustomEvent(R.OPERATION_BEFORE_BACKWARD, { detail: { operation: this, dz: e } }));
    for (const t of this._retained_tensors)
      t.grad || (t.grad = new h(new Array(t.dataLength()).fill(0))), t.grad = t.grad.add(e);
    this._backward(e), k.dispatchEvent(new CustomEvent(R.OPERATION_AFTER_BACKWARD, { detail: { operation: this, dz: e } }));
  }
}
class le extends A {
  _forward(...e) {
    throw new Error("NullOp should not be called");
  }
  _backward(e) {
  }
}
const b = new le();
class se extends A {
}
class re extends A {
}
class J extends se {
  variable;
  _forward(e) {
    return this.variable = e, e;
  }
  _backward(e) {
    if (this.variable.grad || (this.variable.grad = N(this.variable)), k.dispatchEvent(new CustomEvent(R.OPERATION_BEFORE_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } })), typeof e == "number")
      this.variable.grad = this.variable.grad.add(e);
    else {
      const t = Z(e.shape, this.variable.shape, e.data);
      this.variable.grad = this.variable.grad.add(new h(t, {}, { shape: this.variable.shape }));
    }
    k.dispatchEvent(new CustomEvent(R.OPERATION_AFTER_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } }));
  }
}
const ne = /* @__PURE__ */ new Map(), j = /* @__PURE__ */ new Map();
function v(s, e) {
  ne.set(s, e);
}
function F(s) {
  const e = ne.get(s);
  if (!e)
    throw new Error(`Operation '${s}' is not registered.`);
  return e;
}
function Q(s) {
  const e = j.get(s);
  return e || (j.set(s, new (F(s))()), j.get(s));
}
function _e(s) {
  if (ArrayBuffer.isView(s))
    return [s.length];
  const e = [];
  for (; Array.isArray(s); )
    e.push(s.length), s = s[0];
  return e;
}
function ae(s) {
  return Array.isArray(s) ? s.flatMap((e) => ae(e)) : ArrayBuffer.isView(s) ? Array.from(s) : [s];
}
class h {
  // Auto-generated ID
  id = te();
  // Optional user-defined name
  name = null;
  data;
  grad_fn = null;
  grad = null;
  requires_grad;
  _shape;
  constructor(e, t = {}, r = {}) {
    if (this.data = ae(e), this.requires_grad = t.requires_grad ?? !1, t.name && (this.name = t.name), this._shape = r.shape ?? _e(e), this.grad_fn = r.operation ?? null, this.requires_grad && !this.grad_fn) {
      const n = new J();
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
    const t = this.data, r = (n) => {
      const i = this.shape[n], a = new Array(i), o = n === this.shape.length - 1;
      for (let c = 0; c < i; c++)
        o ? a[c] = t[e++] : a[c] = r(n + 1);
      return a;
    };
    return r(0);
  }
  dataLength() {
    return this.data.length;
  }
  set shape(e) {
    this._shape = e;
  }
  _executeUnaryOp(e) {
    return (this.requires_grad ? new (F(e))() : Q(e)).forward(this);
  }
  _executeBinaryOp(e, t) {
    return typeof t == "number" && (t = new h(t)), (this.requires_grad || t.requires_grad ? new (F(e))() : Q(e)).forward(this, t);
  }
  _executeOpRaw(e, ...t) {
    return new (F(e))().forward(this, ...t);
  }
  item() {
    if (this.dataLength() !== 1)
      throw new Error("Tensor.item() is only valid for scalars");
    return this.data[0];
  }
  detach() {
    return new h(this.data, { requires_grad: !1 }, { shape: this.shape });
  }
  detach_() {
    this.requires_grad = !1, this.grad = null, this.grad_fn = null;
  }
  zero_() {
    this.data = Array(this.dataLength()).fill(0);
  }
  is_retain_grad = !1;
  retain_grad() {
    this.grad_fn instanceof J || this.is_retain_grad || (this.is_retain_grad = !0, this.grad_fn._retained_tensors.push(this));
  }
  backward(e) {
    if (this.requires_grad) {
      if (e)
        e.toArray_();
      else {
        if (this.dataLength() !== 1)
          throw new Error("Gradient is required for non-scalar tensors");
        e = new h(1);
      }
      this.grad_fn && (k.dispatchEvent(new CustomEvent(R.TENSOR_BEFORE_BACKWARD, { detail: { tensor: this } })), this.grad_fn.backward(e), k.dispatchEvent(new CustomEvent(R.TENSOR_AFTER_BACKWARD, { detail: { tensor: this } })));
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
  expand(e) {
    return this._executeOpRaw("expand", e);
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
  sum(e, t = !1) {
    return this._executeOpRaw("sum", e, t);
  }
  mean(e, t = !1) {
    return this._executeOpRaw("mean", e, t);
  }
  max(e, t = !1) {
    return this._executeOpRaw("max", e, t);
  }
  min(e, t = !1) {
    return this._executeOpRaw("min", e, t);
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
function E(s) {
  return (...e) => new (F(s))().forward(...e);
}
function q(s) {
  return (e) => (typeof e == "number" && (e = new h(e)), new (F(s))().forward(e));
}
function g(s) {
  return (e, t) => (typeof e == "number" && (e = new h(e)), typeof t == "number" && (t = new h(t)), new (F(s))().forward(e, t));
}
const Ge = g("__left_index__"), je = g("__right_index__"), Ke = g("add"), $e = g("sub"), Ve = g("mul"), He = g("div"), Je = g("pow"), Qe = g("fmod"), Xe = g("maximum"), Ye = g("minimum"), Ze = q("log"), et = q("sqrt"), tt = q("exp"), st = q("square"), rt = q("abs"), pe = q("sign"), nt = q("neg"), at = q("reciprocal"), it = E("reshape"), ot = E("squeeze"), ut = E("unsqueeze"), ct = E("expand"), ht = q("sin"), dt = q("cos"), lt = q("tan"), _t = E("sum"), pt = E("mean"), ft = E("min"), gt = E("max"), mt = E("transpose"), wt = g("matmul"), xt = g("lt"), bt = g("gt"), qt = g("le"), yt = g("ge"), At = g("eq"), vt = g("ne");
function X(s) {
  const e = new Array(s.length).fill(1);
  for (let t = s.length - 2; t >= 0; t--)
    e[t] = e[t + 1] * s[t + 1];
  return e;
}
function fe(s, e) {
  return e.map((t) => {
    const r = Math.floor(s / t);
    return s %= t, r;
  });
}
function ge(s, e) {
  return s.reduce((t, r, n) => t + r * e[n], 0);
}
function K(s, e, t = !1) {
  if (e === void 0) return t ? s.map(() => 1) : [];
  const n = (Array.isArray(e) ? e : [e]).map((i) => i < 0 ? i + s.length : i);
  return t ? s.map((i, a) => n.includes(a) ? 1 : i) : s.filter((i, a) => !n.includes(a));
}
function m(s, e, t = null) {
  const r = (a, o, c, u, d, l) => {
    const _ = Array(l);
    for (let p = 0; p < l; p++) {
      const O = I(o, d, p), w = I(u, d, p);
      _[p] = s(a, c, O, w);
    }
    return _;
  }, n = (a, o, c = null) => {
    const u = Y(a.shape, o.shape), d = D(a.shape, u), l = D(o.shape, u), _ = u.reduce((p, O) => p * O, 1);
    return new h(
      r(a.data, d, o.data, l, u, _),
      { requires_grad: a.requires_grad || o.requires_grad },
      { operation: c, shape: u }
    );
  }, i = class extends re {
    _forward(a, o) {
      return (a.requires_grad || o.requires_grad) && (this.saved_tensors = [a, o]), this.next_functions.push(a.grad_fn ? a.grad_fn : b), this.next_functions.push(o.grad_fn ? o.grad_fn : b), n(a, o, a.requires_grad || o.requires_grad ? this : null);
    }
    _backward(a) {
      const [o, c] = this.saved_tensors, [u, d] = this.next_functions;
      e(o, c, u, d, a);
    }
  };
  return t && v(t, i), i;
}
function x(s, e, t = null) {
  const r = (a, o) => {
    const c = Array(o);
    for (let u = 0; u < o; u++)
      c[u] = s(a, u);
    return c;
  }, n = (a, o = null) => {
    const c = a.dataLength();
    return new h(
      r(a.data, c),
      { requires_grad: a.requires_grad },
      { operation: o, shape: a.shape }
    );
  }, i = class extends se {
    _forward(a) {
      return a.requires_grad && (this.saved_tensors = [a]), this.next_functions.push(a.grad_fn ? a.grad_fn : b), n(a, a.requires_grad ? this : null);
    }
    _backward(a) {
      const [o] = this.saved_tensors, [c] = this.next_functions;
      e(o, c, a);
    }
  };
  return t && v(t, i), i;
}
function G(s, e, t, r = null, n) {
  const i = class extends A {
    dim;
    keepdim;
    _forward(a, o, c = !1) {
      this.dim = o, this.keepdim = c, a.requires_grad && (this.saved_tensors = [a]), this.next_functions.push(a.grad_fn ? a.grad_fn : b);
      const u = K(a.shape, o, c), d = u.reduce((f, C) => f * C, 1), l = new Array(d).fill(s), _ = new Array(d).fill(0), p = X(a.shape), O = X(u), y = (o === void 0 ? [] : Array.isArray(o) ? o : [o]).map((f) => f < 0 ? f + a.shape.length : f), T = o === void 0;
      for (let f = 0; f < a.data.length; f++) {
        const C = fe(f, p);
        let M;
        if (T)
          M = c ? C.map(() => 0) : [];
        else {
          M = [];
          for (let B = 0; B < a.shape.length; B++)
            y.includes(B) ? c && M.push(0) : M.push(C[B]);
        }
        const L = ge(M, O);
        l[L] = e(l[L], a.data[f]), _[L]++;
      }
      if (n)
        for (let f = 0; f < d; f++)
          l[f] = n(l[f], _[f]);
      return new h(
        l,
        { requires_grad: a.requires_grad },
        { operation: a.requires_grad ? this : null, shape: u }
      );
    }
    _backward(a) {
      const [o] = this.saved_tensors, [c] = this.next_functions;
      let u = a;
      const d = K(o.shape, this.dim, !0);
      a.shape.length !== d.length && (u = a.reshape(d));
      let l = u.expand(o.shape);
      const _ = t(o, l, this.dim, this.keepdim);
      c.backward(_);
    }
  };
  return r && v(r, i), i;
}
function S(s, e) {
  const t = Z(s.shape, e, s.data);
  return new h(t, { requires_grad: s.requires_grad }, { shape: e });
}
function me(s, e) {
  return s.mul(ee(e));
}
m(
  (s, e, t, r) => t,
  (s, e, t, r, n) => {
  },
  "__left_index__"
);
m(
  (s, e, t, r) => r,
  (s, e, t, r, n) => {
  },
  "__right_index__"
);
m(
  (s, e, t, r) => s[t] + e[r],
  (s, e, t, r, n) => {
    t.backward(n), r.backward(n);
  },
  "add"
);
m(
  (s, e, t, r) => s[t] - e[r],
  (s, e, t, r, n) => {
    t.backward(n), r.backward(n.mul(new h(-1)));
  },
  "sub"
);
m(
  (s, e, t, r) => s[t] * e[r],
  (s, e, t, r, n) => {
    t.backward(n.mul(e)), r.backward(n.mul(s));
  },
  "mul"
);
m(
  (s, e, t, r) => s[t] / e[r],
  (s, e, t, r, n) => {
    t.backward(n.div(e)), r.backward(n.mul(s).mul(new h(-1)).div(e).div(e));
  },
  "div"
);
m(
  (s, e, t, r) => Math.pow(s[t], e[r]),
  (s, e, t, r, n) => {
    t.backward(n.mul(e).mul(s.pow(e.sub(new h(1))))), r.backward(n.mul(s.pow(e)).mul(s.log()));
  },
  "pow"
);
m(
  (s, e, t, r) => s[t] % e[r],
  (s, e, t, r, n) => {
    t.backward(n);
  },
  "fmod"
);
m(
  (s, e, t, r) => Math.max(s[t], e[r]),
  (s, e, t, r, n) => {
    t.backward(n.mul(s.ge(e))), r.backward(n.mul(e.gt(s)));
  },
  "maximum"
);
m(
  (s, e, t, r) => Math.min(s[t], e[r]),
  (s, e, t, r, n) => {
    t.backward(n.mul(s.le(e))), r.backward(n.mul(e.lt(s)));
  },
  "minimum"
);
function we(s, e, t = null) {
  const r = new Array(s.dataLength());
  for (let n = 0; n < r.length; n++)
    r[n] = Math.pow(s.data[n], e);
  return new h(
    r,
    { requires_grad: s.requires_grad },
    { operation: t, shape: s.shape }
  );
}
class xe extends A {
  n;
  _forward(e, t) {
    return e.requires_grad && (this.saved_tensors = [e], this.n = t), this.next_functions.push(e.grad_fn ? e.grad_fn : b), we(e, t, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, r = this.n, [n] = this.next_functions;
    n.backward(e.mul(r).mul(t.pow(r - 1)));
  }
}
v("powint", xe);
x(
  (s, e) => Math.log(s[e]),
  (s, e, t) => {
    e.backward(t.mul(new h(1).div(s)));
  },
  "log"
);
x(
  (s, e) => Math.sqrt(s[e]),
  (s, e, t) => {
    e.backward(t.mul(new h(1).div(s.sqrt()).div(2)));
  },
  "sqrt"
);
x(
  (s, e) => Math.exp(s[e]),
  (s, e, t) => {
    e.backward(t.mul(t.mul(s.exp())));
  },
  "exp"
);
x(
  (s, e) => s[e] * s[e],
  (s, e, t) => {
    e.backward(t.mul(t.mul(s).mul(new h(2))));
  },
  "square"
);
x(
  (s, e) => Math.abs(s[e]),
  (s, e, t) => {
    e.backward(t.mul(t.mul(pe(s))));
  },
  "abs"
);
x(
  (s, e) => Math.sign(s[e]),
  (s, e, t) => {
    e.backward(0);
  },
  "sign"
);
x(
  (s, e) => -s[e],
  (s, e, t) => {
    e.backward(t.mul(t.mul(new h(-1))));
  },
  "neg"
);
x(
  (s, e) => 1 / s[e],
  (s, e, t) => {
    e.backward(t.mul(t.mul(s.pow(-2))).neg());
  },
  "reciprocal"
);
class be extends A {
  _forward(e, t) {
    const r = e.dataLength(), n = t.reduce((i, a) => i * a, 1);
    if (r !== n)
      throw new Error("Shape mismatch: " + e.shape + " and " + t);
    return e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(b), new h(
      e.data,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: t }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [r] = this.next_functions;
    r.backward(e.reshape(t.shape));
  }
}
v("reshape", be);
class qe extends A {
  _forward(e, t) {
    e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(b);
    let r = [...e.shape];
    return t !== void 0 ? (t < 0 && (t += e.shape.length), r[t] === 1 && r.splice(t, 1)) : r = r.filter((n) => n !== 1), new h(
      e.data,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: r }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [r] = this.next_functions;
    r.backward(e.reshape(t.shape));
  }
}
v("squeeze", qe);
class ye extends A {
  _forward(e, t) {
    e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(b), t < 0 && (t += e.shape.length + 1);
    const r = [...e.shape];
    return r.splice(t, 0, 1), new h(
      e.data,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: r }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [r] = this.next_functions;
    r.backward(e.reshape(t.shape));
  }
}
v("unsqueeze", ye);
class Ae extends A {
  _forward(e, t) {
    e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(b);
    const r = t.length - e.shape.length, n = t.map((a, o) => {
      if (a === -1) {
        const c = o - r;
        return c >= 0 ? e.shape[c] : 1;
      }
      return a;
    }), i = me(e, n).data;
    return new h(
      i,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: n }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [r] = this.next_functions;
    r.backward(S(e, t.shape));
  }
}
v("expand", Ae);
x(
  (s, e) => Math.sin(s[e]),
  (s, e, t) => {
    e.backward(t.mul(t.mul(s.cos())));
  },
  "sin"
);
x(
  (s, e) => Math.cos(s[e]),
  (s, e, t) => {
    e.backward(t.mul(t.mul(s.sin().neg())));
  },
  "cos"
);
x(
  (s, e) => Math.tan(s[e]),
  (s, e, t) => {
    e.backward(t.mul(t.mul(s.cos().pow(-2))));
  },
  "tan"
);
const Ot = G(
  0,
  (s, e) => s + e,
  (s, e) => e,
  "sum"
), Et = G(
  0,
  (s, e) => s + e,
  (s, e, t) => {
    const r = K(s.shape, t, !1), n = r.length > 0 ? r.reduce((a, o) => a * o, 1) : 1, i = s.dataLength() / n;
    return e.mul(new h([1 / i]));
  },
  "mean",
  (s, e) => s / e
), kt = G(
  -1 / 0,
  (s, e) => Math.max(s, e),
  (s, e, t) => {
    const n = s.max(t, !0).expand(s.shape), i = s.eq(n).detach();
    return e.mul(i);
  },
  "max"
), Rt = G(
  1 / 0,
  (s, e) => Math.min(s, e),
  (s, e, t) => {
    const n = s.min(t, !0).expand(s.shape), i = s.eq(n).detach();
    return e.mul(i);
  },
  "min"
);
function ve(s, e, t, r = null) {
  if (s.shape.length + e < 0 || s.shape.length + t < 0)
    throw new Error(`Transpose: Dimension out of range (${e} and ${t})`);
  e = e < 0 ? s.shape.length + e : e, t = t < 0 ? s.shape.length + t : t;
  const n = [...s.shape];
  [n[e], n[t]] = [n[t], n[e]];
  const i = s.dataLength(), a = new Array(i), o = new Array(s.shape.length), c = new Array(n.length);
  for (let u = s.shape.length - 1, d = 1; u >= 0; u--)
    o[u] = d, d *= s.shape[u];
  for (let u = n.length - 1, d = 1; u >= 0; u--)
    c[u] = d, d *= n[u];
  for (let u = 0; u < i; u++) {
    let d = u, l = 0;
    for (let _ = 0; _ < n.length; _++) {
      const p = c[_], O = Math.floor(d / p);
      d %= p;
      let w = _;
      _ === e ? w = t : _ === t && (w = e), l += O * o[w];
    }
    a[u] = s.data[l];
  }
  return new h(
    a,
    { requires_grad: s.requires_grad },
    { operation: r, shape: n }
  );
}
class Oe extends A {
  dim0;
  dim1;
  _forward(e, t, r) {
    return e.requires_grad && (this.saved_tensors = [e], this.dim0 = t, this.dim1 = r), this.next_functions.push(e.grad_fn ? e.grad_fn : b), ve(e, t, r, this);
  }
  _backward(e) {
    const [t] = this.saved_tensors, r = this.dim0, n = this.dim1, [i] = this.next_functions;
    i.backward(e.transpose(r, n));
  }
}
v("transpose", Oe);
function Ee(s, e, t = null) {
  if (s.shape.length == 1 && e.shape.length == 1)
    return [s.mul(e).sum(), []];
  const r = s.shape.length == 1, n = e.shape.length == 1, i = r ? [1, s.shape[0]] : s.shape, a = n ? [e.shape[0], 1] : e.shape;
  if (i[i.length - 1] != a[a.length - 2])
    throw new Error("Shape mismatch: " + s.shape + " and " + e.shape);
  const o = Y(i.slice(0, -2), a.slice(0, -2)).concat([
    i[i.length - 2],
    a[a.length - 1]
  ]), c = o.reduce((y, T) => y * T, 1), u = new Array(c).fill(0), d = D(i, o), l = D(a, o), _ = o[o.length - 2], p = o[o.length - 1], O = i[i.length - 1];
  for (let y = 0; y < c; y++) {
    const T = y % (_ * p), f = Math.floor(T / p), C = T % p;
    let M = I(d, o, y - C), L = I(l, o, y - f * p), B = 0;
    for (let z = 0; z < O; z++)
      B += s.data[M + z] * e.data[L + z * p];
    u[y] = B;
  }
  let w = [...o];
  return r && (w = w.slice(0, -2).concat([o[o.length - 1]])), n && (w = w.slice(0, -1)), [new h(
    u,
    { requires_grad: s.requires_grad || e.requires_grad },
    { operation: t, shape: w }
  ), w];
}
class ke extends re {
  shape;
  _forward(e, t) {
    (e.requires_grad || t.requires_grad) && (this.saved_tensors = [e, t]), this.next_functions.push(e.grad_fn ? e.grad_fn : b), this.next_functions.push(t.grad_fn ? t.grad_fn : b);
    const r = Ee(e, t, e.requires_grad || t.requires_grad ? this : null);
    return this.shape = r[1], r[0];
  }
  _backward(e) {
    const [t, r] = this.saved_tensors, [n, i] = this.next_functions;
    if (t.shape.length === 1 && r.shape.length === 1) {
      n.backward(e.mul(r)), i.backward(e.mul(t));
      return;
    }
    if (t.shape.length === 1) {
      const c = e.unsqueeze(-2), u = t.unsqueeze(-2);
      let d = c.matmul(r.transpose(-2, -1)), l = u.transpose(-2, -1).matmul(c);
      d = d.squeeze(-2), l = S(l, r.shape), n.backward(d), i.backward(l);
      return;
    }
    if (r.shape.length === 1) {
      const c = e.unsqueeze(-1), u = r.unsqueeze(-1);
      let d = c.matmul(u.transpose(-2, -1)), l = t.transpose(-2, -1).matmul(c);
      d = S(d, t.shape), l = l.squeeze(-1), n.backward(d), i.backward(l);
      return;
    }
    let a = e.matmul(r.transpose(-2, -1)), o = t.transpose(-2, -1).matmul(e);
    a = S(a, t.shape), o = S(o, r.shape), n.backward(a), i.backward(o);
  }
}
v("matmul", ke);
m(
  (s, e, t, r) => s[t] < e[r] ? 1 : 0,
  (s, e, t, r) => {
  },
  "lt"
);
m(
  (s, e, t, r) => s[t] > e[r] ? 1 : 0,
  (s, e, t, r) => {
  },
  "gt"
);
m(
  (s, e, t, r) => s[t] <= e[r] ? 1 : 0,
  (s, e, t, r) => {
  },
  "le"
);
m(
  (s, e, t, r) => s[t] >= e[r] ? 1 : 0,
  (s, e, t, r) => {
  },
  "ge"
);
m(
  (s, e, t, r) => s[t] == e[r] ? 1 : 0,
  (s, e, t, r) => {
  },
  "eq"
);
m(
  (s, e, t, r) => s[t] != e[r] ? 1 : 0,
  (s, e, t, r) => {
  },
  "ne"
);
x(
  (s, e) => Math.max(s[e], 0),
  (s, e, t) => {
    e.backward(t.mul(t.mul(s.gt(0))));
  },
  "relu"
);
x(
  (s, e) => 1 / (1 + Math.exp(-s[e])),
  (s, e, t) => {
    const r = s.sigmoid();
    e.backward(r.mul(r.mul(-1).add(1)).mul(t));
  },
  "sigmoid"
);
class U extends h {
  constructor(e, t = {
    requires_grad: !0
  }, r = {}) {
    e instanceof h ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : e instanceof U ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : super(e, t, r);
  }
}
class P {
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
    t instanceof U ? this.register_parameter(e, t) : this.register_module(e, t);
  }
  parameters() {
    let e = Object.values(this._parameters);
    for (const t of Object.values(this._modules))
      e = e.concat(t.parameters());
    return e;
  }
}
class Re extends P {
  weight;
  bias;
  constructor(e, t) {
    super();
    const r = Math.sqrt(1 / e);
    this.weight = new U(
      H([t, e]).mul(2 * r).sub(r)
    ), this.bias = new U(
      H([t]).mul(2 * r).sub(r)
    ), this.register("weight", this.weight), this.register("bias", this.bias);
  }
  forward(e) {
    return e.matmul(this.weight.transpose(0, 1)).add(this.bias);
  }
}
class Fe extends P {
  constructor() {
    super();
  }
  forward(e) {
    return oe(e);
  }
}
class Me extends P {
  constructor() {
    super();
  }
  forward(e) {
    return ue(e);
  }
}
class Be extends P {
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
    for (let r = e; r < this._modulesArr.length; r++)
      this.register(r.toString(), this._modulesArr[r]);
    return this;
  }
  forward(e) {
    let t = e;
    for (const r of this._modulesArr)
      t = r.forward(t);
    return t;
  }
}
class $ {
}
class Te extends $ {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).pow(2).mean();
  }
}
class Ce extends $ {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).abs().mean();
  }
}
class Ue extends $ {
  weight;
  constructor(e = null) {
    super(), this.weight = e;
  }
  forward(e, t) {
    const r = t.mul(e.log()), n = t.neg().add(1).mul(e.neg().add(1).log()), i = r.add(n).neg().mean();
    return this.weight ? i.mul(this.weight) : i;
  }
}
function ie(s) {
  return (e) => (typeof e == "number" && (e = new h(e)), new (F(s))().forward(e));
}
const oe = ie("relu"), ue = ie("sigmoid"), Le = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  relu: oe,
  sigmoid: ue
}, Symbol.toStringTag, { value: "Module" })), Ft = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  BCELoss: Ue,
  L1Loss: Ce,
  Linear: Re,
  MSELoss: Te,
  Module: P,
  Parameter: U,
  ReLU: Fe,
  Sequential: Be,
  Sigmoid: Me,
  functional: Le
}, Symbol.toStringTag, { value: "Module" }));
class V {
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
class Se extends V {
  state = /* @__PURE__ */ new Map();
  lr;
  momentum;
  dampening;
  weight_decay;
  nesterov;
  maximize;
  constructor(e, t = 1e-3, r = 0, n = 0, i = 0, a = !1, o = !1) {
    super(e, {}), this.lr = t, this.momentum = r, this.dampening = n, this.weight_decay = i, this.nesterov = a, this.maximize = o;
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
      const r = e.sub(t.mul(this.lr));
      e.data = r.data;
    }
  }
}
class De extends V {
  state = /* @__PURE__ */ new Map();
  step_count = 0;
  lr;
  beta1;
  beta2;
  eps;
  weight_decay;
  amsgrad;
  maximize;
  constructor(e, t = 1e-3, r = [0.9, 0.999], n = 1e-8, i = 0, a = !1, o = !1) {
    super(e, {}), this.lr = t, this.beta1 = r[0], this.beta2 = r[1], this.eps = n, this.weight_decay = i, this.amsgrad = a, this.maximize = o;
  }
  step() {
    this.step_count += 1;
    for (const e of this.params) {
      let t = this.maximize ? e.grad.mul(-1) : e.grad;
      this.weight_decay !== 0 && (t = t.add(e.mul(this.weight_decay))), this.state.has(e) || this.state.set(e, {
        m: N(e),
        v: N(e),
        vmax: N(e)
      });
      const r = this.state.get(e);
      r.m = r.m.mul(this.beta1).add(t.mul(1 - this.beta1)), r.v = r.v.mul(this.beta2).add(t.mul(t).mul(1 - this.beta2));
      const n = 1 - Math.pow(this.beta1, this.step_count), i = 1 - Math.pow(this.beta2, this.step_count);
      let a;
      const o = r.m.div(n);
      this.amsgrad ? (r.vmax = r.vmax.maximum(r.v), a = r.vmax.div(i)) : a = r.v.div(i);
      const c = o.div(a.sqrt().add(this.eps)).mul(this.lr), u = e.sub(c);
      e.data = u.data;
    }
  }
}
const Mt = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Adam: De,
  Optimizer: V,
  SGD: Se
}, Symbol.toStringTag, { value: "Module" }));
export {
  J as AccumulateGrad,
  kt as Max,
  Et as Mean,
  Rt as Min,
  Ot as Sum,
  h as Tensor,
  A as TorchFunction,
  Ge as __left_index__,
  je as __right_index__,
  rt as abs,
  Ke as add,
  We as arange,
  dt as cos,
  He as div,
  At as eq,
  k as eventBus,
  R as events,
  tt as exp,
  ct as expand,
  Qe as fmod,
  yt as ge,
  bt as gt,
  qt as le,
  Ne as linspace,
  Ze as log,
  xt as lt,
  wt as matmul,
  gt as max,
  Xe as maximum,
  pt as mean,
  ft as min,
  Ye as minimum,
  Ve as mul,
  vt as ne,
  nt as neg,
  Ft as nn,
  ee as ones,
  ze as ones_like,
  Mt as optim,
  Je as pow,
  H as rand,
  Pe as randint,
  Ie as randn,
  at as reciprocal,
  it as reshape,
  pe as sign,
  ht as sin,
  et as sqrt,
  st as square,
  ot as squeeze,
  $e as sub,
  _t as sum,
  lt as tan,
  mt as transpose,
  ut as unsqueeze,
  ce as zeros,
  N as zeros_like
};
