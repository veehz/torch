function Y(r, e) {
  const t = Math.max(r.length, e.length), s = [...Array(t - r.length).fill(1), ...r], n = [...Array(t - e.length).fill(1), ...e], i = [];
  for (let a = 0; a < t; a++) {
    if (s[a] !== n[a] && s[a] !== 1 && n[a] !== 1)
      throw new Error(`Shape mismatch: ${r} and ${e}`);
    i.push(Math.max(s[a], n[a]));
  }
  return i;
}
function Z(r, e, t) {
  const s = D(e, r), n = new Array(e.reduce((i, a) => i * a, 1)).fill(0);
  for (let i = 0; i < t.length; i++)
    n[I(s, r, i)] += t[i];
  return n;
}
function D(r, e) {
  return r.length >= e.length ? r : [...Array(e.length - r.length).fill(1), ...r];
}
function I(r, e, t) {
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
function G(r) {
  return Array.isArray(r[0]) ? r[0] : r;
}
function Ie(...r) {
  const e = G(r), t = new h(Array(e.reduce((s, n) => s * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
function H(...r) {
  const e = G(r), t = new h(Array(e.reduce((s, n) => s * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
function Pe(r, e, t) {
  const s = new h(
    Array(t.reduce((n, i) => n * i, 1)).fill(Math.floor(Math.random() * (e - r) + r))
  );
  return s.shape = t, s;
}
function ee(...r) {
  const e = G(r), t = new h(Array(e.reduce((s, n) => s * n, 1)).fill(1));
  return t.shape = e, t;
}
function ce(...r) {
  const e = G(r), t = new h(Array(e.reduce((s, n) => s * n, 1)).fill(0));
  return t.shape = e, t;
}
function ze(r) {
  return ee(r.shape);
}
function W(r) {
  return ce(r.shape);
}
function We(r, e, t) {
  const s = [], n = (e - r) / (t - 1);
  for (let i = 0; i < t - 1; i++)
    s.push(r + i * n);
  return s.push(e), new h(s);
}
function Ge(r, e = void 0, t = 1) {
  const s = [];
  for (let n = r; n < e; n += t)
    s.push(n);
  return new h(s);
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
function de(...r) {
  for (const e of r)
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
    const s = this._forward(...e);
    return k.dispatchEvent(new CustomEvent(R.OPERATION_AFTER_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e,
        result: s
      }
    })), s;
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
class re extends A {
}
class se extends A {
}
class J extends re {
  variable;
  _forward(e) {
    return this.variable = e, e;
  }
  _backward(e) {
    if (this.variable.grad || (this.variable.grad = W(this.variable)), k.dispatchEvent(new CustomEvent(R.OPERATION_BEFORE_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } })), typeof e == "number")
      this.variable.grad = this.variable.grad.add(e);
    else {
      const t = Z(e.shape, this.variable.shape, e.data);
      this.variable.grad = this.variable.grad.add(new h(t, {}, { shape: this.variable.shape }));
    }
    k.dispatchEvent(new CustomEvent(R.OPERATION_AFTER_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } }));
  }
}
const ne = /* @__PURE__ */ new Map(), j = /* @__PURE__ */ new Map();
function v(r, e) {
  ne.set(r, e);
}
function F(r) {
  const e = ne.get(r);
  if (!e)
    throw new Error(`Operation '${r}' is not registered.`);
  return e;
}
function Q(r) {
  const e = j.get(r);
  return e || (j.set(r, new (F(r))()), j.get(r));
}
function _e(r) {
  if (ArrayBuffer.isView(r))
    return [r.length];
  const e = [];
  for (; Array.isArray(r); )
    e.push(r.length), r = r[0];
  return e;
}
function ae(r) {
  return Array.isArray(r) ? r.flatMap((e) => ae(e)) : ArrayBuffer.isView(r) ? Array.from(r) : [r];
}
class h {
  // Auto-generated ID
  id = te();
  // Optional user-defined name
  name = null;
  data;
  shape;
  grad_fn = null;
  grad = null;
  requires_grad;
  constructor(e, t = {}, s = {}) {
    if (this.data = ae(e), this.requires_grad = t.requires_grad ?? !1, t.name && (this.name = t.name), this.shape = s.shape ?? _e(e), this.grad_fn = s.operation ?? null, this.requires_grad && !this.grad_fn) {
      const n = new J();
      n.variable = this, this.grad_fn = n;
    }
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
      for (let c = 0; c < i; c++)
        o ? a[c] = t[e++] : a[c] = s(n + 1);
      return a;
    };
    return s(0);
  }
  dataLength() {
    return this.data.length;
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
function E(r) {
  return (...e) => new (F(r))().forward(...e);
}
function q(r) {
  return (e) => (typeof e == "number" && (e = new h(e)), new (F(r))().forward(e));
}
function g(r) {
  return (e, t) => (typeof e == "number" && (e = new h(e)), typeof t == "number" && (t = new h(t)), new (F(r))().forward(e, t));
}
const Ne = g("__left_index__"), je = g("__right_index__"), Ke = g("add"), $e = g("sub"), Ve = g("mul"), He = g("div"), Je = g("pow"), Qe = g("fmod"), Xe = g("maximum"), Ye = g("minimum"), Ze = q("log"), et = q("sqrt"), tt = q("exp"), rt = q("square"), st = q("abs"), pe = q("sign"), nt = q("neg"), at = q("reciprocal"), it = E("reshape"), ot = E("squeeze"), ut = E("unsqueeze"), ct = E("expand"), ht = q("sin"), dt = q("cos"), lt = q("tan"), _t = E("sum"), pt = E("mean"), ft = E("min"), gt = E("max"), mt = E("transpose"), wt = g("matmul"), xt = g("lt"), bt = g("gt"), qt = g("le"), yt = g("ge"), At = g("eq"), vt = g("ne");
function X(r) {
  const e = new Array(r.length).fill(1);
  for (let t = r.length - 2; t >= 0; t--)
    e[t] = e[t + 1] * r[t + 1];
  return e;
}
function fe(r, e) {
  return e.map((t) => {
    const s = Math.floor(r / t);
    return r %= t, s;
  });
}
function ge(r, e) {
  return r.reduce((t, s, n) => t + s * e[n], 0);
}
function K(r, e, t = !1) {
  if (e === void 0) return t ? r.map(() => 1) : [];
  const n = (Array.isArray(e) ? e : [e]).map((i) => i < 0 ? i + r.length : i);
  return t ? r.map((i, a) => n.includes(a) ? 1 : i) : r.filter((i, a) => !n.includes(a));
}
function m(r, e, t = null) {
  const s = (a, o, c, u, d, l) => {
    const _ = Array(l);
    for (let p = 0; p < l; p++) {
      const O = I(o, d, p), w = I(u, d, p);
      _[p] = r(a, c, O, w);
    }
    return _;
  }, n = (a, o, c = null) => {
    const u = Y(a.shape, o.shape), d = D(a.shape, u), l = D(o.shape, u), _ = u.reduce((p, O) => p * O, 1);
    return new h(
      s(
        a.data,
        d,
        o.data,
        l,
        u,
        _
      ),
      { requires_grad: a.requires_grad || o.requires_grad },
      { operation: c, shape: u }
    );
  }, i = {
    [t]: class extends se {
      _forward(a, o) {
        return (a.requires_grad || o.requires_grad) && (this.saved_tensors = [a, o]), this.next_functions.push(a.grad_fn ? a.grad_fn : b), this.next_functions.push(o.grad_fn ? o.grad_fn : b), n(a, o, a.requires_grad || o.requires_grad ? this : null);
      }
      _backward(a) {
        const [o, c] = this.saved_tensors, [u, d] = this.next_functions;
        e(o, c, u, d, a);
      }
    }
  }[t];
  return t && v(t, i), i;
}
function x(r, e, t = null) {
  const s = (a, o) => {
    const c = Array(o);
    for (let u = 0; u < o; u++)
      c[u] = r(a, u);
    return c;
  }, n = (a, o = null) => {
    const c = a.dataLength();
    return new h(
      s(a.data, c),
      { requires_grad: a.requires_grad },
      { operation: o, shape: a.shape }
    );
  }, i = {
    [t]: class extends re {
      _forward(a) {
        return a.requires_grad && (this.saved_tensors = [a]), this.next_functions.push(a.grad_fn ? a.grad_fn : b), n(a, a.requires_grad ? this : null);
      }
      _backward(a) {
        const [o] = this.saved_tensors, [c] = this.next_functions;
        e(o, c, a);
      }
    }
  }[t];
  return t && v(t, i), i;
}
function N(r, e, t, s = null, n) {
  const i = {
    [s]: class extends A {
      dim;
      keepdim;
      _forward(a, o, c = !1) {
        this.dim = o, this.keepdim = c, a.requires_grad && (this.saved_tensors = [a]), this.next_functions.push(a.grad_fn ? a.grad_fn : b);
        const u = K(a.shape, o, c), d = u.reduce((f, C) => f * C, 1), l = new Array(d).fill(r), _ = new Array(d).fill(0), p = X(a.shape), O = X(u), y = (o === void 0 ? [] : Array.isArray(o) ? o : [o]).map((f) => f < 0 ? f + a.shape.length : f), T = o === void 0;
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
    }
  }[s];
  return s && v(s, i), i;
}
function S(r, e) {
  const t = Z(r.shape, e, r.data);
  return new h(t, { requires_grad: r.requires_grad }, { shape: e });
}
function me(r, e) {
  return r.mul(ee(e));
}
m(
  (r, e, t, s) => t,
  (r, e, t, s, n) => {
  },
  "__left_index__"
);
m(
  (r, e, t, s) => s,
  (r, e, t, s, n) => {
  },
  "__right_index__"
);
m(
  (r, e, t, s) => r[t] + e[s],
  (r, e, t, s, n) => {
    t.backward(n), s.backward(n);
  },
  "add"
);
m(
  (r, e, t, s) => r[t] - e[s],
  (r, e, t, s, n) => {
    t.backward(n), s.backward(n.mul(new h(-1)));
  },
  "sub"
);
m(
  (r, e, t, s) => r[t] * e[s],
  (r, e, t, s, n) => {
    t.backward(n.mul(e)), s.backward(n.mul(r));
  },
  "mul"
);
m(
  (r, e, t, s) => r[t] / e[s],
  (r, e, t, s, n) => {
    t.backward(n.div(e)), s.backward(n.mul(r).mul(new h(-1)).div(e).div(e));
  },
  "div"
);
m(
  (r, e, t, s) => Math.pow(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(e).mul(r.pow(e.sub(new h(1))))), s.backward(n.mul(r.pow(e)).mul(r.log()));
  },
  "pow"
);
m(
  (r, e, t, s) => r[t] % e[s],
  (r, e, t, s, n) => {
    t.backward(n);
  },
  "fmod"
);
m(
  (r, e, t, s) => Math.max(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(r.ge(e))), s.backward(n.mul(e.gt(r)));
  },
  "maximum"
);
m(
  (r, e, t, s) => Math.min(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(r.le(e))), s.backward(n.mul(e.lt(r)));
  },
  "minimum"
);
function we(r, e, t = null) {
  const s = new Array(r.dataLength());
  for (let n = 0; n < s.length; n++)
    s[n] = Math.pow(r.data[n], e);
  return new h(
    s,
    { requires_grad: r.requires_grad },
    { operation: t, shape: r.shape }
  );
}
class xe extends A {
  n;
  _forward(e, t) {
    return e.requires_grad && (this.saved_tensors = [e], this.n = t), this.next_functions.push(e.grad_fn ? e.grad_fn : b), we(e, t, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, s = this.n, [n] = this.next_functions;
    n.backward(e.mul(s).mul(t.pow(s - 1)));
  }
}
v("powint", xe);
x(
  (r, e) => Math.log(r[e]),
  (r, e, t) => {
    e.backward(t.mul(new h(1).div(r)));
  },
  "log"
);
x(
  (r, e) => Math.sqrt(r[e]),
  (r, e, t) => {
    e.backward(t.mul(new h(1).div(r.sqrt()).div(2)));
  },
  "sqrt"
);
x(
  (r, e) => Math.exp(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.exp())));
  },
  "exp"
);
x(
  (r, e) => r[e] * r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(r).mul(new h(2))));
  },
  "square"
);
x(
  (r, e) => Math.abs(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(pe(r))));
  },
  "abs"
);
x(
  (r, e) => Math.sign(r[e]),
  (r, e, t) => {
    e.backward(0);
  },
  "sign"
);
x(
  (r, e) => -r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(new h(-1))));
  },
  "neg"
);
x(
  (r, e) => 1 / r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.pow(-2))).neg());
  },
  "reciprocal"
);
class be extends A {
  _forward(e, t) {
    const s = e.dataLength(), n = t.reduce((i, a) => i * a, 1);
    if (s !== n)
      throw new Error("Shape mismatch: " + e.shape + " and " + t);
    return e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(b), new h(
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
v("reshape", be);
class qe extends A {
  _forward(e, t) {
    e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(b);
    let s = [...e.shape];
    return t !== void 0 ? (t < 0 && (t += e.shape.length), s[t] === 1 && s.splice(t, 1)) : s = s.filter((n) => n !== 1), new h(
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
v("squeeze", qe);
class ye extends A {
  _forward(e, t) {
    e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(b), t < 0 && (t += e.shape.length + 1);
    const s = [...e.shape];
    return s.splice(t, 0, 1), new h(
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
v("unsqueeze", ye);
class Ae extends A {
  _forward(e, t) {
    e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(b);
    const s = t.length - e.shape.length, n = t.map((a, o) => {
      if (a === -1) {
        const c = o - s;
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
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(S(e, t.shape));
  }
}
v("expand", Ae);
x(
  (r, e) => Math.sin(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.cos())));
  },
  "sin"
);
x(
  (r, e) => Math.cos(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.sin().neg())));
  },
  "cos"
);
x(
  (r, e) => Math.tan(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.cos().pow(-2))));
  },
  "tan"
);
const Ot = N(
  0,
  (r, e) => r + e,
  (r, e) => e,
  "sum"
), Et = N(
  0,
  (r, e) => r + e,
  (r, e, t) => {
    const s = K(r.shape, t, !1), n = s.length > 0 ? s.reduce((a, o) => a * o, 1) : 1, i = r.dataLength() / n;
    return e.mul(new h([1 / i]));
  },
  "mean",
  (r, e) => r / e
), kt = N(
  -1 / 0,
  (r, e) => Math.max(r, e),
  (r, e, t) => {
    const n = r.max(t, !0).expand(r.shape), i = r.eq(n).detach();
    return e.mul(i);
  },
  "max"
), Rt = N(
  1 / 0,
  (r, e) => Math.min(r, e),
  (r, e, t) => {
    const n = r.min(t, !0).expand(r.shape), i = r.eq(n).detach();
    return e.mul(i);
  },
  "min"
);
function ve(r, e, t, s = null) {
  if (r.shape.length + e < 0 || r.shape.length + t < 0)
    throw new Error(`Transpose: Dimension out of range (${e} and ${t})`);
  e = e < 0 ? r.shape.length + e : e, t = t < 0 ? r.shape.length + t : t;
  const n = [...r.shape];
  [n[e], n[t]] = [n[t], n[e]];
  const i = r.dataLength(), a = new Array(i), o = new Array(r.shape.length), c = new Array(n.length);
  for (let u = r.shape.length - 1, d = 1; u >= 0; u--)
    o[u] = d, d *= r.shape[u];
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
    a[u] = r.data[l];
  }
  return new h(
    a,
    { requires_grad: r.requires_grad },
    { operation: s, shape: n }
  );
}
class Oe extends A {
  dim0;
  dim1;
  _forward(e, t, s) {
    return e.requires_grad && (this.saved_tensors = [e], this.dim0 = t, this.dim1 = s), this.next_functions.push(e.grad_fn ? e.grad_fn : b), ve(e, t, s, this);
  }
  _backward(e) {
    const [t] = this.saved_tensors, s = this.dim0, n = this.dim1, [i] = this.next_functions;
    i.backward(e.transpose(s, n));
  }
}
v("transpose", Oe);
function Ee(r, e, t = null) {
  if (r.shape.length == 1 && e.shape.length == 1)
    return [r.mul(e).sum(), []];
  const s = r.shape.length == 1, n = e.shape.length == 1, i = s ? [1, r.shape[0]] : r.shape, a = n ? [e.shape[0], 1] : e.shape;
  if (i[i.length - 1] != a[a.length - 2])
    throw new Error("Shape mismatch: " + r.shape + " and " + e.shape);
  const o = Y(i.slice(0, -2), a.slice(0, -2)).concat([
    i[i.length - 2],
    a[a.length - 1]
  ]), c = o.reduce((y, T) => y * T, 1), u = new Array(c).fill(0), d = D(i, o), l = D(a, o), _ = o[o.length - 2], p = o[o.length - 1], O = i[i.length - 1];
  for (let y = 0; y < c; y++) {
    const T = y % (_ * p), f = Math.floor(T / p), C = T % p;
    let M = I(d, o, y - C), L = I(l, o, y - f * p), B = 0;
    for (let z = 0; z < O; z++)
      B += r.data[M + z] * e.data[L + z * p];
    u[y] = B;
  }
  let w = [...o];
  return s && (w = w.slice(0, -2).concat([o[o.length - 1]])), n && (w = w.slice(0, -1)), [new h(
    u,
    { requires_grad: r.requires_grad || e.requires_grad },
    { operation: t, shape: w }
  ), w];
}
class ke extends se {
  shape;
  _forward(e, t) {
    (e.requires_grad || t.requires_grad) && (this.saved_tensors = [e, t]), this.next_functions.push(e.grad_fn ? e.grad_fn : b), this.next_functions.push(t.grad_fn ? t.grad_fn : b);
    const s = Ee(e, t, e.requires_grad || t.requires_grad ? this : null);
    return this.shape = s[1], s[0];
  }
  _backward(e) {
    const [t, s] = this.saved_tensors, [n, i] = this.next_functions;
    if (t.shape.length === 1 && s.shape.length === 1) {
      n.backward(e.mul(s)), i.backward(e.mul(t));
      return;
    }
    if (t.shape.length === 1) {
      const c = e.unsqueeze(-2), u = t.unsqueeze(-2);
      let d = c.matmul(s.transpose(-2, -1)), l = u.transpose(-2, -1).matmul(c);
      d = d.squeeze(-2), l = S(l, s.shape), n.backward(d), i.backward(l);
      return;
    }
    if (s.shape.length === 1) {
      const c = e.unsqueeze(-1), u = s.unsqueeze(-1);
      let d = c.matmul(u.transpose(-2, -1)), l = t.transpose(-2, -1).matmul(c);
      d = S(d, t.shape), l = l.squeeze(-1), n.backward(d), i.backward(l);
      return;
    }
    let a = e.matmul(s.transpose(-2, -1)), o = t.transpose(-2, -1).matmul(e);
    a = S(a, t.shape), o = S(o, s.shape), n.backward(a), i.backward(o);
  }
}
v("matmul", ke);
m(
  (r, e, t, s) => r[t] < e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "lt"
);
m(
  (r, e, t, s) => r[t] > e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "gt"
);
m(
  (r, e, t, s) => r[t] <= e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "le"
);
m(
  (r, e, t, s) => r[t] >= e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "ge"
);
m(
  (r, e, t, s) => r[t] == e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "eq"
);
m(
  (r, e, t, s) => r[t] != e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "ne"
);
x(
  (r, e) => Math.max(r[e], 0),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.gt(0))));
  },
  "relu"
);
x(
  (r, e) => 1 / (1 + Math.exp(-r[e])),
  (r, e, t) => {
    const s = r.sigmoid();
    e.backward(s.mul(s.mul(-1).add(1)).mul(t));
  },
  "sigmoid"
);
class U extends h {
  constructor(e, t = {
    requires_grad: !0
  }, s = {}) {
    e instanceof h ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : e instanceof U ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : super(e, t, s);
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
    const s = Math.sqrt(1 / e);
    this.weight = new U(
      H([t, e]).mul(2 * s).sub(s)
    ), this.bias = new U(
      H([t]).mul(2 * s).sub(s)
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
    const s = t.mul(e.log()), n = t.neg().add(1).mul(e.neg().add(1).log()), i = s.add(n).neg().mean();
    return this.weight ? i.mul(this.weight) : i;
  }
}
function ie(r) {
  return (e) => (typeof e == "number" && (e = new h(e)), new (F(r))().forward(e));
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
  constructor(e, t = 1e-3, s = [0.9, 0.999], n = 1e-8, i = 0, a = !1, o = !1) {
    super(e, {}), this.lr = t, this.beta1 = s[0], this.beta2 = s[1], this.eps = n, this.weight_decay = i, this.amsgrad = a, this.maximize = o;
  }
  step() {
    this.step_count += 1;
    for (const e of this.params) {
      let t = this.maximize ? e.grad.mul(-1) : e.grad;
      this.weight_decay !== 0 && (t = t.add(e.mul(this.weight_decay))), this.state.has(e) || this.state.set(e, {
        m: W(e),
        v: W(e),
        vmax: W(e)
      });
      const s = this.state.get(e);
      s.m = s.m.mul(this.beta1).add(t.mul(1 - this.beta1)), s.v = s.v.mul(this.beta2).add(t.mul(t).mul(1 - this.beta2));
      const n = 1 - Math.pow(this.beta1, this.step_count), i = 1 - Math.pow(this.beta2, this.step_count);
      let a;
      const o = s.m.div(n);
      this.amsgrad ? (s.vmax = s.vmax.maximum(s.v), a = s.vmax.div(i)) : a = s.v.div(i);
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
  Ne as __left_index__,
  je as __right_index__,
  st as abs,
  Ke as add,
  Ge as arange,
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
  We as linspace,
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
  rt as square,
  ot as squeeze,
  $e as sub,
  _t as sum,
  lt as tan,
  mt as transpose,
  ut as unsqueeze,
  ce as zeros,
  W as zeros_like
};
