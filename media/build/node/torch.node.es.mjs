function Tt(s, t) {
  const e = Math.max(s.length, t.length), n = [...Array(e - s.length).fill(1), ...s], r = [...Array(e - t.length).fill(1), ...t], i = [];
  for (let a = 0; a < e; a++) {
    if (n[a] !== r[a] && n[a] !== 1 && r[a] !== 1)
      throw new Error(`Shape mismatch: ${s} and ${t}`);
    i.push(Math.max(n[a], r[a]));
  }
  return i;
}
function Ut(s, t, e) {
  const n = ut(t, s), r = new Array(t.reduce((i, a) => i * a, 1)).fill(0);
  for (let i = 0; i < e.length; i++)
    r[ct(n, s, i)] += e[i];
  return r;
}
function ut(s, t) {
  return s.length >= t.length ? s : [...Array(t.length - s.length).fill(1), ...s];
}
function ct(s, t, e) {
  let n = 0, r = 1, i = e;
  for (let a = s.length - 1; a >= 0; a--) {
    if (s[a] > 1) {
      const o = i % t[a];
      n = n + o * r;
    }
    r *= s[a], i = Math.floor(i / t[a]);
  }
  return n;
}
function mt(s) {
  return Array.isArray(s[0]) ? s[0] : s;
}
function Me(...s) {
  const t = mt(s), e = new h(Array(t.reduce((n, r) => n * r, 1)).fill(Math.random()));
  return e.shape = t, e;
}
function pt(...s) {
  const t = mt(s), e = new h(Array(t.reduce((n, r) => n * r, 1)).fill(Math.random()));
  return e.shape = t, e;
}
function Be(s, t, e) {
  const n = new h(
    Array(e.reduce((r, i) => r * i, 1)).fill(Math.floor(Math.random() * (t - s) + s))
  );
  return n.shape = e, n;
}
function It(...s) {
  const t = mt(s), e = new h(Array(t.reduce((n, r) => n * r, 1)).fill(1));
  return e.shape = t, e;
}
function Vt(...s) {
  const t = mt(s), e = new h(Array(t.reduce((n, r) => n * r, 1)).fill(0));
  return e.shape = t, e;
}
function Ce(s) {
  return It(s.shape);
}
function ft(s) {
  return Vt(s.shape);
}
function Te(s, t, e) {
  const n = [], r = (t - s) / (e - 1);
  for (let i = 0; i < e - 1; i++)
    n.push(s + i * r);
  return n.push(t), new h(n);
}
function Ue(s, t = void 0, e = 1) {
  const n = [];
  for (let r = s; r < t; r += e)
    n.push(r);
  return new h(n);
}
let gt = !0;
function Ht() {
  return gt;
}
function Jt() {
  const s = gt;
  return gt = !1, s;
}
function Qt(s) {
  gt = s;
}
function Ie(s) {
  const t = Jt();
  try {
    return s();
  } finally {
    Qt(t);
  }
}
let Xt = 0;
const Lt = () => Xt++, Y = new EventTarget(), Z = {
  TENSOR_BEFORE_BACKWARD: "tensor.beforeBackward",
  TENSOR_AFTER_BACKWARD: "tensor.afterBackward",
  OPERATION_BEFORE_FORWARD: "operation.beforeForward",
  OPERATION_AFTER_FORWARD: "operation.afterForward",
  OPERATION_BEFORE_BACKWARD: "operation.beforeBackward",
  OPERATION_AFTER_BACKWARD: "operation.afterBackward",
  OPERATION_BEFORE_ACCUMULATE_GRAD: "operation.beforeAccumulateGrad",
  OPERATION_AFTER_ACCUMULATE_GRAD: "operation.afterAccumulateGrad"
};
function b(...s) {
  if (!Ht()) return !1;
  for (const t of s)
    if (t instanceof h && t.requires_grad)
      return !0;
  return !1;
}
class S {
  id = Lt();
  next_functions = [];
  saved_tensors = [];
  _retained_tensors = [];
  forward(...t) {
    const e = b(...t);
    Y.dispatchEvent(new CustomEvent(Z.OPERATION_BEFORE_FORWARD, {
      detail: {
        operation: this,
        requires_grad: e,
        args: t
      }
    }));
    const n = this._forward(...t);
    return Y.dispatchEvent(new CustomEvent(Z.OPERATION_AFTER_FORWARD, {
      detail: {
        operation: this,
        requires_grad: e,
        args: t,
        result: n
      }
    })), n;
  }
  backward(t) {
    Y.dispatchEvent(new CustomEvent(Z.OPERATION_BEFORE_BACKWARD, { detail: { operation: this, dz: t } }));
    for (const e of this._retained_tensors)
      e.grad || (e.grad = new h(new Array(e.dataLength()).fill(0))), e.grad = e.grad.add(t);
    this._backward(t), Y.dispatchEvent(new CustomEvent(Z.OPERATION_AFTER_BACKWARD, { detail: { operation: this, dz: t } }));
  }
}
class Yt extends S {
  _forward(...t) {
    throw new Error("NullOp should not be called");
  }
  _backward(t) {
  }
}
const x = new Yt();
class St extends S {
}
class zt extends S {
}
class Rt extends St {
  variable;
  _forward(t) {
    return this.variable = t, t;
  }
  _backward(t) {
    if (this.variable.grad || (this.variable.grad = ft(this.variable)), Y.dispatchEvent(new CustomEvent(Z.OPERATION_BEFORE_ACCUMULATE_GRAD, { detail: { operation: this, dz: t } })), typeof t == "number")
      this.variable.grad = this.variable.grad.add(t);
    else {
      const e = Ut(t.shape, this.variable.shape, t.data);
      this.variable.grad = this.variable.grad.add(new h(e, {}, { shape: this.variable.shape }));
    }
    Y.dispatchEvent(new CustomEvent(Z.OPERATION_AFTER_ACCUMULATE_GRAD, { detail: { operation: this, dz: t } }));
  }
}
const Dt = /* @__PURE__ */ new Map(), xt = /* @__PURE__ */ new Map();
function z(s, t) {
  Dt.set(s, t);
}
function H(s) {
  const t = Dt.get(s);
  if (!t)
    throw new Error(`Operation '${s}' is not registered.`);
  return t;
}
function Mt(s) {
  const t = xt.get(s);
  return t || (xt.set(s, new (H(s))()), xt.get(s));
}
function Zt(s) {
  if (ArrayBuffer.isView(s))
    return [s.length];
  const t = [];
  for (; Array.isArray(s); )
    t.push(s.length), s = s[0];
  return t;
}
function Pt(s) {
  return Array.isArray(s) ? s.flatMap((t) => Pt(t)) : ArrayBuffer.isView(s) ? Array.from(s) : [s];
}
class h {
  // Auto-generated ID
  id = Lt();
  // Optional user-defined name
  name = null;
  data;
  shape;
  grad_fn = null;
  grad = null;
  requires_grad;
  constructor(t, e = {}, n = {}) {
    if (this.data = Pt(t), this.requires_grad = e.requires_grad ?? !1, e.name && (this.name = e.name), this.shape = n.shape ?? Zt(t), this.grad_fn = n.operation ?? null, this.requires_grad && !this.grad_fn) {
      const r = new Rt();
      r.variable = this, this.grad_fn = r;
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
    let t = 0;
    const e = this.data, n = (r) => {
      const i = this.shape[r], a = new Array(i), o = r === this.shape.length - 1;
      for (let u = 0; u < i; u++)
        o ? a[u] = e[t++] : a[u] = n(r + 1);
      return a;
    };
    return n(0);
  }
  dataLength() {
    return this.data.length;
  }
  _executeUnaryOp(t) {
    return (b(this) ? new (H(t))() : Mt(t)).forward(this);
  }
  _executeBinaryOp(t, e) {
    return typeof e == "number" && (e = new h(e)), (b(this, e) ? new (H(t))() : Mt(t)).forward(this, e);
  }
  _executeOpRaw(t, ...e) {
    return new (H(t))().forward(this, ...e);
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
    this.grad_fn instanceof Rt || this.is_retain_grad || (this.is_retain_grad = !0, this.grad_fn._retained_tensors.push(this));
  }
  backward(t) {
    if (this.requires_grad) {
      if (t)
        t.toArray_();
      else {
        if (this.dataLength() !== 1)
          throw new Error("Gradient is required for non-scalar tensors");
        t = new h(1);
      }
      this.grad_fn && (Y.dispatchEvent(new CustomEvent(Z.TENSOR_BEFORE_BACKWARD, { detail: { tensor: this } })), this.grad_fn.backward(t), Y.dispatchEvent(new CustomEvent(Z.TENSOR_AFTER_BACKWARD, { detail: { tensor: this } })));
    }
  }
  // operations
  // binary pointwise
  add(t) {
    return this._executeBinaryOp("add", t);
  }
  sub(t) {
    return this._executeBinaryOp("sub", t);
  }
  mul(t) {
    return this._executeBinaryOp("mul", t);
  }
  div(t) {
    return this._executeBinaryOp("div", t);
  }
  pow(t) {
    return typeof t == "number" && t % 1 === 0 ? this._executeOpRaw("powint", t) : this._executeBinaryOp("pow", t);
  }
  fmod(t) {
    return this._executeBinaryOp("fmod", t);
  }
  maximum(t) {
    return this._executeBinaryOp("maximum", t);
  }
  minimum(t) {
    return this._executeBinaryOp("minimum", t);
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
  nan_to_num() {
    return this._executeUnaryOp("nan_to_num");
  }
  reshape(t) {
    return this._executeOpRaw("reshape", t);
  }
  squeeze(t) {
    return this._executeOpRaw("squeeze", t);
  }
  unsqueeze(t) {
    return this._executeOpRaw("unsqueeze", t);
  }
  expand(t) {
    return this._executeOpRaw("expand", t);
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
  sum(t, e = !1) {
    return this._executeOpRaw("sum", t, e);
  }
  mean(t, e = !1) {
    return this._executeOpRaw("mean", t, e);
  }
  max(t, e = !1) {
    return this._executeOpRaw("max", t, e);
  }
  min(t, e = !1) {
    return this._executeOpRaw("min", t, e);
  }
  // linalg
  transpose(t, e) {
    return this._executeOpRaw("transpose", t, e);
  }
  matmul(t) {
    return this._executeBinaryOp("matmul", t);
  }
  // comparison
  lt(t) {
    return this._executeBinaryOp("lt", t);
  }
  gt(t) {
    return this._executeBinaryOp("gt", t);
  }
  le(t) {
    return this._executeBinaryOp("le", t);
  }
  ge(t) {
    return this._executeBinaryOp("ge", t);
  }
  eq(t) {
    return this._executeBinaryOp("eq", t);
  }
  ne(t) {
    return this._executeBinaryOp("ne", t);
  }
  // other
  sigmoid() {
    return this._executeUnaryOp("sigmoid");
  }
}
function J(s) {
  return (...t) => new (H(s))().forward(...t);
}
function W(s) {
  return (t) => (typeof t == "number" && (t = new h(t)), new (H(s))().forward(t));
}
function E(s) {
  return (t, e) => (typeof t == "number" && (t = new h(t)), typeof e == "number" && (e = new h(e)), new (H(s))().forward(t, e));
}
const Le = E("__left_index__"), Se = E("__right_index__"), ze = E("add"), De = E("sub"), Pe = E("mul"), We = E("div"), Ne = E("pow"), $e = E("fmod"), Ge = E("maximum"), je = E("minimum"), Ke = W("log"), Ve = W("sqrt"), He = W("exp"), Je = W("square"), Qe = W("abs"), te = W("sign"), Xe = W("neg"), Ye = W("reciprocal"), Ze = W("nan_to_num"), ts = J("reshape"), es = J("squeeze"), ss = J("unsqueeze"), ns = J("expand"), rs = W("sin"), as = W("cos"), is = W("tan"), os = J("sum"), us = J("mean"), cs = J("min"), ds = J("max"), hs = J("transpose"), ls = E("matmul"), _s = E("lt"), fs = E("gt"), ps = E("le"), gs = E("ge"), ms = E("eq"), ws = E("ne");
function Bt(s) {
  const t = new Array(s.length).fill(1);
  for (let e = s.length - 2; e >= 0; e--)
    t[e] = t[e + 1] * s[e + 1];
  return t;
}
function ee(s, t) {
  return t.map((e) => {
    const n = Math.floor(s / e);
    return s %= e, n;
  });
}
function se(s, t) {
  return s.reduce((e, n, r) => e + n * t[r], 0);
}
function bt(s, t, e = !1) {
  if (t === void 0) return e ? s.map(() => 1) : [];
  const r = (Array.isArray(t) ? t : [t]).map((i) => i < 0 ? i + s.length : i);
  return e ? s.map((i, a) => r.includes(a) ? 1 : i) : s.filter((i, a) => !r.includes(a));
}
function M(s, t, e = null) {
  const n = (a, o, u, c, d, _) => {
    const l = Array(_);
    for (let f = 0; f < _; f++) {
      const F = ct(o, d, f), m = ct(c, d, f);
      l[f] = s(a, u, F, m);
    }
    return l;
  }, r = (a, o, u = null) => {
    const c = Tt(a.shape, o.shape), d = ut(a.shape, c), _ = ut(o.shape, c), l = c.reduce((f, F) => f * F, 1);
    return new h(
      n(
        a.data,
        d,
        o.data,
        _,
        c,
        l
      ),
      { requires_grad: b(a, o) },
      { operation: u, shape: c }
    );
  }, i = {
    [e]: class extends zt {
      _forward(a, o) {
        const u = b(a, o);
        return u && (this.saved_tensors = [a, o]), this.next_functions.push(a.grad_fn ? a.grad_fn : x), this.next_functions.push(o.grad_fn ? o.grad_fn : x), r(a, o, u ? this : null);
      }
      _backward(a) {
        const [o, u] = this.saved_tensors, [c, d] = this.next_functions;
        t(o, u, c, d, a);
      }
    }
  }[e];
  return e && z(e, i), i;
}
function I(s, t, e = null) {
  const n = (a, o) => {
    const u = Array(o);
    for (let c = 0; c < o; c++)
      u[c] = s(a, c);
    return u;
  }, r = (a, o = null) => {
    const u = a.dataLength();
    return new h(
      n(a.data, u),
      { requires_grad: b(a) },
      { operation: o, shape: a.shape }
    );
  }, i = {
    [e]: class extends St {
      _forward(a) {
        const o = b(a);
        return o && (this.saved_tensors = [a]), this.next_functions.push(a.grad_fn ? a.grad_fn : x), r(a, o ? this : null);
      }
      _backward(a) {
        const [o] = this.saved_tensors, [u] = this.next_functions;
        t(o, u, a);
      }
    }
  }[e];
  return e && z(e, i), i;
}
function wt(s, t, e, n = null, r) {
  const i = {
    [n]: class extends S {
      dim;
      keepdim;
      _forward(a, o, u = !1) {
        this.dim = o, this.keepdim = u;
        const c = b(a);
        c && (this.saved_tensors = [a]), this.next_functions.push(a.grad_fn ? a.grad_fn : x);
        const d = bt(a.shape, o, u), _ = d.reduce((p, T) => p * T, 1), l = new Array(_).fill(s), f = new Array(_).fill(0), F = Bt(a.shape), m = Bt(d), D = (o === void 0 ? [] : Array.isArray(o) ? o : [o]).map((p) => p < 0 ? p + a.shape.length : p), K = o === void 0;
        for (let p = 0; p < a.data.length; p++) {
          const T = ee(p, F);
          let B;
          if (K)
            B = u ? T.map(() => 0) : [];
          else {
            B = [];
            for (let k = 0; k < a.shape.length; k++)
              D.includes(k) ? u && B.push(0) : B.push(T[k]);
          }
          const U = se(B, m);
          l[U] = t(l[U], a.data[p]), f[U]++;
        }
        if (r)
          for (let p = 0; p < _; p++)
            l[p] = r(l[p], f[p]);
        return new h(
          l,
          { requires_grad: c },
          { operation: c ? this : null, shape: d }
        );
      }
      _backward(a) {
        const [o] = this.saved_tensors, [u] = this.next_functions;
        let c = a;
        const d = bt(o.shape, this.dim, !0);
        a.shape.length !== d.length && (c = a.reshape(d));
        const _ = c.expand(o.shape), l = e(o, _, this.dim, this.keepdim);
        u.backward(l);
      }
    }
  }[n];
  return n && z(n, i), i;
}
function ot(s, t) {
  const e = Ut(s.shape, t, s.data);
  return new h(e, { requires_grad: s.requires_grad }, { shape: t });
}
function ne(s, t) {
  return s.mul(It(t));
}
M(
  (s, t, e, n) => e,
  (s, t, e, n, r) => {
  },
  "__left_index__"
);
M(
  (s, t, e, n) => n,
  (s, t, e, n, r) => {
  },
  "__right_index__"
);
M(
  (s, t, e, n) => s[e] + t[n],
  (s, t, e, n, r) => {
    e.backward(r), n.backward(r);
  },
  "add"
);
M(
  (s, t, e, n) => s[e] - t[n],
  (s, t, e, n, r) => {
    e.backward(r), n.backward(r.mul(new h(-1)));
  },
  "sub"
);
M(
  (s, t, e, n) => s[e] * t[n],
  (s, t, e, n, r) => {
    e.backward(r.mul(t)), n.backward(r.mul(s));
  },
  "mul"
);
M(
  (s, t, e, n) => s[e] / t[n],
  (s, t, e, n, r) => {
    e.backward(r.div(t)), n.backward(r.mul(s).mul(new h(-1)).div(t).div(t));
  },
  "div"
);
function Ct(s, t, e) {
  const n = typeof e == "number" ? e : null, r = new Array(t.dataLength());
  for (let i = 0; i < r.length; i++)
    r[i] = s.data[i] ? t.data[i] : n !== null ? n : e.data[i];
  return new h(r, {}, { shape: t.shape });
}
M(
  (s, t, e, n) => Math.pow(s[e], t[n]),
  (s, t, e, n, r) => {
    const i = r.mul(t).mul(s.pow(t.sub(new h(1)))), a = r.mul(s.pow(t)).mul(s.log());
    e.backward(Ct(s.ne(0), i, i.nan_to_num())), n.backward(Ct(s.ne(0), a, 0));
  },
  "pow"
);
M(
  (s, t, e, n) => s[e] % t[n],
  (s, t, e, n, r) => {
    e.backward(r);
  },
  "fmod"
);
M(
  (s, t, e, n) => Math.max(s[e], t[n]),
  (s, t, e, n, r) => {
    const i = s.eq(t), a = s.gt(t).add(i.mul(new h(0.5))), o = t.gt(s).add(i.mul(new h(0.5)));
    e.backward(r.mul(a)), n.backward(r.mul(o));
  },
  "maximum"
);
M(
  (s, t, e, n) => Math.min(s[e], t[n]),
  (s, t, e, n, r) => {
    const i = s.eq(t), a = s.lt(t).add(i.mul(new h(0.5))), o = t.lt(s).add(i.mul(new h(0.5)));
    e.backward(r.mul(a)), n.backward(r.mul(o));
  },
  "minimum"
);
function re(s, t, e = null) {
  const n = new Array(s.dataLength());
  for (let r = 0; r < n.length; r++)
    n[r] = Math.pow(s.data[r], t);
  return new h(
    n,
    { requires_grad: b(s) },
    { operation: e, shape: s.shape }
  );
}
class ae extends S {
  n;
  _forward(t, e) {
    const n = b(t);
    return n && (this.saved_tensors = [t], this.n = e), this.next_functions.push(t.grad_fn ? t.grad_fn : x), re(t, e, n ? this : null);
  }
  _backward(t) {
    const [e] = this.saved_tensors, n = this.n, [r] = this.next_functions;
    r.backward(t.mul(n).mul(e.pow(n - 1)));
  }
}
z("powint", ae);
I(
  (s, t) => Math.log(s[t]),
  (s, t, e) => {
    t.backward(e.mul(new h(1).div(s)));
  },
  "log"
);
I(
  (s, t) => Math.sqrt(s[t]),
  (s, t, e) => {
    t.backward(e.mul(new h(1).div(s.sqrt()).div(2)));
  },
  "sqrt"
);
I(
  (s, t) => Math.exp(s[t]),
  (s, t, e) => {
    t.backward(e.mul(e.mul(s.exp())));
  },
  "exp"
);
I(
  (s, t) => s[t] * s[t],
  (s, t, e) => {
    t.backward(e.mul(e.mul(s).mul(new h(2))));
  },
  "square"
);
I(
  (s, t) => Math.abs(s[t]),
  (s, t, e) => {
    t.backward(e.mul(e.mul(te(s))));
  },
  "abs"
);
I(
  (s, t) => Math.sign(s[t]),
  (s, t, e) => {
    t.backward(0);
  },
  "sign"
);
I(
  (s, t) => -s[t],
  (s, t, e) => {
    t.backward(e.mul(e.mul(new h(-1))));
  },
  "neg"
);
I(
  (s, t) => 1 / s[t],
  (s, t, e) => {
    t.backward(e.mul(e.mul(s.pow(-2))).neg());
  },
  "reciprocal"
);
I(
  (s, t) => {
    const e = s[t];
    return Number.isNaN(e) ? 0 : e === 1 / 0 ? 34028235e31 : e === -1 / 0 ? -34028235e31 : e;
  },
  (s, t, e) => {
    t.backward(e);
  },
  "nan_to_num"
);
class ie extends S {
  _forward(t, e) {
    const n = t.dataLength(), r = e.reduce((a, o) => a * o, 1);
    if (n !== r)
      throw new Error("Shape mismatch: " + t.shape + " and " + e);
    const i = b(t);
    return i && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(x), new h(
      t.data,
      { requires_grad: i },
      { operation: i ? this : null, shape: e }
    );
  }
  _backward(t) {
    const [e] = this.saved_tensors, [n] = this.next_functions;
    n.backward(t.reshape(e.shape));
  }
}
z("reshape", ie);
class oe extends S {
  _forward(t, e) {
    const n = b(t);
    n && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(x);
    let r = [...t.shape];
    return e !== void 0 ? (e < 0 && (e += t.shape.length), r[e] === 1 && r.splice(e, 1)) : r = r.filter((i) => i !== 1), new h(
      t.data,
      { requires_grad: n },
      { operation: n ? this : null, shape: r }
    );
  }
  _backward(t) {
    const [e] = this.saved_tensors, [n] = this.next_functions;
    n.backward(t.reshape(e.shape));
  }
}
z("squeeze", oe);
class ue extends S {
  _forward(t, e) {
    const n = b(t);
    n && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(x), e < 0 && (e += t.shape.length + 1);
    const r = [...t.shape];
    return r.splice(e, 0, 1), new h(
      t.data,
      { requires_grad: n },
      { operation: n ? this : null, shape: r }
    );
  }
  _backward(t) {
    const [e] = this.saved_tensors, [n] = this.next_functions;
    n.backward(t.reshape(e.shape));
  }
}
z("unsqueeze", ue);
class ce extends S {
  _forward(t, e) {
    const n = b(t);
    n && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(x);
    const r = e.length - t.shape.length, i = e.map((o, u) => {
      if (o === -1) {
        const c = u - r;
        return c >= 0 ? t.shape[c] : 1;
      }
      return o;
    }), a = ne(t, i).data;
    return new h(
      a,
      { requires_grad: n },
      { operation: n ? this : null, shape: i }
    );
  }
  _backward(t) {
    const [e] = this.saved_tensors, [n] = this.next_functions;
    n.backward(ot(t, e.shape));
  }
}
z("expand", ce);
I(
  (s, t) => Math.sin(s[t]),
  (s, t, e) => {
    t.backward(e.mul(e.mul(s.cos())));
  },
  "sin"
);
I(
  (s, t) => Math.cos(s[t]),
  (s, t, e) => {
    t.backward(e.mul(e.mul(s.sin().neg())));
  },
  "cos"
);
I(
  (s, t) => Math.tan(s[t]),
  (s, t, e) => {
    t.backward(e.mul(e.mul(s.cos().pow(-2))));
  },
  "tan"
);
const xs = wt(
  0,
  (s, t) => s + t,
  (s, t) => t,
  "sum"
), bs = wt(
  0,
  (s, t) => s + t,
  (s, t, e) => {
    const n = bt(s.shape, e, !1), r = n.length > 0 ? n.reduce((a, o) => a * o, 1) : 1, i = s.dataLength() / r;
    return t.mul(new h([1 / i]));
  },
  "mean",
  (s, t) => s / t
), vs = wt(
  -1 / 0,
  (s, t) => Math.max(s, t),
  (s, t, e) => {
    const r = s.max(e, !0).expand(s.shape), i = s.eq(r).detach();
    return t.mul(i);
  },
  "max"
), ys = wt(
  1 / 0,
  (s, t) => Math.min(s, t),
  (s, t, e) => {
    const r = s.min(e, !0).expand(s.shape), i = s.eq(r).detach();
    return t.mul(i);
  },
  "min"
);
function de(s, t, e, n = null) {
  if (s.shape.length + t < 0 || s.shape.length + e < 0)
    throw new Error(`Transpose: Dimension out of range (${t} and ${e})`);
  t = t < 0 ? s.shape.length + t : t, e = e < 0 ? s.shape.length + e : e;
  const r = [...s.shape];
  [r[t], r[e]] = [r[e], r[t]];
  const i = s.dataLength(), a = new Array(i), o = new Array(s.shape.length), u = new Array(r.length);
  for (let c = s.shape.length - 1, d = 1; c >= 0; c--)
    o[c] = d, d *= s.shape[c];
  for (let c = r.length - 1, d = 1; c >= 0; c--)
    u[c] = d, d *= r[c];
  for (let c = 0; c < i; c++) {
    let d = c, _ = 0;
    for (let l = 0; l < r.length; l++) {
      const f = u[l], F = Math.floor(d / f);
      d %= f;
      let m = l;
      l === t ? m = e : l === e && (m = t), _ += F * o[m];
    }
    a[c] = s.data[_];
  }
  return new h(
    a,
    { requires_grad: b(s) },
    { operation: n, shape: r }
  );
}
class he extends S {
  dim0;
  dim1;
  _forward(t, e, n) {
    const r = b(t);
    return r && (this.saved_tensors = [t], this.dim0 = e, this.dim1 = n), this.next_functions.push(t.grad_fn ? t.grad_fn : x), de(t, e, n, r ? this : null);
  }
  _backward(t) {
    const [e] = this.saved_tensors, n = this.dim0, r = this.dim1, [i] = this.next_functions;
    i.backward(t.transpose(n, r));
  }
}
z("transpose", he);
function le(s, t, e = null) {
  if (s.shape.length == 1 && t.shape.length == 1)
    return [s.mul(t).sum(), []];
  const n = s.shape.length == 1, r = t.shape.length == 1, i = n ? [1, s.shape[0]] : s.shape, a = r ? [t.shape[0], 1] : t.shape;
  if (i[i.length - 1] != a[a.length - 2])
    throw new Error("Shape mismatch: " + s.shape + " and " + t.shape);
  const o = Tt(i.slice(0, -2), a.slice(0, -2)).concat([
    i[i.length - 2],
    a[a.length - 1]
  ]), u = o.reduce((R, D) => R * D, 1), c = new Array(u).fill(0), d = ut(i, o), _ = ut(a, o), l = o[o.length - 2], f = o[o.length - 1], F = i[i.length - 1];
  for (let R = 0; R < u; R++) {
    const D = R % (l * f), K = Math.floor(D / f), p = D % f, T = ct(d, o, R - p), B = ct(_, o, R - K * f);
    let U = 0;
    for (let k = 0; k < F; k++)
      U += s.data[T + k] * t.data[B + k * f];
    c[R] = U;
  }
  let m = [...o];
  return n && (m = m.slice(0, -2).concat([o[o.length - 1]])), r && (m = m.slice(0, -1)), [new h(
    c,
    { requires_grad: b(s, t) },
    { operation: e, shape: m }
  ), m];
}
class _e extends zt {
  shape;
  _forward(t, e) {
    const n = b(t, e);
    n && (this.saved_tensors = [t, e]), this.next_functions.push(t.grad_fn ? t.grad_fn : x), this.next_functions.push(e.grad_fn ? e.grad_fn : x);
    const r = le(t, e, n ? this : null);
    return this.shape = r[1], r[0];
  }
  _backward(t) {
    const [e, n] = this.saved_tensors, [r, i] = this.next_functions;
    if (e.shape.length === 1 && n.shape.length === 1) {
      r.backward(t.mul(n)), i.backward(t.mul(e));
      return;
    }
    if (e.shape.length === 1) {
      const u = t.unsqueeze(-2), c = e.unsqueeze(-2);
      let d = u.matmul(n.transpose(-2, -1)), _ = c.transpose(-2, -1).matmul(u);
      d = d.squeeze(-2), _ = ot(_, n.shape), r.backward(d), i.backward(_);
      return;
    }
    if (n.shape.length === 1) {
      const u = t.unsqueeze(-1), c = n.unsqueeze(-1);
      let d = u.matmul(c.transpose(-2, -1)), _ = e.transpose(-2, -1).matmul(u);
      d = ot(d, e.shape), _ = _.squeeze(-1), r.backward(d), i.backward(_);
      return;
    }
    let a = t.matmul(n.transpose(-2, -1)), o = e.transpose(-2, -1).matmul(t);
    a = ot(a, e.shape), o = ot(o, n.shape), r.backward(a), i.backward(o);
  }
}
z("matmul", _e);
function vt(s, t, e, n, r, i, a, o) {
  const u = typeof n == "number" ? new Array(o).fill(n) : n, c = typeof r == "number" ? new Array(o).fill(r) : r, d = typeof i == "number" ? new Array(o).fill(i) : i, _ = s.shape[0], l = s.shape[1], f = t.shape[0], F = s.shape.slice(2), m = t.shape.slice(2);
  if (l !== t.shape[1] * a)
    throw new Error(`in_channels (${l}) must be divisible by groups (${a}) and match weight.shape[1] * groups (${t.shape[1] * a})`);
  const R = F.map((C, A) => Math.floor((C + 2 * c[A] - d[A] * (m[A] - 1) - 1) / u[A] + 1)), D = [_, f, ...R], K = D.reduce((C, A) => C * A, 1), p = new Array(K).fill(0), T = (C) => {
    const A = new Array(C.length);
    let $ = 1;
    for (let N = C.length - 1; N >= 0; N--)
      A[N] = $, $ *= C[N];
    return A;
  }, B = T(s.shape), U = T(t.shape), k = T(D), rt = l / a, at = f / a;
  for (let C = 0; C < _; C++)
    for (let A = 0; A < a; A++)
      for (let $ = 0; $ < at; $++) {
        const N = A * at + $, dt = R.reduce((O, q) => O * q, 1);
        for (let O = 0; O < dt; O++) {
          const q = new Array(o);
          let G = O;
          for (let w = o - 1; w >= 0; w--)
            q[w] = G % R[w], G = Math.floor(G / R[w]);
          let j = e ? e.data[N] : 0;
          for (let w = 0; w < rt; w++) {
            const et = A * rt + w, it = m.reduce((Q, X) => Q * X, 1);
            for (let Q = 0; Q < it; Q++) {
              const X = new Array(o);
              let v = Q;
              for (let g = o - 1; g >= 0; g--)
                X[g] = v % m[g], v = Math.floor(v / m[g]);
              let lt = !0;
              const _t = new Array(o);
              for (let g = 0; g < o; g++) {
                const P = q[g] * u[g] + X[g] * d[g] - c[g];
                if (P < 0 || P >= F[g]) {
                  lt = !1;
                  break;
                }
                _t[g] = P;
              }
              if (lt) {
                let g = C * B[0] + et * B[1];
                for (let L = 0; L < o; L++) g += _t[L] * B[L + 2];
                let P = N * U[0] + w * U[1];
                for (let L = 0; L < o; L++) P += X[L] * U[L + 2];
                j += s.data[g] * t.data[P];
              }
            }
          }
          let ht = C * k[0] + N * k[1];
          for (let w = 0; w < o; w++) ht += q[w] * k[w + 2];
          p[ht] = j;
        }
      }
  return new h(p, { requires_grad: !1 }, { shape: D });
}
function yt(s, t, e, n, r, i, a, o, u, c, d) {
  const _ = typeof r == "number" ? new Array(u).fill(r) : r, l = typeof i == "number" ? new Array(u).fill(i) : i, f = typeof a == "number" ? new Array(u).fill(a) : a, F = t.shape[0], m = t.shape[1], R = e.shape[0], D = t.shape.slice(2), K = e.shape.slice(2), p = s.shape.slice(2), T = (O) => {
    const q = new Array(O.length);
    let G = 1;
    for (let j = O.length - 1; j >= 0; j--)
      q[j] = G, G *= O[j];
    return q;
  }, B = T(t.shape), U = T(e.shape), k = T(s.shape);
  let rt = null, at = null, C = null, A = null, $ = null;
  c && (A = new Array(t.dataLength()).fill(0)), d && ($ = new Array(e.dataLength()).fill(0));
  const N = m / o, dt = R / o;
  for (let O = 0; O < F; O++)
    for (let q = 0; q < o; q++)
      for (let G = 0; G < dt; G++) {
        const j = q * dt + G, ht = p.reduce((w, et) => w * et, 1);
        for (let w = 0; w < ht; w++) {
          const et = new Array(u);
          let it = w;
          for (let v = u - 1; v >= 0; v--)
            et[v] = it % p[v], it = Math.floor(it / p[v]);
          let Q = O * k[0] + j * k[1];
          for (let v = 0; v < u; v++) Q += et[v] * k[v + 2];
          const X = s.data[Q];
          for (let v = 0; v < N; v++) {
            const lt = q * N + v, _t = K.reduce((g, P) => g * P, 1);
            for (let g = 0; g < _t; g++) {
              const P = new Array(u);
              let L = g;
              for (let y = u - 1; y >= 0; y--)
                P[y] = L % K[y], L = Math.floor(L / K[y]);
              let Et = !0;
              const Ft = new Array(u);
              for (let y = 0; y < u; y++) {
                const st = et[y] * _[y] + P[y] * f[y] - l[y];
                if (st < 0 || st >= D[y]) {
                  Et = !1;
                  break;
                }
                Ft[y] = st;
              }
              if (Et) {
                let y = O * B[0] + lt * B[1];
                for (let V = 0; V < u; V++) y += Ft[V] * B[V + 2];
                let st = j * U[0] + v * U[1];
                for (let V = 0; V < u; V++) st += P[V] * U[V + 2];
                c && (A[y] += X * e.data[st]), d && ($[st] += X * t.data[y]);
              }
            }
          }
        }
      }
  if (c && (rt = new h(A, { requires_grad: !1 }, { shape: t.shape })), d && (at = new h($, { requires_grad: !1 }, { shape: e.shape })), n && n.requires_grad) {
    const O = [0];
    for (let q = 2; q < s.shape.length; q++) O.push(q);
    C = s.sum(O);
  }
  return [rt, at, C];
}
class fe extends S {
  stride;
  padding;
  dilation;
  groups;
  _forward(t, e, n, r = 1, i = 0, a = 1, o = 1) {
    const u = b(t, e, ...n ? [n] : []);
    u && (this.saved_tensors = [t, e], n && this.saved_tensors.push(n)), this.next_functions.push(t.grad_fn ? t.grad_fn : x), this.next_functions.push(e.grad_fn ? e.grad_fn : x), n && this.next_functions.push(n.grad_fn ? n.grad_fn : x), this.stride = r, this.padding = i, this.dilation = a, this.groups = o;
    const c = vt(t, e, n, r, i, a, o, 1);
    return c.requires_grad = u, c.grad_fn = u ? this : null, c;
  }
  _backward(t) {
    const e = this.saved_tensors[0], n = this.saved_tensors[1], r = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [i, a, o] = this.next_functions, [u, c, d] = yt(
      t,
      e,
      n,
      r,
      this.stride,
      this.padding,
      this.dilation,
      this.groups,
      1,
      e.requires_grad,
      n.requires_grad
    );
    e.requires_grad && i.backward(u), n.requires_grad && a.backward(c), r && r.requires_grad && o.backward(d);
  }
}
z("conv1d", fe);
class pe extends S {
  stride;
  padding;
  dilation;
  groups;
  _forward(t, e, n, r = 1, i = 0, a = 1, o = 1) {
    const u = b(t, e, ...n ? [n] : []);
    u && (this.saved_tensors = [t, e], n && this.saved_tensors.push(n)), this.next_functions.push(t.grad_fn ? t.grad_fn : x), this.next_functions.push(e.grad_fn ? e.grad_fn : x), n && this.next_functions.push(n.grad_fn ? n.grad_fn : x), this.stride = r, this.padding = i, this.dilation = a, this.groups = o;
    const c = vt(t, e, n, r, i, a, o, 2);
    return c.requires_grad = u, c.grad_fn = u ? this : null, c;
  }
  _backward(t) {
    const e = this.saved_tensors[0], n = this.saved_tensors[1], r = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [i, a, o] = this.next_functions, [u, c, d] = yt(
      t,
      e,
      n,
      r,
      this.stride,
      this.padding,
      this.dilation,
      this.groups,
      2,
      e.requires_grad,
      n.requires_grad
    );
    e.requires_grad && i.backward(u), n.requires_grad && a.backward(c), r && r.requires_grad && o.backward(d);
  }
}
z("conv2d", pe);
class ge extends S {
  stride;
  padding;
  dilation;
  groups;
  _forward(t, e, n, r = 1, i = 0, a = 1, o = 1) {
    const u = b(t, e, ...n ? [n] : []);
    u && (this.saved_tensors = [t, e], n && this.saved_tensors.push(n)), this.next_functions.push(t.grad_fn ? t.grad_fn : x), this.next_functions.push(e.grad_fn ? e.grad_fn : x), n && this.next_functions.push(n.grad_fn ? n.grad_fn : x), this.stride = r, this.padding = i, this.dilation = a, this.groups = o;
    const c = vt(t, e, n, r, i, a, o, 3);
    return c.requires_grad = u, c.grad_fn = u ? this : null, c;
  }
  _backward(t) {
    const e = this.saved_tensors[0], n = this.saved_tensors[1], r = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [i, a, o] = this.next_functions, [u, c, d] = yt(
      t,
      e,
      n,
      r,
      this.stride,
      this.padding,
      this.dilation,
      this.groups,
      3,
      e.requires_grad,
      n.requires_grad
    );
    e.requires_grad && i.backward(u), n.requires_grad && a.backward(c), r && r.requires_grad && o.backward(d);
  }
}
z("conv3d", ge);
M(
  (s, t, e, n) => s[e] < t[n] ? 1 : 0,
  (s, t, e, n) => {
  },
  "lt"
);
M(
  (s, t, e, n) => s[e] > t[n] ? 1 : 0,
  (s, t, e, n) => {
  },
  "gt"
);
M(
  (s, t, e, n) => s[e] <= t[n] ? 1 : 0,
  (s, t, e, n) => {
  },
  "le"
);
M(
  (s, t, e, n) => s[e] >= t[n] ? 1 : 0,
  (s, t, e, n) => {
  },
  "ge"
);
M(
  (s, t, e, n) => s[e] == t[n] ? 1 : 0,
  (s, t, e, n) => {
  },
  "eq"
);
M(
  (s, t, e, n) => s[e] != t[n] ? 1 : 0,
  (s, t, e, n) => {
  },
  "ne"
);
I(
  (s, t) => Math.max(s[t], 0),
  (s, t, e) => {
    t.backward(e.mul(e.mul(s.gt(0))));
  },
  "relu"
);
I(
  (s, t) => 1 / (1 + Math.exp(-s[t])),
  (s, t, e) => {
    const n = s.sigmoid();
    t.backward(n.mul(n.mul(-1).add(1)).mul(e));
  },
  "sigmoid"
);
class tt extends h {
  constructor(t, e = {
    requires_grad: !0
  }, n = {}) {
    t instanceof h ? super(t.data, { requires_grad: !0 }, { shape: t.shape }) : t instanceof tt ? super(t.data, { requires_grad: !0 }, { shape: t.shape }) : super(t, e, n);
  }
}
class nt {
  _modules;
  _parameters;
  constructor() {
    this._parameters = {}, this._modules = {};
  }
  register_parameter(t, e) {
    this._parameters[t] = e;
  }
  register_module(t, e) {
    this._modules[t] = e;
  }
  register(t, e) {
    e instanceof tt ? this.register_parameter(t, e) : this.register_module(t, e);
  }
  parameters() {
    let t = Object.values(this._parameters);
    for (const e of Object.values(this._modules))
      t = t.concat(e.parameters());
    return t;
  }
}
class me extends nt {
  _modulesArr;
  constructor(...t) {
    super(), this._modulesArr = t;
    for (let e = 0; e < t.length; e++)
      this.register(e.toString(), t[e]);
  }
  append(t) {
    return this.register(this._modulesArr.length.toString(), t), this._modulesArr.push(t), this;
  }
  extend(t) {
    for (const e of t._modulesArr)
      this.append(e);
    return this;
  }
  insert(t, e) {
    this._modulesArr.splice(t, 0, e);
    for (let n = t; n < this._modulesArr.length; n++)
      this.register(n.toString(), this._modulesArr[n]);
    return this;
  }
  forward(t) {
    let e = t;
    for (const n of this._modulesArr)
      e = n.forward(e);
    return e;
  }
}
class At {
}
class we extends At {
  constructor() {
    super();
  }
  forward(t, e) {
    return t.sub(e).pow(2).mean();
  }
}
class xe extends At {
  constructor() {
    super();
  }
  forward(t, e) {
    return t.sub(e).abs().mean();
  }
}
class be extends At {
  weight;
  constructor(t = null) {
    super(), this.weight = t;
  }
  forward(t, e) {
    const n = e.mul(t.log()), r = e.neg().add(1).mul(t.neg().add(1).log()), i = n.add(r).neg().mean();
    return this.weight ? i.mul(this.weight) : i;
  }
}
function kt(s) {
  return (...t) => new (H(s))().forward(...t);
}
function Wt(s) {
  return (t) => (typeof t == "number" && (t = new h(t)), new (H(s))().forward(t));
}
const Nt = Wt("relu"), $t = Wt("sigmoid"), Gt = kt("conv1d"), jt = kt("conv2d"), Kt = kt("conv3d"), ve = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  conv1d: Gt,
  conv2d: jt,
  conv3d: Kt,
  relu: Nt,
  sigmoid: $t
}, Symbol.toStringTag, { value: "Module" }));
class ye extends nt {
  weight;
  bias;
  constructor(t, e) {
    super();
    const n = Math.sqrt(1 / t);
    this.weight = new tt(
      pt([e, t]).mul(2 * n).sub(n)
    ), this.bias = new tt(
      pt([e]).mul(2 * n).sub(n)
    ), this.register("weight", this.weight), this.register("bias", this.bias);
  }
  forward(t) {
    return t.matmul(this.weight.transpose(0, 1)).add(this.bias);
  }
}
class Ae extends nt {
  constructor() {
    super();
  }
  forward(t) {
    return Nt(t);
  }
}
class ke extends nt {
  constructor() {
    super();
  }
  forward(t) {
    return $t(t);
  }
}
class Ot extends nt {
  weight;
  bias;
  in_channels;
  out_channels;
  kernel_size;
  stride;
  padding;
  dilation;
  groups;
  constructor(t, e, n, r, i, a, o, u, c) {
    if (super(), this.in_channels = t, this.out_channels = e, this.kernel_size = n, this.stride = r, this.padding = i, this.dilation = a, this.groups = o, t % o !== 0)
      throw new Error("in_channels must be divisible by groups");
    if (e % o !== 0)
      throw new Error("out_channels must be divisible by groups");
    const d = typeof n == "number" ? new Array(c).fill(n) : n, _ = d.reduce((f, F) => f * F, 1), l = Math.sqrt(o / (t * _));
    this.weight = new tt(
      pt([e, t / o, ...d]).mul(2 * l).sub(l)
    ), this.register("weight", this.weight), u ? (this.bias = new tt(
      pt([e]).mul(2 * l).sub(l)
    ), this.register("bias", this.bias)) : this.bias = null;
  }
}
class Oe extends Ot {
  constructor(t, e, n, r = 1, i = 0, a = 1, o = 1, u = !0) {
    super(t, e, n, r, i, a, o, u, 1);
  }
  forward(t) {
    return Gt(t, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}
class qe extends Ot {
  constructor(t, e, n, r = 1, i = 0, a = 1, o = 1, u = !0) {
    super(t, e, n, r, i, a, o, u, 2);
  }
  forward(t) {
    return jt(t, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}
class Ee extends Ot {
  constructor(t, e, n, r = 1, i = 0, a = 1, o = 1, u = !0) {
    super(t, e, n, r, i, a, o, u, 3);
  }
  forward(t) {
    return Kt(t, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}
const As = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  BCELoss: be,
  Conv1d: Oe,
  Conv2d: qe,
  Conv3d: Ee,
  L1Loss: xe,
  Linear: ye,
  MSELoss: we,
  Module: nt,
  Parameter: tt,
  ReLU: Ae,
  Sequential: me,
  Sigmoid: ke,
  functional: ve
}, Symbol.toStringTag, { value: "Module" }));
class qt {
  params;
  defaults;
  constructor(t, e) {
    this.params = t, this.defaults = e;
  }
  zero_grad() {
    for (const t of this.params)
      t.grad = null;
  }
}
class Fe extends qt {
  state = /* @__PURE__ */ new Map();
  lr;
  momentum;
  dampening;
  weight_decay;
  nesterov;
  maximize;
  constructor(t, e = 1e-3, n = 0, r = 0, i = 0, a = !1, o = !1) {
    super(t, {}), this.lr = e, this.momentum = n, this.dampening = r, this.weight_decay = i, this.nesterov = a, this.maximize = o;
  }
  step() {
    for (const t of this.params) {
      let e = this.maximize ? t.grad.mul(-1) : t.grad;
      if (this.weight_decay !== 0 && (e = e.add(t.mul(this.weight_decay))), this.momentum !== 0) {
        if (this.state.has(t)) {
          let i = this.state.get(t).velocity;
          i = i.mul(this.momentum), i = i.add(e.mul(1 - this.dampening)), this.state.set(t, { velocity: i });
        } else
          this.state.set(t, { velocity: e });
        const r = this.state.get(t).velocity;
        this.nesterov ? e = e.add(r.mul(this.momentum)) : e = r, this.state.set(t, { velocity: r });
      }
      const n = t.sub(e.mul(this.lr));
      t.data = n.data;
    }
  }
}
class Re extends qt {
  state = /* @__PURE__ */ new Map();
  step_count = 0;
  lr;
  beta1;
  beta2;
  eps;
  weight_decay;
  amsgrad;
  maximize;
  constructor(t, e = 1e-3, n = [0.9, 0.999], r = 1e-8, i = 0, a = !1, o = !1) {
    super(t, {}), this.lr = e, this.beta1 = n[0], this.beta2 = n[1], this.eps = r, this.weight_decay = i, this.amsgrad = a, this.maximize = o;
  }
  step() {
    this.step_count += 1;
    for (const t of this.params) {
      let e = this.maximize ? t.grad.mul(-1) : t.grad;
      this.weight_decay !== 0 && (e = e.add(t.mul(this.weight_decay))), this.state.has(t) || this.state.set(t, {
        m: ft(t),
        v: ft(t),
        vmax: ft(t)
      });
      const n = this.state.get(t);
      n.m = n.m.mul(this.beta1).add(e.mul(1 - this.beta1)), n.v = n.v.mul(this.beta2).add(e.mul(e).mul(1 - this.beta2));
      const r = 1 - Math.pow(this.beta1, this.step_count), i = 1 - Math.pow(this.beta2, this.step_count);
      let a;
      const o = n.m.div(r);
      this.amsgrad ? (n.vmax = n.vmax.maximum(n.v), a = n.vmax.div(i)) : a = n.v.div(i);
      const u = o.div(a.sqrt().add(this.eps)).mul(this.lr), c = t.sub(u);
      t.data = c.data;
    }
  }
}
const ks = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Adam: Re,
  Optimizer: qt,
  SGD: Fe
}, Symbol.toStringTag, { value: "Module" }));
export {
  Rt as AccumulateGrad,
  vs as Max,
  bs as Mean,
  ys as Min,
  xs as Sum,
  h as Tensor,
  S as TorchFunction,
  Le as __left_index__,
  Se as __right_index__,
  Qe as abs,
  ze as add,
  Ue as arange,
  as as cos,
  Qt as disable_no_grad,
  We as div,
  Jt as enable_no_grad,
  ms as eq,
  Y as eventBus,
  Z as events,
  He as exp,
  ns as expand,
  $e as fmod,
  gs as ge,
  fs as gt,
  Ht as is_grad_enabled,
  ps as le,
  Te as linspace,
  Ke as log,
  _s as lt,
  ls as matmul,
  ds as max,
  Ge as maximum,
  us as mean,
  cs as min,
  je as minimum,
  Pe as mul,
  Ze as nan_to_num,
  ws as ne,
  Xe as neg,
  As as nn,
  Ie as no_grad,
  It as ones,
  Ce as ones_like,
  ks as optim,
  Ne as pow,
  pt as rand,
  Be as randint,
  Me as randn,
  Ye as reciprocal,
  ts as reshape,
  te as sign,
  rs as sin,
  Ve as sqrt,
  Je as square,
  es as squeeze,
  De as sub,
  os as sum,
  is as tan,
  hs as transpose,
  ss as unsqueeze,
  Vt as zeros,
  ft as zeros_like
};
