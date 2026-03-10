function Me(s, e) {
  const t = Math.max(s.length, e.length), r = [...Array(t - s.length).fill(1), ...s], n = [...Array(t - e.length).fill(1), ...e], i = [];
  for (let a = 0; a < t; a++) {
    if (r[a] !== n[a] && r[a] !== 1 && n[a] !== 1)
      throw new Error(`Shape mismatch: ${s} and ${e}`);
    i.push(Math.max(r[a], n[a]));
  }
  return i;
}
function Be(s, e, t) {
  const r = oe(e, s), n = new Array(e.reduce((i, a) => i * a, 1)).fill(0);
  for (let i = 0; i < t.length; i++)
    n[ue(r, s, i)] += t[i];
  return n;
}
function oe(s, e) {
  return s.length >= e.length ? s : [...Array(e.length - s.length).fill(1), ...s];
}
function ue(s, e, t) {
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
function pe(s) {
  return Array.isArray(s[0]) ? s[0] : s;
}
function Ot(...s) {
  const e = pe(s), t = new h(Array(e.reduce((r, n) => r * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
function fe(...s) {
  const e = pe(s), t = new h(Array(e.reduce((r, n) => r * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
function kt(s, e, t) {
  const r = new h(
    Array(t.reduce((n, i) => n * i, 1)).fill(Math.floor(Math.random() * (e - s) + s))
  );
  return r.shape = t, r;
}
function Ce(...s) {
  const e = pe(s), t = new h(Array(e.reduce((r, n) => r * n, 1)).fill(1));
  return t.shape = e, t;
}
function Ge(...s) {
  const e = pe(s), t = new h(Array(e.reduce((r, n) => r * n, 1)).fill(0));
  return t.shape = e, t;
}
function Et(s) {
  return Ce(s.shape);
}
function _e(s) {
  return Ge(s.shape);
}
function Ft(s, e, t) {
  const r = [], n = (e - s) / (t - 1);
  for (let i = 0; i < t - 1; i++)
    r.push(s + i * n);
  return r.push(e), new h(r);
}
function Rt(s, e = void 0, t = 1) {
  const r = [];
  for (let n = s; n < e; n += t)
    r.push(n);
  return new h(r);
}
let je = 0;
const Te = () => je++, X = new EventTarget(), Y = {
  TENSOR_BEFORE_BACKWARD: "tensor.beforeBackward",
  TENSOR_AFTER_BACKWARD: "tensor.afterBackward",
  OPERATION_BEFORE_FORWARD: "operation.beforeForward",
  OPERATION_AFTER_FORWARD: "operation.afterForward",
  OPERATION_BEFORE_BACKWARD: "operation.beforeBackward",
  OPERATION_AFTER_BACKWARD: "operation.afterBackward",
  OPERATION_BEFORE_ACCUMULATE_GRAD: "operation.beforeAccumulateGrad",
  OPERATION_AFTER_ACCUMULATE_GRAD: "operation.afterAccumulateGrad"
};
function Ke(...s) {
  for (const e of s)
    if (e instanceof h && e.requires_grad)
      return !0;
  return !1;
}
class L {
  id = Te();
  next_functions = [];
  saved_tensors = [];
  _retained_tensors = [];
  forward(...e) {
    const t = Ke(...e);
    X.dispatchEvent(new CustomEvent(Y.OPERATION_BEFORE_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e
      }
    }));
    const r = this._forward(...e);
    return X.dispatchEvent(new CustomEvent(Y.OPERATION_AFTER_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e,
        result: r
      }
    })), r;
  }
  backward(e) {
    X.dispatchEvent(new CustomEvent(Y.OPERATION_BEFORE_BACKWARD, { detail: { operation: this, dz: e } }));
    for (const t of this._retained_tensors)
      t.grad || (t.grad = new h(new Array(t.dataLength()).fill(0))), t.grad = t.grad.add(e);
    this._backward(e), X.dispatchEvent(new CustomEvent(Y.OPERATION_AFTER_BACKWARD, { detail: { operation: this, dz: e } }));
  }
}
class Ve extends L {
  _forward(...e) {
    throw new Error("NullOp should not be called");
  }
  _backward(e) {
  }
}
const w = new Ve();
class Ue extends L {
}
class Le extends L {
}
class Ee extends Ue {
  variable;
  _forward(e) {
    return this.variable = e, e;
  }
  _backward(e) {
    if (this.variable.grad || (this.variable.grad = _e(this.variable)), X.dispatchEvent(new CustomEvent(Y.OPERATION_BEFORE_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } })), typeof e == "number")
      this.variable.grad = this.variable.grad.add(e);
    else {
      const t = Be(e.shape, this.variable.shape, e.data);
      this.variable.grad = this.variable.grad.add(new h(t, {}, { shape: this.variable.shape }));
    }
    X.dispatchEvent(new CustomEvent(Y.OPERATION_AFTER_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } }));
  }
}
const Ie = /* @__PURE__ */ new Map(), me = /* @__PURE__ */ new Map();
function I(s, e) {
  Ie.set(s, e);
}
function V(s) {
  const e = Ie.get(s);
  if (!e)
    throw new Error(`Operation '${s}' is not registered.`);
  return e;
}
function Fe(s) {
  const e = me.get(s);
  return e || (me.set(s, new (V(s))()), me.get(s));
}
function He(s) {
  if (ArrayBuffer.isView(s))
    return [s.length];
  const e = [];
  for (; Array.isArray(s); )
    e.push(s.length), s = s[0];
  return e;
}
function Se(s) {
  return Array.isArray(s) ? s.flatMap((e) => Se(e)) : ArrayBuffer.isView(s) ? Array.from(s) : [s];
}
class h {
  // Auto-generated ID
  id = Te();
  // Optional user-defined name
  name = null;
  data;
  shape;
  grad_fn = null;
  grad = null;
  requires_grad;
  constructor(e, t = {}, r = {}) {
    if (this.data = Se(e), this.requires_grad = t.requires_grad ?? !1, t.name && (this.name = t.name), this.shape = r.shape ?? He(e), this.grad_fn = r.operation ?? null, this.requires_grad && !this.grad_fn) {
      const n = new Ee();
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
    const t = this.data, r = (n) => {
      const i = this.shape[n], a = new Array(i), o = n === this.shape.length - 1;
      for (let u = 0; u < i; u++)
        o ? a[u] = t[e++] : a[u] = r(n + 1);
      return a;
    };
    return r(0);
  }
  dataLength() {
    return this.data.length;
  }
  _executeUnaryOp(e) {
    return (this.requires_grad ? new (V(e))() : Fe(e)).forward(this);
  }
  _executeBinaryOp(e, t) {
    return typeof t == "number" && (t = new h(t)), (this.requires_grad || t.requires_grad ? new (V(e))() : Fe(e)).forward(this, t);
  }
  _executeOpRaw(e, ...t) {
    return new (V(e))().forward(this, ...t);
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
    this.grad_fn instanceof Ee || this.is_retain_grad || (this.is_retain_grad = !0, this.grad_fn._retained_tensors.push(this));
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
      this.grad_fn && (X.dispatchEvent(new CustomEvent(Y.TENSOR_BEFORE_BACKWARD, { detail: { tensor: this } })), this.grad_fn.backward(e), X.dispatchEvent(new CustomEvent(Y.TENSOR_AFTER_BACKWARD, { detail: { tensor: this } })));
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
function H(s) {
  return (...e) => new (V(s))().forward(...e);
}
function N(s) {
  return (e) => (typeof e == "number" && (e = new h(e)), new (V(s))().forward(e));
}
function k(s) {
  return (e, t) => (typeof e == "number" && (e = new h(e)), typeof t == "number" && (t = new h(t)), new (V(s))().forward(e, t));
}
const Mt = k("__left_index__"), Bt = k("__right_index__"), Ct = k("add"), Tt = k("sub"), Ut = k("mul"), Lt = k("div"), It = k("pow"), St = k("fmod"), zt = k("maximum"), Dt = k("minimum"), Pt = N("log"), Wt = N("sqrt"), Nt = N("exp"), $t = N("square"), Gt = N("abs"), Je = N("sign"), jt = N("neg"), Kt = N("reciprocal"), Vt = H("reshape"), Ht = H("squeeze"), Jt = H("unsqueeze"), Qt = H("expand"), Xt = N("sin"), Yt = N("cos"), Zt = N("tan"), es = H("sum"), ts = H("mean"), ss = H("min"), rs = H("max"), ns = H("transpose"), as = k("matmul"), is = k("lt"), os = k("gt"), us = k("le"), cs = k("ge"), ds = k("eq"), hs = k("ne");
function Re(s) {
  const e = new Array(s.length).fill(1);
  for (let t = s.length - 2; t >= 0; t--)
    e[t] = e[t + 1] * s[t + 1];
  return e;
}
function Qe(s, e) {
  return e.map((t) => {
    const r = Math.floor(s / t);
    return s %= t, r;
  });
}
function Xe(s, e) {
  return s.reduce((t, r, n) => t + r * e[n], 0);
}
function we(s, e, t = !1) {
  if (e === void 0) return t ? s.map(() => 1) : [];
  const n = (Array.isArray(e) ? e : [e]).map((i) => i < 0 ? i + s.length : i);
  return t ? s.map((i, a) => n.includes(a) ? 1 : i) : s.filter((i, a) => !n.includes(a));
}
function F(s, e, t = null) {
  const r = (a, o, u, c, d, l) => {
    const _ = Array(l);
    for (let f = 0; f < l; f++) {
      const E = ue(o, d, f), x = ue(c, d, f);
      _[f] = s(a, u, E, x);
    }
    return _;
  }, n = (a, o, u = null) => {
    const c = Me(a.shape, o.shape), d = oe(a.shape, c), l = oe(o.shape, c), _ = c.reduce((f, E) => f * E, 1);
    return new h(
      r(
        a.data,
        d,
        o.data,
        l,
        c,
        _
      ),
      { requires_grad: a.requires_grad || o.requires_grad },
      { operation: u, shape: c }
    );
  }, i = {
    [t]: class extends Le {
      _forward(a, o) {
        return (a.requires_grad || o.requires_grad) && (this.saved_tensors = [a, o]), this.next_functions.push(a.grad_fn ? a.grad_fn : w), this.next_functions.push(o.grad_fn ? o.grad_fn : w), n(a, o, a.requires_grad || o.requires_grad ? this : null);
      }
      _backward(a) {
        const [o, u] = this.saved_tensors, [c, d] = this.next_functions;
        e(o, u, c, d, a);
      }
    }
  }[t];
  return t && I(t, i), i;
}
function S(s, e, t = null) {
  const r = (a, o) => {
    const u = Array(o);
    for (let c = 0; c < o; c++)
      u[c] = s(a, c);
    return u;
  }, n = (a, o = null) => {
    const u = a.dataLength();
    return new h(
      r(a.data, u),
      { requires_grad: a.requires_grad },
      { operation: o, shape: a.shape }
    );
  }, i = {
    [t]: class extends Ue {
      _forward(a) {
        return a.requires_grad && (this.saved_tensors = [a]), this.next_functions.push(a.grad_fn ? a.grad_fn : w), n(a, a.requires_grad ? this : null);
      }
      _backward(a) {
        const [o] = this.saved_tensors, [u] = this.next_functions;
        e(o, u, a);
      }
    }
  }[t];
  return t && I(t, i), i;
}
function ge(s, e, t, r = null, n) {
  const i = {
    [r]: class extends L {
      dim;
      keepdim;
      _forward(a, o, u = !1) {
        this.dim = o, this.keepdim = u, a.requires_grad && (this.saved_tensors = [a]), this.next_functions.push(a.grad_fn ? a.grad_fn : w);
        const c = we(a.shape, o, u), d = c.reduce((p, C) => p * C, 1), l = new Array(d).fill(s), _ = new Array(d).fill(0), f = Re(a.shape), E = Re(c), y = (o === void 0 ? [] : Array.isArray(o) ? o : [o]).map((p) => p < 0 ? p + a.shape.length : p), z = o === void 0;
        for (let p = 0; p < a.data.length; p++) {
          const C = Qe(p, f);
          let R;
          if (z)
            R = u ? C.map(() => 0) : [];
          else {
            R = [];
            for (let M = 0; M < a.shape.length; M++)
              y.includes(M) ? u && R.push(0) : R.push(C[M]);
          }
          const T = Xe(R, E);
          l[T] = e(l[T], a.data[p]), _[T]++;
        }
        if (n)
          for (let p = 0; p < d; p++)
            l[p] = n(l[p], _[p]);
        return new h(
          l,
          { requires_grad: a.requires_grad },
          { operation: a.requires_grad ? this : null, shape: c }
        );
      }
      _backward(a) {
        const [o] = this.saved_tensors, [u] = this.next_functions;
        let c = a;
        const d = we(o.shape, this.dim, !0);
        a.shape.length !== d.length && (c = a.reshape(d));
        const l = c.expand(o.shape), _ = t(o, l, this.dim, this.keepdim);
        u.backward(_);
      }
    }
  }[r];
  return r && I(r, i), i;
}
function ie(s, e) {
  const t = Be(s.shape, e, s.data);
  return new h(t, { requires_grad: s.requires_grad }, { shape: e });
}
function Ye(s, e) {
  return s.mul(Ce(e));
}
F(
  (s, e, t, r) => t,
  (s, e, t, r, n) => {
  },
  "__left_index__"
);
F(
  (s, e, t, r) => r,
  (s, e, t, r, n) => {
  },
  "__right_index__"
);
F(
  (s, e, t, r) => s[t] + e[r],
  (s, e, t, r, n) => {
    t.backward(n), r.backward(n);
  },
  "add"
);
F(
  (s, e, t, r) => s[t] - e[r],
  (s, e, t, r, n) => {
    t.backward(n), r.backward(n.mul(new h(-1)));
  },
  "sub"
);
F(
  (s, e, t, r) => s[t] * e[r],
  (s, e, t, r, n) => {
    t.backward(n.mul(e)), r.backward(n.mul(s));
  },
  "mul"
);
F(
  (s, e, t, r) => s[t] / e[r],
  (s, e, t, r, n) => {
    t.backward(n.div(e)), r.backward(n.mul(s).mul(new h(-1)).div(e).div(e));
  },
  "div"
);
F(
  (s, e, t, r) => Math.pow(s[t], e[r]),
  (s, e, t, r, n) => {
    t.backward(n.mul(e).mul(s.pow(e.sub(new h(1))))), r.backward(n.mul(s.pow(e)).mul(s.log()));
  },
  "pow"
);
F(
  (s, e, t, r) => s[t] % e[r],
  (s, e, t, r, n) => {
    t.backward(n);
  },
  "fmod"
);
F(
  (s, e, t, r) => Math.max(s[t], e[r]),
  (s, e, t, r, n) => {
    t.backward(n.mul(s.ge(e))), r.backward(n.mul(e.gt(s)));
  },
  "maximum"
);
F(
  (s, e, t, r) => Math.min(s[t], e[r]),
  (s, e, t, r, n) => {
    t.backward(n.mul(s.le(e))), r.backward(n.mul(e.lt(s)));
  },
  "minimum"
);
function Ze(s, e, t = null) {
  const r = new Array(s.dataLength());
  for (let n = 0; n < r.length; n++)
    r[n] = Math.pow(s.data[n], e);
  return new h(
    r,
    { requires_grad: s.requires_grad },
    { operation: t, shape: s.shape }
  );
}
class et extends L {
  n;
  _forward(e, t) {
    return e.requires_grad && (this.saved_tensors = [e], this.n = t), this.next_functions.push(e.grad_fn ? e.grad_fn : w), Ze(e, t, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, r = this.n, [n] = this.next_functions;
    n.backward(e.mul(r).mul(t.pow(r - 1)));
  }
}
I("powint", et);
S(
  (s, e) => Math.log(s[e]),
  (s, e, t) => {
    e.backward(t.mul(new h(1).div(s)));
  },
  "log"
);
S(
  (s, e) => Math.sqrt(s[e]),
  (s, e, t) => {
    e.backward(t.mul(new h(1).div(s.sqrt()).div(2)));
  },
  "sqrt"
);
S(
  (s, e) => Math.exp(s[e]),
  (s, e, t) => {
    e.backward(t.mul(t.mul(s.exp())));
  },
  "exp"
);
S(
  (s, e) => s[e] * s[e],
  (s, e, t) => {
    e.backward(t.mul(t.mul(s).mul(new h(2))));
  },
  "square"
);
S(
  (s, e) => Math.abs(s[e]),
  (s, e, t) => {
    e.backward(t.mul(t.mul(Je(s))));
  },
  "abs"
);
S(
  (s, e) => Math.sign(s[e]),
  (s, e, t) => {
    e.backward(0);
  },
  "sign"
);
S(
  (s, e) => -s[e],
  (s, e, t) => {
    e.backward(t.mul(t.mul(new h(-1))));
  },
  "neg"
);
S(
  (s, e) => 1 / s[e],
  (s, e, t) => {
    e.backward(t.mul(t.mul(s.pow(-2))).neg());
  },
  "reciprocal"
);
class tt extends L {
  _forward(e, t) {
    const r = e.dataLength(), n = t.reduce((i, a) => i * a, 1);
    if (r !== n)
      throw new Error("Shape mismatch: " + e.shape + " and " + t);
    return e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(w), new h(
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
I("reshape", tt);
class st extends L {
  _forward(e, t) {
    e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(w);
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
I("squeeze", st);
class rt extends L {
  _forward(e, t) {
    e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(w), t < 0 && (t += e.shape.length + 1);
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
I("unsqueeze", rt);
class nt extends L {
  _forward(e, t) {
    e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(w);
    const r = t.length - e.shape.length, n = t.map((a, o) => {
      if (a === -1) {
        const u = o - r;
        return u >= 0 ? e.shape[u] : 1;
      }
      return a;
    }), i = Ye(e, n).data;
    return new h(
      i,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: n }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [r] = this.next_functions;
    r.backward(ie(e, t.shape));
  }
}
I("expand", nt);
S(
  (s, e) => Math.sin(s[e]),
  (s, e, t) => {
    e.backward(t.mul(t.mul(s.cos())));
  },
  "sin"
);
S(
  (s, e) => Math.cos(s[e]),
  (s, e, t) => {
    e.backward(t.mul(t.mul(s.sin().neg())));
  },
  "cos"
);
S(
  (s, e) => Math.tan(s[e]),
  (s, e, t) => {
    e.backward(t.mul(t.mul(s.cos().pow(-2))));
  },
  "tan"
);
const ls = ge(
  0,
  (s, e) => s + e,
  (s, e) => e,
  "sum"
), _s = ge(
  0,
  (s, e) => s + e,
  (s, e, t) => {
    const r = we(s.shape, t, !1), n = r.length > 0 ? r.reduce((a, o) => a * o, 1) : 1, i = s.dataLength() / n;
    return e.mul(new h([1 / i]));
  },
  "mean",
  (s, e) => s / e
), fs = ge(
  -1 / 0,
  (s, e) => Math.max(s, e),
  (s, e, t) => {
    const n = s.max(t, !0).expand(s.shape), i = s.eq(n).detach();
    return e.mul(i);
  },
  "max"
), ps = ge(
  1 / 0,
  (s, e) => Math.min(s, e),
  (s, e, t) => {
    const n = s.min(t, !0).expand(s.shape), i = s.eq(n).detach();
    return e.mul(i);
  },
  "min"
);
function at(s, e, t, r = null) {
  if (s.shape.length + e < 0 || s.shape.length + t < 0)
    throw new Error(`Transpose: Dimension out of range (${e} and ${t})`);
  e = e < 0 ? s.shape.length + e : e, t = t < 0 ? s.shape.length + t : t;
  const n = [...s.shape];
  [n[e], n[t]] = [n[t], n[e]];
  const i = s.dataLength(), a = new Array(i), o = new Array(s.shape.length), u = new Array(n.length);
  for (let c = s.shape.length - 1, d = 1; c >= 0; c--)
    o[c] = d, d *= s.shape[c];
  for (let c = n.length - 1, d = 1; c >= 0; c--)
    u[c] = d, d *= n[c];
  for (let c = 0; c < i; c++) {
    let d = c, l = 0;
    for (let _ = 0; _ < n.length; _++) {
      const f = u[_], E = Math.floor(d / f);
      d %= f;
      let x = _;
      _ === e ? x = t : _ === t && (x = e), l += E * o[x];
    }
    a[c] = s.data[l];
  }
  return new h(
    a,
    { requires_grad: s.requires_grad },
    { operation: r, shape: n }
  );
}
class it extends L {
  dim0;
  dim1;
  _forward(e, t, r) {
    return e.requires_grad && (this.saved_tensors = [e], this.dim0 = t, this.dim1 = r), this.next_functions.push(e.grad_fn ? e.grad_fn : w), at(e, t, r, this);
  }
  _backward(e) {
    const [t] = this.saved_tensors, r = this.dim0, n = this.dim1, [i] = this.next_functions;
    i.backward(e.transpose(r, n));
  }
}
I("transpose", it);
function ot(s, e, t = null) {
  if (s.shape.length == 1 && e.shape.length == 1)
    return [s.mul(e).sum(), []];
  const r = s.shape.length == 1, n = e.shape.length == 1, i = r ? [1, s.shape[0]] : s.shape, a = n ? [e.shape[0], 1] : e.shape;
  if (i[i.length - 1] != a[a.length - 2])
    throw new Error("Shape mismatch: " + s.shape + " and " + e.shape);
  const o = Me(i.slice(0, -2), a.slice(0, -2)).concat([
    i[i.length - 2],
    a[a.length - 1]
  ]), u = o.reduce((y, z) => y * z, 1), c = new Array(u).fill(0), d = oe(i, o), l = oe(a, o), _ = o[o.length - 2], f = o[o.length - 1], E = i[i.length - 1];
  for (let y = 0; y < u; y++) {
    const z = y % (_ * f), p = Math.floor(z / f), C = z % f, R = ue(d, o, y - C), T = ue(l, o, y - p * f);
    let M = 0;
    for (let D = 0; D < E; D++)
      M += s.data[R + D] * e.data[T + D * f];
    c[y] = M;
  }
  let x = [...o];
  return r && (x = x.slice(0, -2).concat([o[o.length - 1]])), n && (x = x.slice(0, -1)), [new h(
    c,
    { requires_grad: s.requires_grad || e.requires_grad },
    { operation: t, shape: x }
  ), x];
}
class ut extends Le {
  shape;
  _forward(e, t) {
    (e.requires_grad || t.requires_grad) && (this.saved_tensors = [e, t]), this.next_functions.push(e.grad_fn ? e.grad_fn : w), this.next_functions.push(t.grad_fn ? t.grad_fn : w);
    const r = ot(e, t, e.requires_grad || t.requires_grad ? this : null);
    return this.shape = r[1], r[0];
  }
  _backward(e) {
    const [t, r] = this.saved_tensors, [n, i] = this.next_functions;
    if (t.shape.length === 1 && r.shape.length === 1) {
      n.backward(e.mul(r)), i.backward(e.mul(t));
      return;
    }
    if (t.shape.length === 1) {
      const u = e.unsqueeze(-2), c = t.unsqueeze(-2);
      let d = u.matmul(r.transpose(-2, -1)), l = c.transpose(-2, -1).matmul(u);
      d = d.squeeze(-2), l = ie(l, r.shape), n.backward(d), i.backward(l);
      return;
    }
    if (r.shape.length === 1) {
      const u = e.unsqueeze(-1), c = r.unsqueeze(-1);
      let d = u.matmul(c.transpose(-2, -1)), l = t.transpose(-2, -1).matmul(u);
      d = ie(d, t.shape), l = l.squeeze(-1), n.backward(d), i.backward(l);
      return;
    }
    let a = e.matmul(r.transpose(-2, -1)), o = t.transpose(-2, -1).matmul(e);
    a = ie(a, t.shape), o = ie(o, r.shape), n.backward(a), i.backward(o);
  }
}
I("matmul", ut);
function xe(s, e, t, r, n, i, a, o) {
  const u = typeof r == "number" ? new Array(o).fill(r) : r, c = typeof n == "number" ? new Array(o).fill(n) : n, d = typeof i == "number" ? new Array(o).fill(i) : i, l = s.shape[0], _ = s.shape[1], f = e.shape[0], E = s.shape.slice(2), x = e.shape.slice(2);
  if (_ !== e.shape[1] * a)
    throw new Error(`in_channels (${_}) must be divisible by groups (${a}) and match weight.shape[1] * groups (${e.shape[1] * a})`);
  const y = E.map((B, v) => Math.floor((B + 2 * c[v] - d[v] * (x[v] - 1) - 1) / u[v] + 1)), z = [l, f, ...y], p = z.reduce((B, v) => B * v, 1), C = new Array(p).fill(0), R = (B) => {
    const v = new Array(B.length);
    let $ = 1;
    for (let W = B.length - 1; W >= 0; W--)
      v[W] = $, $ *= B[W];
    return v;
  }, T = R(s.shape), M = R(e.shape), D = R(z), re = _ / a, ne = f / a;
  for (let B = 0; B < l; B++)
    for (let v = 0; v < a; v++)
      for (let $ = 0; $ < ne; $++) {
        const W = v * ne + $, ce = y.reduce((A, O) => A * O, 1);
        for (let A = 0; A < ce; A++) {
          const O = new Array(o);
          let G = A;
          for (let m = o - 1; m >= 0; m--)
            O[m] = G % y[m], G = Math.floor(G / y[m]);
          let j = t ? t.data[W] : 0;
          for (let m = 0; m < re; m++) {
            const ee = v * re + m, ae = x.reduce((J, Q) => J * Q, 1);
            for (let J = 0; J < ae; J++) {
              const Q = new Array(o);
              let b = J;
              for (let g = o - 1; g >= 0; g--)
                Q[g] = b % x[g], b = Math.floor(b / x[g]);
              let he = !0;
              const le = new Array(o);
              for (let g = 0; g < o; g++) {
                const P = O[g] * u[g] + Q[g] * d[g] - c[g];
                if (P < 0 || P >= E[g]) {
                  he = !1;
                  break;
                }
                le[g] = P;
              }
              if (he) {
                let g = B * T[0] + ee * T[1];
                for (let U = 0; U < o; U++) g += le[U] * T[U + 2];
                let P = W * M[0] + m * M[1];
                for (let U = 0; U < o; U++) P += Q[U] * M[U + 2];
                j += s.data[g] * e.data[P];
              }
            }
          }
          let de = B * D[0] + W * D[1];
          for (let m = 0; m < o; m++) de += O[m] * D[m + 2];
          C[de] = j;
        }
      }
  return new h(C, { requires_grad: !1 }, { shape: z });
}
function be(s, e, t, r, n, i, a, o, u, c, d) {
  const l = typeof n == "number" ? new Array(u).fill(n) : n, _ = typeof i == "number" ? new Array(u).fill(i) : i, f = typeof a == "number" ? new Array(u).fill(a) : a, E = e.shape[0], x = e.shape[1], y = t.shape[0], z = e.shape.slice(2), p = t.shape.slice(2), C = s.shape.slice(2), R = (A) => {
    const O = new Array(A.length);
    let G = 1;
    for (let j = A.length - 1; j >= 0; j--)
      O[j] = G, G *= A[j];
    return O;
  }, T = R(e.shape), M = R(t.shape), D = R(s.shape);
  let re = null, ne = null, B = null, v = null, $ = null;
  c && (v = new Array(e.dataLength()).fill(0)), d && ($ = new Array(t.dataLength()).fill(0));
  const W = x / o, ce = y / o;
  for (let A = 0; A < E; A++)
    for (let O = 0; O < o; O++)
      for (let G = 0; G < ce; G++) {
        const j = O * ce + G, de = C.reduce((m, ee) => m * ee, 1);
        for (let m = 0; m < de; m++) {
          const ee = new Array(u);
          let ae = m;
          for (let b = u - 1; b >= 0; b--)
            ee[b] = ae % C[b], ae = Math.floor(ae / C[b]);
          let J = A * D[0] + j * D[1];
          for (let b = 0; b < u; b++) J += ee[b] * D[b + 2];
          const Q = s.data[J];
          for (let b = 0; b < W; b++) {
            const he = O * W + b, le = p.reduce((g, P) => g * P, 1);
            for (let g = 0; g < le; g++) {
              const P = new Array(u);
              let U = g;
              for (let q = u - 1; q >= 0; q--)
                P[q] = U % p[q], U = Math.floor(U / p[q]);
              let Oe = !0;
              const ke = new Array(u);
              for (let q = 0; q < u; q++) {
                const te = ee[q] * l[q] + P[q] * f[q] - _[q];
                if (te < 0 || te >= z[q]) {
                  Oe = !1;
                  break;
                }
                ke[q] = te;
              }
              if (Oe) {
                let q = A * T[0] + he * T[1];
                for (let K = 0; K < u; K++) q += ke[K] * T[K + 2];
                let te = j * M[0] + b * M[1];
                for (let K = 0; K < u; K++) te += P[K] * M[K + 2];
                c && (v[q] += Q * t.data[te]), d && ($[te] += Q * e.data[q]);
              }
            }
          }
        }
      }
  if (c && (re = new h(v, { requires_grad: !1 }, { shape: e.shape })), d && (ne = new h($, { requires_grad: !1 }, { shape: t.shape })), r && r.requires_grad) {
    const A = [0];
    for (let O = 2; O < s.shape.length; O++) A.push(O);
    B = s.sum(A);
  }
  return [re, ne, B];
}
class ct extends L {
  stride;
  padding;
  dilation;
  groups;
  _forward(e, t, r, n = 1, i = 0, a = 1, o = 1) {
    (e.requires_grad || t.requires_grad || r?.requires_grad) && (this.saved_tensors = [e, t], r && this.saved_tensors.push(r)), this.next_functions.push(e.grad_fn ? e.grad_fn : w), this.next_functions.push(t.grad_fn ? t.grad_fn : w), r && this.next_functions.push(r.grad_fn ? r.grad_fn : w), this.stride = n, this.padding = i, this.dilation = a, this.groups = o;
    const u = xe(e, t, r, n, i, a, o, 1);
    return u.requires_grad = e.requires_grad || t.requires_grad || (r?.requires_grad ?? !1), u.grad_fn = u.requires_grad ? this : null, u;
  }
  _backward(e) {
    const t = this.saved_tensors[0], r = this.saved_tensors[1], n = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [i, a, o] = this.next_functions, [u, c, d] = be(
      e,
      t,
      r,
      n,
      this.stride,
      this.padding,
      this.dilation,
      this.groups,
      1,
      t.requires_grad,
      r.requires_grad
    );
    t.requires_grad && i.backward(u), r.requires_grad && a.backward(c), n && n.requires_grad && o.backward(d);
  }
}
I("conv1d", ct);
class dt extends L {
  stride;
  padding;
  dilation;
  groups;
  _forward(e, t, r, n = 1, i = 0, a = 1, o = 1) {
    (e.requires_grad || t.requires_grad || r?.requires_grad) && (this.saved_tensors = [e, t], r && this.saved_tensors.push(r)), this.next_functions.push(e.grad_fn ? e.grad_fn : w), this.next_functions.push(t.grad_fn ? t.grad_fn : w), r && this.next_functions.push(r.grad_fn ? r.grad_fn : w), this.stride = n, this.padding = i, this.dilation = a, this.groups = o;
    const u = xe(e, t, r, n, i, a, o, 2);
    return u.requires_grad = e.requires_grad || t.requires_grad || (r?.requires_grad ?? !1), u.grad_fn = u.requires_grad ? this : null, u;
  }
  _backward(e) {
    const t = this.saved_tensors[0], r = this.saved_tensors[1], n = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [i, a, o] = this.next_functions, [u, c, d] = be(
      e,
      t,
      r,
      n,
      this.stride,
      this.padding,
      this.dilation,
      this.groups,
      2,
      t.requires_grad,
      r.requires_grad
    );
    t.requires_grad && i.backward(u), r.requires_grad && a.backward(c), n && n.requires_grad && o.backward(d);
  }
}
I("conv2d", dt);
class ht extends L {
  stride;
  padding;
  dilation;
  groups;
  _forward(e, t, r, n = 1, i = 0, a = 1, o = 1) {
    (e.requires_grad || t.requires_grad || r?.requires_grad) && (this.saved_tensors = [e, t], r && this.saved_tensors.push(r)), this.next_functions.push(e.grad_fn ? e.grad_fn : w), this.next_functions.push(t.grad_fn ? t.grad_fn : w), r && this.next_functions.push(r.grad_fn ? r.grad_fn : w), this.stride = n, this.padding = i, this.dilation = a, this.groups = o;
    const u = xe(e, t, r, n, i, a, o, 3);
    return u.requires_grad = e.requires_grad || t.requires_grad || (r?.requires_grad ?? !1), u.grad_fn = u.requires_grad ? this : null, u;
  }
  _backward(e) {
    const t = this.saved_tensors[0], r = this.saved_tensors[1], n = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [i, a, o] = this.next_functions, [u, c, d] = be(
      e,
      t,
      r,
      n,
      this.stride,
      this.padding,
      this.dilation,
      this.groups,
      3,
      t.requires_grad,
      r.requires_grad
    );
    t.requires_grad && i.backward(u), r.requires_grad && a.backward(c), n && n.requires_grad && o.backward(d);
  }
}
I("conv3d", ht);
F(
  (s, e, t, r) => s[t] < e[r] ? 1 : 0,
  (s, e, t, r) => {
  },
  "lt"
);
F(
  (s, e, t, r) => s[t] > e[r] ? 1 : 0,
  (s, e, t, r) => {
  },
  "gt"
);
F(
  (s, e, t, r) => s[t] <= e[r] ? 1 : 0,
  (s, e, t, r) => {
  },
  "le"
);
F(
  (s, e, t, r) => s[t] >= e[r] ? 1 : 0,
  (s, e, t, r) => {
  },
  "ge"
);
F(
  (s, e, t, r) => s[t] == e[r] ? 1 : 0,
  (s, e, t, r) => {
  },
  "eq"
);
F(
  (s, e, t, r) => s[t] != e[r] ? 1 : 0,
  (s, e, t, r) => {
  },
  "ne"
);
S(
  (s, e) => Math.max(s[e], 0),
  (s, e, t) => {
    e.backward(t.mul(t.mul(s.gt(0))));
  },
  "relu"
);
S(
  (s, e) => 1 / (1 + Math.exp(-s[e])),
  (s, e, t) => {
    const r = s.sigmoid();
    e.backward(r.mul(r.mul(-1).add(1)).mul(t));
  },
  "sigmoid"
);
class Z extends h {
  constructor(e, t = {
    requires_grad: !0
  }, r = {}) {
    e instanceof h ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : e instanceof Z ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : super(e, t, r);
  }
}
class se {
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
    t instanceof Z ? this.register_parameter(e, t) : this.register_module(e, t);
  }
  parameters() {
    let e = Object.values(this._parameters);
    for (const t of Object.values(this._modules))
      e = e.concat(t.parameters());
    return e;
  }
}
class lt extends se {
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
class qe {
}
class _t extends qe {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).pow(2).mean();
  }
}
class ft extends qe {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).abs().mean();
  }
}
class pt extends qe {
  weight;
  constructor(e = null) {
    super(), this.weight = e;
  }
  forward(e, t) {
    const r = t.mul(e.log()), n = t.neg().add(1).mul(e.neg().add(1).log()), i = r.add(n).neg().mean();
    return this.weight ? i.mul(this.weight) : i;
  }
}
function ve(s) {
  return (...e) => new (V(s))().forward(...e);
}
function ze(s) {
  return (e) => (typeof e == "number" && (e = new h(e)), new (V(s))().forward(e));
}
const De = ze("relu"), Pe = ze("sigmoid"), We = ve("conv1d"), Ne = ve("conv2d"), $e = ve("conv3d"), gt = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  conv1d: We,
  conv2d: Ne,
  conv3d: $e,
  relu: De,
  sigmoid: Pe
}, Symbol.toStringTag, { value: "Module" }));
class mt extends se {
  weight;
  bias;
  constructor(e, t) {
    super();
    const r = Math.sqrt(1 / e);
    this.weight = new Z(
      fe([t, e]).mul(2 * r).sub(r)
    ), this.bias = new Z(
      fe([t]).mul(2 * r).sub(r)
    ), this.register("weight", this.weight), this.register("bias", this.bias);
  }
  forward(e) {
    return e.matmul(this.weight.transpose(0, 1)).add(this.bias);
  }
}
class wt extends se {
  constructor() {
    super();
  }
  forward(e) {
    return De(e);
  }
}
class xt extends se {
  constructor() {
    super();
  }
  forward(e) {
    return Pe(e);
  }
}
class ye extends se {
  weight;
  bias;
  in_channels;
  out_channels;
  kernel_size;
  stride;
  padding;
  dilation;
  groups;
  constructor(e, t, r, n, i, a, o, u, c) {
    if (super(), this.in_channels = e, this.out_channels = t, this.kernel_size = r, this.stride = n, this.padding = i, this.dilation = a, this.groups = o, e % o !== 0)
      throw new Error("in_channels must be divisible by groups");
    if (t % o !== 0)
      throw new Error("out_channels must be divisible by groups");
    const d = typeof r == "number" ? new Array(c).fill(r) : r, l = d.reduce((f, E) => f * E, 1), _ = Math.sqrt(o / (e * l));
    this.weight = new Z(
      fe([t, e / o, ...d]).mul(2 * _).sub(_)
    ), this.register("weight", this.weight), u ? (this.bias = new Z(
      fe([t]).mul(2 * _).sub(_)
    ), this.register("bias", this.bias)) : this.bias = null;
  }
}
class bt extends ye {
  constructor(e, t, r, n = 1, i = 0, a = 1, o = 1, u = !0) {
    super(e, t, r, n, i, a, o, u, 1);
  }
  forward(e) {
    return We(e, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}
class qt extends ye {
  constructor(e, t, r, n = 1, i = 0, a = 1, o = 1, u = !0) {
    super(e, t, r, n, i, a, o, u, 2);
  }
  forward(e) {
    return Ne(e, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}
class vt extends ye {
  constructor(e, t, r, n = 1, i = 0, a = 1, o = 1, u = !0) {
    super(e, t, r, n, i, a, o, u, 3);
  }
  forward(e) {
    return $e(e, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}
const gs = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  BCELoss: pt,
  Conv1d: bt,
  Conv2d: qt,
  Conv3d: vt,
  L1Loss: ft,
  Linear: mt,
  MSELoss: _t,
  Module: se,
  Parameter: Z,
  ReLU: wt,
  Sequential: lt,
  Sigmoid: xt,
  functional: gt
}, Symbol.toStringTag, { value: "Module" }));
class Ae {
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
class yt extends Ae {
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
        const n = this.state.get(e).velocity;
        this.nesterov ? t = t.add(n.mul(this.momentum)) : t = n, this.state.set(e, { velocity: n });
      }
      const r = e.sub(t.mul(this.lr));
      e.data = r.data;
    }
  }
}
class At extends Ae {
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
        m: _e(e),
        v: _e(e),
        vmax: _e(e)
      });
      const r = this.state.get(e);
      r.m = r.m.mul(this.beta1).add(t.mul(1 - this.beta1)), r.v = r.v.mul(this.beta2).add(t.mul(t).mul(1 - this.beta2));
      const n = 1 - Math.pow(this.beta1, this.step_count), i = 1 - Math.pow(this.beta2, this.step_count);
      let a;
      const o = r.m.div(n);
      this.amsgrad ? (r.vmax = r.vmax.maximum(r.v), a = r.vmax.div(i)) : a = r.v.div(i);
      const u = o.div(a.sqrt().add(this.eps)).mul(this.lr), c = e.sub(u);
      e.data = c.data;
    }
  }
}
const ms = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Adam: At,
  Optimizer: Ae,
  SGD: yt
}, Symbol.toStringTag, { value: "Module" }));
export {
  Ee as AccumulateGrad,
  fs as Max,
  _s as Mean,
  ps as Min,
  ls as Sum,
  h as Tensor,
  L as TorchFunction,
  Mt as __left_index__,
  Bt as __right_index__,
  Gt as abs,
  Ct as add,
  Rt as arange,
  Yt as cos,
  Lt as div,
  ds as eq,
  X as eventBus,
  Y as events,
  Nt as exp,
  Qt as expand,
  St as fmod,
  cs as ge,
  os as gt,
  us as le,
  Ft as linspace,
  Pt as log,
  is as lt,
  as as matmul,
  rs as max,
  zt as maximum,
  ts as mean,
  ss as min,
  Dt as minimum,
  Ut as mul,
  hs as ne,
  jt as neg,
  gs as nn,
  Ce as ones,
  Et as ones_like,
  ms as optim,
  It as pow,
  fe as rand,
  kt as randint,
  Ot as randn,
  Kt as reciprocal,
  Vt as reshape,
  Je as sign,
  Xt as sin,
  Wt as sqrt,
  $t as square,
  Ht as squeeze,
  Tt as sub,
  es as sum,
  Zt as tan,
  ns as transpose,
  Jt as unsqueeze,
  Ge as zeros,
  _e as zeros_like
};
