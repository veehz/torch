function Ct(s, t) {
  const e = Math.max(s.length, t.length), n = [...Array(e - s.length).fill(1), ...s], r = [...Array(e - t.length).fill(1), ...t], a = [];
  for (let o = 0; o < e; o++) {
    if (n[o] !== r[o] && n[o] !== 1 && r[o] !== 1)
      throw new Error(`Shape mismatch: ${s} and ${t}`);
    a.push(Math.max(n[o], r[o]));
  }
  return a;
}
function Nt(s, t, e) {
  const n = ut(t, s), r = new Array(t.reduce((a, o) => a * o, 1)).fill(0);
  for (let a = 0; a < e.length; a++)
    r[ct(n, s, a)] += e[a];
  return r;
}
function ut(s, t) {
  return s.length >= t.length ? s : [...Array(t.length - s.length).fill(1), ...s];
}
function ct(s, t, e) {
  let n = 0, r = 1, a = e;
  for (let o = s.length - 1; o >= 0; o--) {
    if (s[o] > 1) {
      const i = a % t[o];
      n = n + i * r;
    }
    r *= s[o], a = Math.floor(a / t[o]);
  }
  return n;
}
function mt(s) {
  return Array.isArray(s[0]) ? s[0] : s;
}
function $e(...s) {
  const t = mt(s), e = new l(Array(t.reduce((n, r) => n * r, 1)).fill(Math.random()));
  return e.shape = t, e;
}
function pt(...s) {
  const t = mt(s), e = new l(Array(t.reduce((n, r) => n * r, 1)).fill(Math.random()));
  return e.shape = t, e;
}
function ze(s, t, e) {
  const n = new l(
    Array(e.reduce((r, a) => r * a, 1)).fill(Math.floor(Math.random() * (t - s) + s))
  );
  return n.shape = e, n;
}
function Ut(...s) {
  const t = mt(s), e = new l(Array(t.reduce((n, r) => n * r, 1)).fill(1));
  return e.shape = t, e;
}
function Jt(...s) {
  const t = mt(s), e = new l(Array(t.reduce((n, r) => n * r, 1)).fill(0));
  return e.shape = t, e;
}
function Se(s) {
  return Ut(s.shape);
}
function ft(s) {
  return Jt(s.shape);
}
function Pe(s, t, e) {
  const n = [], r = (t - s) / (e - 1);
  for (let a = 0; a < e - 1; a++)
    n.push(s + a * r);
  return n.push(t), new l(n);
}
function Ie(s, t = void 0, e = 1) {
  const n = [];
  for (let r = s; r < t; r += e)
    n.push(r);
  return new l(n);
}
let gt = !0;
function Ht() {
  return gt;
}
function Qt() {
  const s = gt;
  return gt = !1, s;
}
function Xt(s) {
  gt = s;
}
function Yt(s) {
  const t = Qt();
  try {
    return s();
  } finally {
    Xt(t);
  }
}
let Zt = 0;
const $t = () => Zt++, V = new EventTarget(), J = {
  TENSOR_BEFORE_BACKWARD: "tensor.beforeBackward",
  TENSOR_AFTER_BACKWARD: "tensor.afterBackward",
  OPERATION_BEFORE_FORWARD: "operation.beforeForward",
  OPERATION_AFTER_FORWARD: "operation.afterForward",
  OPERATION_BEFORE_BACKWARD: "operation.beforeBackward",
  OPERATION_AFTER_BACKWARD: "operation.afterBackward",
  OPERATION_BEFORE_ACCUMULATE_GRAD: "operation.beforeAccumulateGrad",
  OPERATION_AFTER_ACCUMULATE_GRAD: "operation.afterAccumulateGrad"
};
function y(...s) {
  if (!Ht()) return !1;
  for (const t of s)
    if (t instanceof l && t.requires_grad)
      return !0;
  return !1;
}
class S {
  id = $t();
  opName = "";
  next_functions = [];
  saved_tensors = [];
  _retained_tensors = [];
  forward(...t) {
    const e = y(...t);
    V.dispatchEvent(new CustomEvent(J.OPERATION_BEFORE_FORWARD, {
      detail: {
        operation: this,
        requires_grad: e,
        args: t
      }
    }));
    const n = this._forward(...t);
    return V.dispatchEvent(new CustomEvent(J.OPERATION_AFTER_FORWARD, {
      detail: {
        operation: this,
        requires_grad: e,
        args: t,
        result: n
      }
    })), n;
  }
  backward(t) {
    V.dispatchEvent(new CustomEvent(J.OPERATION_BEFORE_BACKWARD, { detail: { operation: this, dz: t } }));
    for (const e of this._retained_tensors)
      e.grad || (e.grad = new l(new Array(e.dataLength()).fill(0))), e.grad = e.grad.add(t);
    this._backward(t), V.dispatchEvent(new CustomEvent(J.OPERATION_AFTER_BACKWARD, { detail: { operation: this, dz: t } }));
  }
}
class te extends S {
  _forward(...t) {
    throw new Error("NullOp should not be called");
  }
  _backward(t) {
  }
}
const v = new te();
class zt extends S {
}
class St extends S {
}
class Rt extends zt {
  variable;
  _forward(t) {
    return this.variable = t, t;
  }
  _backward(t) {
    if (this.variable.grad || (this.variable.grad = ft(this.variable)), V.dispatchEvent(new CustomEvent(J.OPERATION_BEFORE_ACCUMULATE_GRAD, { detail: { operation: this, dz: t } })), typeof t == "number")
      this.variable.grad = this.variable.grad.add(t);
    else {
      const e = Nt(t.shape, this.variable.shape, t.data);
      this.variable.grad = this.variable.grad.add(new l(e, {}, { shape: this.variable.shape }));
    }
    V.dispatchEvent(new CustomEvent(J.OPERATION_AFTER_ACCUMULATE_GRAD, { detail: { operation: this, dz: t } }));
  }
}
const Pt = /* @__PURE__ */ new Map(), Mt = /* @__PURE__ */ new Map();
function P(s, t) {
  Pt.set(s, t);
}
function It(s) {
  const t = Pt.get(s);
  if (!t)
    throw new Error(`Operation '${s}' is not registered.`);
  return t;
}
function Tt(s) {
  const t = Mt.get(s);
  if (!t) {
    const e = new (It(s))();
    return e.opName = s, Mt.set(s, e), e;
  }
  return t;
}
function Z(s) {
  const t = new (It(s))();
  return t.opName = s, t;
}
function ee(s) {
  if (ArrayBuffer.isView(s))
    return [s.length];
  const t = [];
  for (; Array.isArray(s); )
    t.push(s.length), s = s[0];
  return t;
}
function Lt(s) {
  return Array.isArray(s) ? s.flatMap((t) => Lt(t)) : ArrayBuffer.isView(s) ? Array.from(s) : [s];
}
class l {
  // Auto-generated ID
  id = $t();
  // Optional user-defined name
  name = null;
  data;
  shape;
  grad_fn = null;
  grad = null;
  requires_grad;
  constructor(t, e = {}, n = {}) {
    if (this.data = Lt(t), this.requires_grad = e.requires_grad ?? !1, e.name && (this.name = e.name), this.shape = n.shape ?? ee(t), this.grad_fn = n.operation ?? null, this.requires_grad && !this.grad_fn) {
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
      const a = this.shape[r], o = new Array(a), i = r === this.shape.length - 1;
      for (let u = 0; u < a; u++)
        i ? o[u] = e[t++] : o[u] = n(r + 1);
      return o;
    };
    return n(0);
  }
  dataLength() {
    return this.data.length;
  }
  _executeUnaryOp(t) {
    return (y(this) ? Z(t) : Tt(t)).forward(this);
  }
  _executeBinaryOp(t, e) {
    return typeof e == "number" && (e = new l(e)), (y(this, e) ? Z(t) : Tt(t)).forward(this, e);
  }
  _executeOpRaw(t, ...e) {
    return Z(t).forward(this, ...e);
  }
  item() {
    if (this.dataLength() !== 1)
      throw new Error("Tensor.item() is only valid for scalars");
    return this.data[0];
  }
  detach() {
    return new l(this.data, { requires_grad: !1 }, { shape: this.shape });
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
        t = new l(1);
      }
      this.grad_fn && (V.dispatchEvent(new CustomEvent(J.TENSOR_BEFORE_BACKWARD, { detail: { tensor: this } })), this.grad_fn.backward(t), V.dispatchEvent(new CustomEvent(J.TENSOR_AFTER_BACKWARD, { detail: { tensor: this } })));
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
function Q(s) {
  return (...t) => Z(s).forward(...t);
}
function D(s) {
  return (t) => (typeof t == "number" && (t = new l(t)), Z(s).forward(t));
}
function F(s) {
  return (t, e) => (typeof t == "number" && (t = new l(t)), typeof e == "number" && (e = new l(e)), Z(s).forward(t, e));
}
const Le = F("__left_index__"), De = F("__right_index__"), We = F("add"), je = F("sub"), Ge = F("mul"), Ke = F("div"), Ve = F("pow"), Je = F("fmod"), He = F("maximum"), Qe = F("minimum"), Xe = D("log"), Ye = D("sqrt"), Ze = D("exp"), ts = D("square"), es = D("abs"), se = D("sign"), ss = D("neg"), ns = D("reciprocal"), rs = D("nan_to_num"), as = Q("reshape"), os = Q("squeeze"), is = Q("unsqueeze"), us = Q("expand"), cs = D("sin"), ds = D("cos"), ls = D("tan"), hs = Q("sum"), _s = Q("mean"), fs = Q("min"), ps = Q("max"), gs = Q("transpose"), ms = F("matmul"), ws = F("lt"), xs = F("gt"), bs = F("le"), vs = F("ge"), ys = F("eq"), As = F("ne");
function Ft(s) {
  const t = new Array(s.length).fill(1);
  for (let e = s.length - 2; e >= 0; e--)
    t[e] = t[e + 1] * s[e + 1];
  return t;
}
function ne(s, t) {
  return t.map((e) => {
    const n = Math.floor(s / e);
    return s %= e, n;
  });
}
function re(s, t) {
  return s.reduce((e, n, r) => e + n * t[r], 0);
}
function xt(s, t, e = !1) {
  if (t === void 0) return e ? s.map(() => 1) : [];
  const r = (Array.isArray(t) ? t : [t]).map((a) => a < 0 ? a + s.length : a);
  return e ? s.map((a, o) => r.includes(o) ? 1 : a) : s.filter((a, o) => !r.includes(o));
}
function C(s, t, e = null) {
  const n = (o, i, u, c, d, f) => {
    const _ = Array(f);
    for (let h = 0; h < f; h++) {
      const m = ct(i, d, h), p = ct(c, d, h);
      _[h] = s(o, u, m, p);
    }
    return _;
  }, r = (o, i, u = null) => {
    const c = Ct(o.shape, i.shape), d = ut(o.shape, c), f = ut(i.shape, c), _ = c.reduce((h, m) => h * m, 1);
    return new l(
      n(
        o.data,
        d,
        i.data,
        f,
        c,
        _
      ),
      { requires_grad: y(o, i) },
      { operation: u, shape: c }
    );
  }, a = {
    [e]: class extends St {
      _forward(o, i) {
        const u = y(o, i);
        return u && (this.saved_tensors = [o, i]), this.next_functions.push(o.grad_fn ? o.grad_fn : v), this.next_functions.push(i.grad_fn ? i.grad_fn : v), r(o, i, u ? this : null);
      }
      _backward(o) {
        const [i, u] = this.saved_tensors, [c, d] = this.next_functions;
        t(i, u, c, d, o);
      }
    }
  }[e];
  return e && P(e, a), a;
}
function $(s, t, e = null) {
  const n = (o, i) => {
    const u = Array(i);
    for (let c = 0; c < i; c++)
      u[c] = s(o, c);
    return u;
  }, r = (o, i = null) => {
    const u = o.dataLength();
    return new l(
      n(o.data, u),
      { requires_grad: y(o) },
      { operation: i, shape: o.shape }
    );
  }, a = {
    [e]: class extends zt {
      _forward(o) {
        const i = y(o);
        return i && (this.saved_tensors = [o]), this.next_functions.push(o.grad_fn ? o.grad_fn : v), r(o, i ? this : null);
      }
      _backward(o) {
        const [i] = this.saved_tensors, [u] = this.next_functions;
        t(i, u, o);
      }
    }
  }[e];
  return e && P(e, a), a;
}
function wt(s, t, e, n = null, r) {
  const a = {
    [n]: class extends S {
      dim;
      keepdim;
      _forward(o, i, u = !1) {
        this.dim = i, this.keepdim = u;
        const c = y(o);
        c && (this.saved_tensors = [o]), this.next_functions.push(o.grad_fn ? o.grad_fn : v);
        const d = xt(o.shape, i, u), f = d.reduce((g, E) => g * E, 1), _ = new Array(f).fill(s), h = new Array(f).fill(0), m = Ft(o.shape), p = Ft(d), B = (i === void 0 ? [] : Array.isArray(i) ? i : [i]).map((g) => g < 0 ? g + o.shape.length : g), I = i === void 0;
        for (let g = 0; g < o.data.length; g++) {
          const E = ne(g, m);
          let k;
          if (I)
            k = u ? E.map(() => 0) : [];
          else {
            k = [];
            for (let R = 0; R < o.shape.length; R++)
              B.includes(R) ? u && k.push(0) : k.push(E[R]);
          }
          const U = re(k, p);
          _[U] = t(_[U], o.data[g]), h[U]++;
        }
        if (r)
          for (let g = 0; g < f; g++)
            _[g] = r(_[g], h[g]);
        return new l(
          _,
          { requires_grad: c },
          { operation: c ? this : null, shape: d }
        );
      }
      _backward(o) {
        const [i] = this.saved_tensors, [u] = this.next_functions;
        let c = o;
        const d = xt(i.shape, this.dim, !0);
        o.shape.length !== d.length && (c = o.reshape(d));
        const f = c.expand(i.shape), _ = e(i, f, this.dim, this.keepdim);
        u.backward(_);
      }
    }
  }[n];
  return n && P(n, a), a;
}
function it(s, t) {
  const e = Nt(s.shape, t, s.data);
  return new l(e, { requires_grad: s.requires_grad }, { shape: t });
}
function ae(s, t) {
  return s.mul(Ut(t));
}
C(
  (s, t, e, n) => e,
  () => {
  },
  "__left_index__"
);
C(
  (s, t, e, n) => n,
  () => {
  },
  "__right_index__"
);
C(
  (s, t, e, n) => s[e] + t[n],
  (s, t, e, n, r) => {
    e.backward(r), n.backward(r);
  },
  "add"
);
C(
  (s, t, e, n) => s[e] - t[n],
  (s, t, e, n, r) => {
    e.backward(r), n.backward(r.mul(new l(-1)));
  },
  "sub"
);
C(
  (s, t, e, n) => s[e] * t[n],
  (s, t, e, n, r) => {
    e.backward(r.mul(t)), n.backward(r.mul(s));
  },
  "mul"
);
C(
  (s, t, e, n) => s[e] / t[n],
  (s, t, e, n, r) => {
    e.backward(r.div(t)), n.backward(r.mul(s).mul(new l(-1)).div(t).div(t));
  },
  "div"
);
function Bt(s, t, e) {
  const n = typeof e == "number" ? e : null, r = new Array(t.dataLength());
  for (let a = 0; a < r.length; a++)
    r[a] = s.data[a] ? t.data[a] : n !== null ? n : e.data[a];
  return new l(r, {}, { shape: t.shape });
}
C(
  (s, t, e, n) => Math.pow(s[e], t[n]),
  (s, t, e, n, r) => {
    const a = r.mul(t).mul(s.pow(t.sub(new l(1)))), o = r.mul(s.pow(t)).mul(s.log());
    e.backward(Bt(s.ne(0), a, a.nan_to_num())), n.backward(Bt(s.ne(0), o, 0));
  },
  "pow"
);
C(
  (s, t, e, n) => s[e] % t[n],
  (s, t, e, n, r) => {
    e.backward(r);
  },
  "fmod"
);
C(
  (s, t, e, n) => Math.max(s[e], t[n]),
  (s, t, e, n, r) => {
    const a = s.eq(t), o = s.gt(t).add(a.mul(new l(0.5))), i = t.gt(s).add(a.mul(new l(0.5)));
    e.backward(r.mul(o)), n.backward(r.mul(i));
  },
  "maximum"
);
C(
  (s, t, e, n) => Math.min(s[e], t[n]),
  (s, t, e, n, r) => {
    const a = s.eq(t), o = s.lt(t).add(a.mul(new l(0.5))), i = t.lt(s).add(a.mul(new l(0.5)));
    e.backward(r.mul(o)), n.backward(r.mul(i));
  },
  "minimum"
);
function oe(s, t, e = null) {
  const n = new Array(s.dataLength());
  for (let r = 0; r < n.length; r++)
    n[r] = Math.pow(s.data[r], t);
  return new l(
    n,
    { requires_grad: y(s) },
    { operation: e, shape: s.shape }
  );
}
class ie extends S {
  n;
  _forward(t, e) {
    const n = y(t);
    return n && (this.saved_tensors = [t], this.n = e), this.next_functions.push(t.grad_fn ? t.grad_fn : v), oe(t, e, n ? this : null);
  }
  _backward(t) {
    const [e] = this.saved_tensors, n = this.n, [r] = this.next_functions;
    r.backward(t.mul(n).mul(e.pow(n - 1)));
  }
}
P("powint", ie);
$(
  (s, t) => Math.log(s[t]),
  (s, t, e) => {
    t.backward(e.mul(new l(1).div(s)));
  },
  "log"
);
$(
  (s, t) => Math.sqrt(s[t]),
  (s, t, e) => {
    t.backward(e.mul(new l(1).div(s.sqrt()).div(2)));
  },
  "sqrt"
);
$(
  (s, t) => Math.exp(s[t]),
  (s, t, e) => {
    t.backward(e.mul(s.exp()));
  },
  "exp"
);
$(
  (s, t) => s[t] * s[t],
  (s, t, e) => {
    t.backward(e.mul(s).mul(new l(2)));
  },
  "square"
);
$(
  (s, t) => Math.abs(s[t]),
  (s, t, e) => {
    t.backward(e.mul(se(s)));
  },
  "abs"
);
$(
  (s, t) => Math.sign(s[t]),
  (s, t) => {
    t.backward(0);
  },
  "sign"
);
$(
  (s, t) => -s[t],
  (s, t, e) => {
    t.backward(e.mul(new l(-1)));
  },
  "neg"
);
$(
  (s, t) => 1 / s[t],
  (s, t, e) => {
    t.backward(e.mul(s.pow(-2)).neg());
  },
  "reciprocal"
);
$(
  (s, t) => {
    const e = s[t];
    return Number.isNaN(e) ? 0 : e === 1 / 0 ? 34028235e31 : e === -1 / 0 ? -34028235e31 : e;
  },
  (s, t, e) => {
    t.backward(e);
  },
  "nan_to_num"
);
class ue extends S {
  _forward(t, e) {
    const n = t.dataLength(), r = e.reduce((o, i) => o * i, 1);
    if (n !== r)
      throw new Error("Shape mismatch: " + t.shape + " and " + e);
    const a = y(t);
    return a && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(v), new l(
      t.data,
      { requires_grad: a },
      { operation: a ? this : null, shape: e }
    );
  }
  _backward(t) {
    const [e] = this.saved_tensors, [n] = this.next_functions;
    n.backward(t.reshape(e.shape));
  }
}
P("reshape", ue);
class ce extends S {
  _forward(t, e) {
    const n = y(t);
    n && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(v);
    let r = [...t.shape];
    return e !== void 0 ? (e < 0 && (e += t.shape.length), r[e] === 1 && r.splice(e, 1)) : r = r.filter((a) => a !== 1), new l(
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
P("squeeze", ce);
class de extends S {
  _forward(t, e) {
    const n = y(t);
    n && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(v), e < 0 && (e += t.shape.length + 1);
    const r = [...t.shape];
    return r.splice(e, 0, 1), new l(
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
P("unsqueeze", de);
class le extends S {
  _forward(t, e) {
    const n = y(t);
    n && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(v);
    const r = e.length - t.shape.length, a = e.map((i, u) => {
      if (i === -1) {
        const c = u - r;
        return c >= 0 ? t.shape[c] : 1;
      }
      return i;
    }), o = ae(t, a).data;
    return new l(
      o,
      { requires_grad: n },
      { operation: n ? this : null, shape: a }
    );
  }
  _backward(t) {
    const [e] = this.saved_tensors, [n] = this.next_functions;
    n.backward(it(t, e.shape));
  }
}
P("expand", le);
$(
  (s, t) => Math.sin(s[t]),
  (s, t, e) => {
    t.backward(e.mul(s.cos()));
  },
  "sin"
);
$(
  (s, t) => Math.cos(s[t]),
  (s, t, e) => {
    t.backward(e.mul(s.sin().neg()));
  },
  "cos"
);
$(
  (s, t) => Math.tan(s[t]),
  (s, t, e) => {
    t.backward(e.mul(s.cos().pow(-2)));
  },
  "tan"
);
const Os = wt(
  0,
  (s, t) => s + t,
  (s, t) => t,
  "sum"
), ks = wt(
  0,
  (s, t) => s + t,
  (s, t, e) => {
    const n = xt(s.shape, e, !1), r = n.length > 0 ? n.reduce((o, i) => o * i, 1) : 1, a = s.dataLength() / r;
    return t.mul(new l([1 / a]));
  },
  "mean",
  (s, t) => s / t
), qs = wt(
  -1 / 0,
  (s, t) => Math.max(s, t),
  (s, t, e) => {
    const r = s.max(e, !0).expand(s.shape), a = s.eq(r).detach();
    return t.mul(a);
  },
  "max"
), Es = wt(
  1 / 0,
  (s, t) => Math.min(s, t),
  (s, t, e) => {
    const r = s.min(e, !0).expand(s.shape), a = s.eq(r).detach();
    return t.mul(a);
  },
  "min"
);
function he(s, t, e, n = null) {
  if (s.shape.length + t < 0 || s.shape.length + e < 0)
    throw new Error(`Transpose: Dimension out of range (${t} and ${e})`);
  t = t < 0 ? s.shape.length + t : t, e = e < 0 ? s.shape.length + e : e;
  const r = [...s.shape];
  [r[t], r[e]] = [r[e], r[t]];
  const a = s.dataLength(), o = new Array(a), i = new Array(s.shape.length), u = new Array(r.length);
  for (let c = s.shape.length - 1, d = 1; c >= 0; c--)
    i[c] = d, d *= s.shape[c];
  for (let c = r.length - 1, d = 1; c >= 0; c--)
    u[c] = d, d *= r[c];
  for (let c = 0; c < a; c++) {
    let d = c, f = 0;
    for (let _ = 0; _ < r.length; _++) {
      const h = u[_], m = Math.floor(d / h);
      d %= h;
      let p = _;
      _ === t ? p = e : _ === e && (p = t), f += m * i[p];
    }
    o[c] = s.data[f];
  }
  return new l(
    o,
    { requires_grad: y(s) },
    { operation: n, shape: r }
  );
}
class _e extends S {
  dim0;
  dim1;
  _forward(t, e, n) {
    const r = y(t);
    return r && (this.saved_tensors = [t], this.dim0 = e, this.dim1 = n), this.next_functions.push(t.grad_fn ? t.grad_fn : v), he(t, e, n, r ? this : null);
  }
  _backward(t) {
    const e = this.dim0, n = this.dim1, [r] = this.next_functions;
    r.backward(t.transpose(e, n));
  }
}
P("transpose", _e);
function fe(s, t, e = null) {
  if (s.shape.length == 1 && t.shape.length == 1)
    return [s.mul(t).sum(), []];
  const n = s.shape.length == 1, r = t.shape.length == 1, a = n ? [1, s.shape[0]] : s.shape, o = r ? [t.shape[0], 1] : t.shape;
  if (a[a.length - 1] != o[o.length - 2])
    throw new Error("Shape mismatch: " + s.shape + " and " + t.shape);
  const i = Ct(a.slice(0, -2), o.slice(0, -2)).concat([
    a[a.length - 2],
    o[o.length - 1]
  ]), u = i.reduce((w, B) => w * B, 1), c = new Array(u).fill(0), d = ut(a, i), f = ut(o, i), _ = i[i.length - 2], h = i[i.length - 1], m = a[a.length - 1];
  for (let w = 0; w < u; w++) {
    const B = w % (_ * h), I = Math.floor(B / h), g = B % h, E = ct(d, i, w - g), k = ct(f, i, w - I * h);
    let U = 0;
    for (let R = 0; R < m; R++)
      U += s.data[E + R] * t.data[k + R * h];
    c[w] = U;
  }
  let p = [...i];
  return n && (p = p.slice(0, -2).concat([i[i.length - 1]])), r && (p = p.slice(0, -1)), [new l(
    c,
    { requires_grad: y(s, t) },
    { operation: e, shape: p }
  ), p];
}
class pe extends St {
  shape;
  _forward(t, e) {
    const n = y(t, e);
    n && (this.saved_tensors = [t, e]), this.next_functions.push(t.grad_fn ? t.grad_fn : v), this.next_functions.push(e.grad_fn ? e.grad_fn : v);
    const r = fe(t, e, n ? this : null);
    return this.shape = r[1], r[0];
  }
  _backward(t) {
    const [e, n] = this.saved_tensors, [r, a] = this.next_functions;
    if (e.shape.length === 1 && n.shape.length === 1) {
      r.backward(t.mul(n)), a.backward(t.mul(e));
      return;
    }
    if (e.shape.length === 1) {
      const u = t.unsqueeze(-2), c = e.unsqueeze(-2);
      let d = u.matmul(n.transpose(-2, -1)), f = c.transpose(-2, -1).matmul(u);
      d = d.squeeze(-2), f = it(f, n.shape), r.backward(d), a.backward(f);
      return;
    }
    if (n.shape.length === 1) {
      const u = t.unsqueeze(-1), c = n.unsqueeze(-1);
      let d = u.matmul(c.transpose(-2, -1)), f = e.transpose(-2, -1).matmul(u);
      d = it(d, e.shape), f = f.squeeze(-1), r.backward(d), a.backward(f);
      return;
    }
    let o = t.matmul(n.transpose(-2, -1)), i = e.transpose(-2, -1).matmul(t);
    o = it(o, e.shape), i = it(i, n.shape), r.backward(o), a.backward(i);
  }
}
P("matmul", pe);
function bt(s, t, e, n, r, a, o, i) {
  const u = typeof n == "number" ? new Array(i).fill(n) : n, c = typeof r == "number" ? new Array(i).fill(r) : r, d = typeof a == "number" ? new Array(i).fill(a) : a, f = s.shape[0], _ = s.shape[1], h = t.shape[0], m = s.shape.slice(2), p = t.shape.slice(2);
  if (_ !== t.shape[1] * o)
    throw new Error(`in_channels (${_}) must be divisible by groups (${o}) and match weight.shape[1] * groups (${t.shape[1] * o})`);
  const w = m.map((N, q) => Math.floor((N + 2 * c[q] - d[q] * (p[q] - 1) - 1) / u[q] + 1)), B = [f, h, ...w], I = B.reduce((N, q) => N * q, 1), g = new Array(I).fill(0), E = (N) => {
    const q = new Array(N.length);
    let j = 1;
    for (let W = N.length - 1; W >= 0; W--)
      q[W] = j, j *= N[W];
    return q;
  }, k = E(s.shape), U = E(t.shape), R = E(B), rt = _ / o, at = h / o;
  for (let N = 0; N < f; N++)
    for (let q = 0; q < o; q++)
      for (let j = 0; j < at; j++) {
        const W = q * at + j, dt = w.reduce((M, T) => M * T, 1);
        for (let M = 0; M < dt; M++) {
          const T = new Array(i);
          let G = M;
          for (let b = i - 1; b >= 0; b--)
            T[b] = G % w[b], G = Math.floor(G / w[b]);
          let K = e ? e.data[W] : 0;
          for (let b = 0; b < rt; b++) {
            const et = q * rt + b, ot = p.reduce((X, Y) => X * Y, 1);
            for (let X = 0; X < ot; X++) {
              const Y = new Array(i);
              let A = X;
              for (let x = i - 1; x >= 0; x--)
                Y[x] = A % p[x], A = Math.floor(A / p[x]);
              let ht = !0;
              const _t = new Array(i);
              for (let x = 0; x < i; x++) {
                const L = T[x] * u[x] + Y[x] * d[x] - c[x];
                if (L < 0 || L >= m[x]) {
                  ht = !1;
                  break;
                }
                _t[x] = L;
              }
              if (ht) {
                let x = N * k[0] + et * k[1];
                for (let z = 0; z < i; z++) x += _t[z] * k[z + 2];
                let L = W * U[0] + b * U[1];
                for (let z = 0; z < i; z++) L += Y[z] * U[z + 2];
                K += s.data[x] * t.data[L];
              }
            }
          }
          let lt = N * R[0] + W * R[1];
          for (let b = 0; b < i; b++) lt += T[b] * R[b + 2];
          g[lt] = K;
        }
      }
  return new l(g, { requires_grad: !1 }, { shape: B });
}
function vt(s, t, e, n, r, a, o, i, u, c, d) {
  const f = typeof r == "number" ? new Array(u).fill(r) : r, _ = typeof a == "number" ? new Array(u).fill(a) : a, h = typeof o == "number" ? new Array(u).fill(o) : o, m = t.shape[0], p = t.shape[1], w = e.shape[0], B = t.shape.slice(2), I = e.shape.slice(2), g = s.shape.slice(2), E = (M) => {
    const T = new Array(M.length);
    let G = 1;
    for (let K = M.length - 1; K >= 0; K--)
      T[K] = G, G *= M[K];
    return T;
  }, k = E(t.shape), U = E(e.shape), R = E(s.shape);
  let rt = null, at = null, N = null, q = null, j = null;
  c && (q = new Array(t.dataLength()).fill(0)), d && (j = new Array(e.dataLength()).fill(0));
  const W = p / i, dt = w / i;
  for (let M = 0; M < m; M++)
    for (let T = 0; T < i; T++)
      for (let G = 0; G < dt; G++) {
        const K = T * dt + G, lt = g.reduce((b, et) => b * et, 1);
        for (let b = 0; b < lt; b++) {
          const et = new Array(u);
          let ot = b;
          for (let A = u - 1; A >= 0; A--)
            et[A] = ot % g[A], ot = Math.floor(ot / g[A]);
          let X = M * R[0] + K * R[1];
          for (let A = 0; A < u; A++) X += et[A] * R[A + 2];
          const Y = s.data[X];
          for (let A = 0; A < W; A++) {
            const ht = T * W + A, _t = I.reduce((x, L) => x * L, 1);
            for (let x = 0; x < _t; x++) {
              const L = new Array(u);
              let z = x;
              for (let O = u - 1; O >= 0; O--)
                L[O] = z % I[O], z = Math.floor(z / I[O]);
              let qt = !0;
              const Et = new Array(u);
              for (let O = 0; O < u; O++) {
                const st = et[O] * f[O] + L[O] * h[O] - _[O];
                if (st < 0 || st >= B[O]) {
                  qt = !1;
                  break;
                }
                Et[O] = st;
              }
              if (qt) {
                let O = M * k[0] + ht * k[1];
                for (let H = 0; H < u; H++) O += Et[H] * k[H + 2];
                let st = K * U[0] + A * U[1];
                for (let H = 0; H < u; H++) st += L[H] * U[H + 2];
                c && (q[O] += Y * e.data[st]), d && (j[st] += Y * t.data[O]);
              }
            }
          }
        }
      }
  if (c && (rt = new l(q, { requires_grad: !1 }, { shape: t.shape })), d && (at = new l(j, { requires_grad: !1 }, { shape: e.shape })), n && n.requires_grad) {
    const M = [0];
    for (let T = 2; T < s.shape.length; T++) M.push(T);
    N = s.sum(M);
  }
  return [rt, at, N];
}
class ge extends S {
  stride;
  padding;
  dilation;
  groups;
  _forward(t, e, n, r = 1, a = 0, o = 1, i = 1) {
    const u = y(t, e, ...n ? [n] : []);
    u && (this.saved_tensors = [t, e], n && this.saved_tensors.push(n)), this.next_functions.push(t.grad_fn ? t.grad_fn : v), this.next_functions.push(e.grad_fn ? e.grad_fn : v), n && this.next_functions.push(n.grad_fn ? n.grad_fn : v), this.stride = r, this.padding = a, this.dilation = o, this.groups = i;
    const c = bt(t, e, n, r, a, o, i, 1);
    return c.requires_grad = u, c.grad_fn = u ? this : null, c;
  }
  _backward(t) {
    const e = this.saved_tensors[0], n = this.saved_tensors[1], r = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [a, o, i] = this.next_functions, [u, c, d] = vt(
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
    e.requires_grad && a.backward(u), n.requires_grad && o.backward(c), r && r.requires_grad && i.backward(d);
  }
}
P("conv1d", ge);
class me extends S {
  stride;
  padding;
  dilation;
  groups;
  _forward(t, e, n, r = 1, a = 0, o = 1, i = 1) {
    const u = y(t, e, ...n ? [n] : []);
    u && (this.saved_tensors = [t, e], n && this.saved_tensors.push(n)), this.next_functions.push(t.grad_fn ? t.grad_fn : v), this.next_functions.push(e.grad_fn ? e.grad_fn : v), n && this.next_functions.push(n.grad_fn ? n.grad_fn : v), this.stride = r, this.padding = a, this.dilation = o, this.groups = i;
    const c = bt(t, e, n, r, a, o, i, 2);
    return c.requires_grad = u, c.grad_fn = u ? this : null, c;
  }
  _backward(t) {
    const e = this.saved_tensors[0], n = this.saved_tensors[1], r = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [a, o, i] = this.next_functions, [u, c, d] = vt(
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
    e.requires_grad && a.backward(u), n.requires_grad && o.backward(c), r && r.requires_grad && i.backward(d);
  }
}
P("conv2d", me);
class we extends S {
  stride;
  padding;
  dilation;
  groups;
  _forward(t, e, n, r = 1, a = 0, o = 1, i = 1) {
    const u = y(t, e, ...n ? [n] : []);
    u && (this.saved_tensors = [t, e], n && this.saved_tensors.push(n)), this.next_functions.push(t.grad_fn ? t.grad_fn : v), this.next_functions.push(e.grad_fn ? e.grad_fn : v), n && this.next_functions.push(n.grad_fn ? n.grad_fn : v), this.stride = r, this.padding = a, this.dilation = o, this.groups = i;
    const c = bt(t, e, n, r, a, o, i, 3);
    return c.requires_grad = u, c.grad_fn = u ? this : null, c;
  }
  _backward(t) {
    const e = this.saved_tensors[0], n = this.saved_tensors[1], r = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [a, o, i] = this.next_functions, [u, c, d] = vt(
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
    e.requires_grad && a.backward(u), n.requires_grad && o.backward(c), r && r.requires_grad && i.backward(d);
  }
}
P("conv3d", we);
C(
  (s, t, e, n) => s[e] < t[n] ? 1 : 0,
  () => {
  },
  "lt"
);
C(
  (s, t, e, n) => s[e] > t[n] ? 1 : 0,
  () => {
  },
  "gt"
);
C(
  (s, t, e, n) => s[e] <= t[n] ? 1 : 0,
  () => {
  },
  "le"
);
C(
  (s, t, e, n) => s[e] >= t[n] ? 1 : 0,
  () => {
  },
  "ge"
);
C(
  (s, t, e, n) => s[e] == t[n] ? 1 : 0,
  () => {
  },
  "eq"
);
C(
  (s, t, e, n) => s[e] != t[n] ? 1 : 0,
  () => {
  },
  "ne"
);
$(
  (s, t) => Math.max(s[t], 0),
  (s, t, e) => {
    t.backward(e.mul(e.mul(s.gt(0))));
  },
  "relu"
);
$(
  (s, t) => 1 / (1 + Math.exp(-s[t])),
  (s, t, e) => {
    const n = s.sigmoid();
    t.backward(n.mul(n.mul(-1).add(1)).mul(e));
  },
  "sigmoid"
);
class tt extends l {
  constructor(t, e = {
    requires_grad: !0
  }, n = {}) {
    t instanceof l ? super(t.data, { requires_grad: !0 }, { shape: t.shape }) : t instanceof tt ? super(t.data, { requires_grad: !0 }, { shape: t.shape }) : super(t, e, n);
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
  named_parameters(t = "") {
    const e = [];
    for (const [n, r] of Object.entries(this._parameters)) {
      const a = t ? `${t}.${n}` : n;
      e.push([a, r]);
    }
    for (const [n, r] of Object.entries(this._modules)) {
      const a = t ? `${t}.${n}` : n;
      e.push(...r.named_parameters(a));
    }
    return e;
  }
}
class xe extends nt {
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
class yt {
}
class be extends yt {
  constructor() {
    super();
  }
  forward(t, e) {
    return t.sub(e).pow(2).mean();
  }
}
class ve extends yt {
  constructor() {
    super();
  }
  forward(t, e) {
    return t.sub(e).abs().mean();
  }
}
class ye extends yt {
  weight;
  constructor(t = null) {
    super(), this.weight = t;
  }
  forward(t, e) {
    const n = e.mul(t.log()), r = e.neg().add(1).mul(t.neg().add(1).log()), a = n.add(r).neg().mean();
    return this.weight ? a.mul(this.weight) : a;
  }
}
function At(s) {
  return (...t) => Z(s).forward(...t);
}
function Dt(s) {
  return (t) => (typeof t == "number" && (t = new l(t)), Z(s).forward(t));
}
const Wt = Dt("relu"), jt = Dt("sigmoid"), Gt = At("conv1d"), Kt = At("conv2d"), Vt = At("conv3d"), Ae = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  conv1d: Gt,
  conv2d: Kt,
  conv3d: Vt,
  relu: Wt,
  sigmoid: jt
}, Symbol.toStringTag, { value: "Module" }));
class Oe extends nt {
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
class ke extends nt {
  constructor() {
    super();
  }
  forward(t) {
    return Wt(t);
  }
}
class qe extends nt {
  constructor() {
    super();
  }
  forward(t) {
    return jt(t);
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
  constructor(t, e, n, r, a, o, i, u, c) {
    if (super(), this.in_channels = t, this.out_channels = e, this.kernel_size = n, this.stride = r, this.padding = a, this.dilation = o, this.groups = i, t % i !== 0)
      throw new Error("in_channels must be divisible by groups");
    if (e % i !== 0)
      throw new Error("out_channels must be divisible by groups");
    const d = typeof n == "number" ? new Array(c).fill(n) : n, f = d.reduce((h, m) => h * m, 1), _ = Math.sqrt(i / (t * f));
    this.weight = new tt(
      pt([e, t / i, ...d]).mul(2 * _).sub(_)
    ), this.register("weight", this.weight), u ? (this.bias = new tt(
      pt([e]).mul(2 * _).sub(_)
    ), this.register("bias", this.bias)) : this.bias = null;
  }
}
class Ee extends Ot {
  constructor(t, e, n, r = 1, a = 0, o = 1, i = 1, u = !0) {
    super(t, e, n, r, a, o, i, u, 1);
  }
  forward(t) {
    return Gt(t, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}
class Re extends Ot {
  constructor(t, e, n, r = 1, a = 0, o = 1, i = 1, u = !0) {
    super(t, e, n, r, a, o, i, u, 2);
  }
  forward(t) {
    return Kt(t, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}
class Me extends Ot {
  constructor(t, e, n, r = 1, a = 0, o = 1, i = 1, u = !0) {
    super(t, e, n, r, a, o, i, u, 3);
  }
  forward(t) {
    return Vt(t, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
}
const Rs = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  BCELoss: ye,
  Conv1d: Ee,
  Conv2d: Re,
  Conv3d: Me,
  L1Loss: ve,
  Linear: Oe,
  MSELoss: be,
  Module: nt,
  Parameter: tt,
  ReLU: ke,
  Sequential: xe,
  Sigmoid: qe,
  functional: Ae
}, Symbol.toStringTag, { value: "Module" }));
class kt {
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
class Te extends kt {
  state = /* @__PURE__ */ new Map();
  lr;
  momentum;
  dampening;
  weight_decay;
  nesterov;
  maximize;
  constructor(t, e = 1e-3, n = 0, r = 0, a = 0, o = !1, i = !1) {
    super(t, {}), this.lr = e, this.momentum = n, this.dampening = r, this.weight_decay = a, this.nesterov = o, this.maximize = i;
  }
  step() {
    for (const t of this.params) {
      let e = this.maximize ? t.grad.mul(-1) : t.grad;
      if (this.weight_decay !== 0 && (e = e.add(t.mul(this.weight_decay))), this.momentum !== 0) {
        if (this.state.has(t)) {
          let a = this.state.get(t).velocity;
          a = a.mul(this.momentum), a = a.add(e.mul(1 - this.dampening)), this.state.set(t, { velocity: a });
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
class Fe extends kt {
  state = /* @__PURE__ */ new Map();
  step_count = 0;
  lr;
  beta1;
  beta2;
  eps;
  weight_decay;
  amsgrad;
  maximize;
  constructor(t, e = 1e-3, n = [0.9, 0.999], r = 1e-8, a = 0, o = !1, i = !1) {
    super(t, {}), this.lr = e, this.beta1 = n[0], this.beta2 = n[1], this.eps = r, this.weight_decay = a, this.amsgrad = o, this.maximize = i;
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
      const r = 1 - Math.pow(this.beta1, this.step_count), a = 1 - Math.pow(this.beta2, this.step_count);
      let o;
      const i = n.m.div(r);
      this.amsgrad ? (n.vmax = n.vmax.maximum(n.v), o = n.vmax.div(a)) : o = n.v.div(a);
      const u = i.div(o.sqrt().add(this.eps)).mul(this.lr), c = t.sub(u);
      t.data = c.data;
    }
  }
}
const Ms = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Adam: Fe,
  Optimizer: kt,
  SGD: Te
}, Symbol.toStringTag, { value: "Module" })), Be = {
  add: "aten.add.Tensor",
  sub: "aten.sub.Tensor",
  mul: "aten.mul.Tensor",
  div: "aten.div.Tensor",
  pow: "aten.pow.Tensor_Tensor",
  powint: "aten.pow.Tensor_Scalar",
  fmod: "aten.fmod.Tensor",
  maximum: "aten.maximum.default",
  minimum: "aten.minimum.default",
  log: "aten.log.default",
  sqrt: "aten.sqrt.default",
  exp: "aten.exp.default",
  square: "aten.square.default",
  abs: "aten.abs.default",
  sign: "aten.sign.default",
  neg: "aten.neg.default",
  reciprocal: "aten.reciprocal.default",
  nan_to_num: "aten.nan_to_num.default",
  reshape: "aten.reshape.default",
  squeeze: "aten.squeeze.dim",
  unsqueeze: "aten.unsqueeze.default",
  expand: "aten.expand.default",
  sin: "aten.sin.default",
  cos: "aten.cos.default",
  tan: "aten.tan.default",
  sum: "aten.sum.default",
  mean: "aten.mean.default",
  min: "aten.min.default",
  max: "aten.max.default",
  transpose: "aten.transpose.int",
  matmul: "aten.matmul.default",
  relu: "aten.relu.default",
  sigmoid: "aten.sigmoid.default",
  lt: "aten.lt.Tensor",
  gt: "aten.gt.Tensor",
  le: "aten.le.Tensor",
  ge: "aten.ge.Tensor",
  eq: "aten.eq.Tensor",
  ne: "aten.ne.Tensor",
  conv1d: "aten.conv1d.default",
  conv2d: "aten.conv2d.default",
  conv3d: "aten.conv3d.default",
  linear: "aten.linear.default"
};
function Ce(s) {
  return Be[s] || `aten.${s}.default`;
}
class Ne {
  counts = /* @__PURE__ */ new Map();
  generate(t) {
    const e = this.counts.get(t) || 0;
    return this.counts.set(t, e + 1), e === 0 ? t : `${t}_${e}`;
  }
}
class Ue {
  constructor(t, e, n) {
    this.graph = t, this.graph_signature = e, this.parameters = n;
  }
  toString() {
    const t = ["ExportedProgram:"], e = this.graph.filter((n) => n.op === "placeholder").map((n) => {
      const r = n.val_shape ? JSON.stringify(n.val_shape) : "?";
      return `${n.name}: "${r}"`;
    }).join(", ");
    t.push("    class GraphModule(torch.nn.Module):"), t.push(`        def forward(self, ${e}):`);
    for (const n of this.graph)
      if (n.op === "call_function") {
        const r = n.args.join(", ");
        t.push(`            ${n.name} = ${n.target}(${r})`);
      } else n.op === "output" && t.push(`            return (${n.args.join(", ")},)`);
    t.push(""), t.push("Graph signature:"), t.push("    # inputs");
    for (const n of this.graph_signature.input_specs) {
      const r = n.target ? ` target='${n.target}'` : "";
      t.push(`    ${n.name}: ${n.kind}${r}`);
    }
    t.push("    # outputs");
    for (const n of this.graph_signature.output_specs)
      t.push(`    ${n.name}: ${n.kind}`);
    return t.join(`
`);
  }
}
function Ts(s, t) {
  const e = [], n = new Ne(), r = /* @__PURE__ */ new Map(), a = s.named_parameters(), o = /* @__PURE__ */ new Set(), i = [];
  for (const [h, m] of a) {
    const p = "p_" + h.replace(/\./g, "_"), w = n.generate(p);
    r.set(m.id, w), o.add(m.id), e.push({
      op: "placeholder",
      name: w,
      target: w,
      args: [],
      val_shape: m.shape
    }), i.push({
      kind: "PARAMETER",
      name: w,
      target: h
    });
  }
  for (let h = 0; h < t.length; h++) {
    const p = n.generate("input");
    r.set(t[h].id, p), e.push({
      op: "placeholder",
      name: p,
      target: p,
      args: [],
      val_shape: t[h].shape
    }), i.push({
      kind: "USER_INPUT",
      name: p
    });
  }
  const u = (h) => {
    const { operation: m, args: p, result: w } = h.detail, B = m.opName;
    if (!B) return;
    const I = [];
    for (const E of p)
      if (E instanceof l) {
        const k = r.get(E.id);
        k && I.push(k);
      }
    const g = n.generate(B);
    r.set(w.id, g), e.push({
      op: "call_function",
      name: g,
      target: Ce(B),
      args: I,
      val_shape: w.shape
    });
  };
  V.addEventListener(
    J.OPERATION_AFTER_FORWARD,
    u
  );
  let c;
  try {
    c = Yt(() => s.forward(...t));
  } finally {
    V.removeEventListener(
      J.OPERATION_AFTER_FORWARD,
      u
    );
  }
  const d = r.get(c.id) || "output";
  e.push({
    op: "output",
    name: "output",
    target: "output",
    args: [d]
  });
  const f = [{
    kind: "USER_OUTPUT",
    name: d
  }], _ = /* @__PURE__ */ new Map();
  for (const [h, m] of a)
    _.set(h, {
      data: [...m.data],
      shape: [...m.shape]
    });
  return new Ue(
    e,
    { input_specs: i, output_specs: f },
    _
  );
}
export {
  Rt as AccumulateGrad,
  Ue as ExportedProgram,
  qs as Max,
  ks as Mean,
  Es as Min,
  Os as Sum,
  l as Tensor,
  S as TorchFunction,
  Le as __left_index__,
  De as __right_index__,
  es as abs,
  We as add,
  Ie as arange,
  ds as cos,
  Xt as disable_no_grad,
  Ke as div,
  Qt as enable_no_grad,
  ys as eq,
  V as eventBus,
  J as events,
  Ze as exp,
  us as expand,
  Ts as export_,
  Je as fmod,
  vs as ge,
  xs as gt,
  Ht as is_grad_enabled,
  bs as le,
  Pe as linspace,
  Xe as log,
  ws as lt,
  ms as matmul,
  ps as max,
  He as maximum,
  _s as mean,
  fs as min,
  Qe as minimum,
  Ge as mul,
  rs as nan_to_num,
  As as ne,
  ss as neg,
  Rs as nn,
  Yt as no_grad,
  Ut as ones,
  Se as ones_like,
  Ms as optim,
  Ve as pow,
  pt as rand,
  ze as randint,
  $e as randn,
  ns as reciprocal,
  as as reshape,
  se as sign,
  cs as sin,
  Ye as sqrt,
  ts as square,
  os as squeeze,
  je as sub,
  hs as sum,
  ls as tan,
  gs as transpose,
  is as unsqueeze,
  Jt as zeros,
  ft as zeros_like
};
