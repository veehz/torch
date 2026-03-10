var Qe = Object.defineProperty;
var c = (s, t) => Qe(s, "name", { value: t, configurable: !0 });
function Pe(s, t) {
  const e = Math.max(s.length, t.length), n = [...Array(e - s.length).fill(1), ...s], r = [...Array(e - t.length).fill(1), ...t], i = [];
  for (let a = 0; a < e; a++) {
    if (n[a] !== r[a] && n[a] !== 1 && r[a] !== 1)
      throw new Error(`Shape mismatch: ${s} and ${t}`);
    i.push(Math.max(n[a], r[a]));
  }
  return i;
}
c(Pe, "_broadcast_shape");
function $e(s, t, e) {
  const n = dt(t, s), r = new Array(t.reduce((i, a) => i * a, 1)).fill(0);
  for (let i = 0; i < e.length; i++)
    r[ht(n, s, i)] += e[i];
  return r;
}
c($e, "_unbroadcast");
function dt(s, t) {
  return s.length >= t.length ? s : [...Array(t.length - s.length).fill(1), ...s];
}
c(dt, "_pad_shape");
function ht(s, t, e) {
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
c(ht, "_get_original_index");
function qt(s) {
  return Array.isArray(s[0]) ? s[0] : s;
}
c(qt, "get_shape_from_args");
function ls(...s) {
  const t = qt(s), e = new l(Array(t.reduce((n, r) => n * r, 1)).fill(Math.random()));
  return e.shape = t, e;
}
c(ls, "randn");
function bt(...s) {
  const t = qt(s), e = new l(Array(t.reduce((n, r) => n * r, 1)).fill(Math.random()));
  return e.shape = t, e;
}
c(bt, "rand");
function _s(s, t, e) {
  const n = new l(
    Array(e.reduce((r, i) => r * i, 1)).fill(Math.floor(Math.random() * (t - s) + s))
  );
  return n.shape = e, n;
}
c(_s, "randint");
function Se(...s) {
  const t = qt(s), e = new l(Array(t.reduce((n, r) => n * r, 1)).fill(1));
  return e.shape = t, e;
}
c(Se, "ones");
function Xe(...s) {
  const t = qt(s), e = new l(Array(t.reduce((n, r) => n * r, 1)).fill(0));
  return e.shape = t, e;
}
c(Xe, "zeros");
function fs(s) {
  return Se(s.shape);
}
c(fs, "ones_like");
function xt(s) {
  return Xe(s.shape);
}
c(xt, "zeros_like");
function ps(s, t, e) {
  const n = [], r = (t - s) / (e - 1);
  for (let i = 0; i < e - 1; i++)
    n.push(s + i * r);
  return n.push(t), new l(n);
}
c(ps, "linspace");
function gs(s, t = void 0, e = 1) {
  const n = [];
  for (let r = s; r < t; r += e)
    n.push(r);
  return new l(n);
}
c(gs, "arange");
let yt = !0;
function Ye() {
  return yt;
}
c(Ye, "is_grad_enabled");
function Ze() {
  const s = yt;
  return yt = !1, s;
}
c(Ze, "enable_no_grad");
function ts(s) {
  yt = s;
}
c(ts, "disable_no_grad");
function ms(s) {
  const t = Ze();
  try {
    return s();
  } finally {
    ts(t);
  }
}
c(ms, "no_grad");
let es = 0;
const ze = /* @__PURE__ */ c(() => es++, "getNextId"), Z = new EventTarget(), tt = {
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
  if (!Ye()) return !1;
  for (const t of s)
    if (t instanceof l && t.requires_grad)
      return !0;
  return !1;
}
c(y, "resultRequiresGrad");
const se = class se {
  id = ze();
  next_functions = [];
  saved_tensors = [];
  _retained_tensors = [];
  forward(...t) {
    const e = y(...t);
    Z.dispatchEvent(new CustomEvent(tt.OPERATION_BEFORE_FORWARD, {
      detail: {
        operation: this,
        requires_grad: e,
        args: t
      }
    }));
    const n = this._forward(...t);
    return Z.dispatchEvent(new CustomEvent(tt.OPERATION_AFTER_FORWARD, {
      detail: {
        operation: this,
        requires_grad: e,
        args: t,
        result: n
      }
    })), n;
  }
  backward(t) {
    Z.dispatchEvent(new CustomEvent(tt.OPERATION_BEFORE_BACKWARD, { detail: { operation: this, dz: t } }));
    for (const e of this._retained_tensors)
      e.grad || (e.grad = new l(new Array(e.dataLength()).fill(0))), e.grad = e.grad.add(t);
    this._backward(t), Z.dispatchEvent(new CustomEvent(tt.OPERATION_AFTER_BACKWARD, { detail: { operation: this, dz: t } }));
  }
};
c(se, "TorchFunction");
let D = se;
const ne = class ne extends D {
  _forward(...t) {
    throw new Error("NullOp should not be called");
  }
  _backward(t) {
  }
};
c(ne, "NullOp");
let Ft = ne;
const b = new Ft(), re = class re extends D {
};
c(re, "UnaryFunction");
let vt = re;
const ae = class ae extends D {
};
c(ae, "BinaryFunction");
let At = ae;
const ie = class ie extends vt {
  variable;
  _forward(t) {
    return this.variable = t, t;
  }
  _backward(t) {
    if (this.variable.grad || (this.variable.grad = xt(this.variable)), Z.dispatchEvent(new CustomEvent(tt.OPERATION_BEFORE_ACCUMULATE_GRAD, { detail: { operation: this, dz: t } })), typeof t == "number")
      this.variable.grad = this.variable.grad.add(t);
    else {
      const e = $e(t.shape, this.variable.shape, t.data);
      this.variable.grad = this.variable.grad.add(new l(e, {}, { shape: this.variable.shape }));
    }
    Z.dispatchEvent(new CustomEvent(tt.OPERATION_AFTER_ACCUMULATE_GRAD, { detail: { operation: this, dz: t } }));
  }
};
c(ie, "AccumulateGrad");
let kt = ie;
const Ce = /* @__PURE__ */ new Map(), Rt = /* @__PURE__ */ new Map();
function $(s, t) {
  Ce.set(s, t);
}
c($, "registerOperation");
function J(s) {
  const t = Ce.get(s);
  if (!t)
    throw new Error(`Operation '${s}' is not registered.`);
  return t;
}
c(J, "getOperation");
function Ue(s) {
  const t = Rt.get(s);
  return t || (Rt.set(s, new (J(s))()), Rt.get(s));
}
c(Ue, "getOperationCache");
function ss(s) {
  if (ArrayBuffer.isView(s))
    return [s.length];
  const t = [];
  for (; Array.isArray(s); )
    t.push(s.length), s = s[0];
  return t;
}
c(ss, "_get_shape");
function je(s) {
  return Array.isArray(s) ? s.flatMap((t) => je(t)) : ArrayBuffer.isView(s) ? Array.from(s) : [s];
}
c(je, "_flatten");
const at = class at {
  // Auto-generated ID
  id = ze();
  // Optional user-defined name
  name = null;
  data;
  shape;
  grad_fn = null;
  grad = null;
  requires_grad;
  constructor(t, e = {}, n = {}) {
    if (this.data = je(t), this.requires_grad = e.requires_grad ?? !1, e.name && (this.name = e.name), this.shape = n.shape ?? ss(t), this.grad_fn = n.operation ?? null, this.requires_grad && !this.grad_fn) {
      const r = new kt();
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
    const e = this.data, n = /* @__PURE__ */ c((r) => {
      const i = this.shape[r], a = new Array(i), o = r === this.shape.length - 1;
      for (let u = 0; u < i; u++)
        o ? a[u] = e[t++] : a[u] = n(r + 1);
      return a;
    }, "buildDimension");
    return n(0);
  }
  dataLength() {
    return this.data.length;
  }
  _executeUnaryOp(t) {
    return (y(this) ? new (J(t))() : Ue(t)).forward(this);
  }
  _executeBinaryOp(t, e) {
    return typeof e == "number" && (e = new at(e)), (y(this, e) ? new (J(t))() : Ue(t)).forward(this, e);
  }
  _executeOpRaw(t, ...e) {
    return new (J(t))().forward(this, ...e);
  }
  item() {
    if (this.dataLength() !== 1)
      throw new Error("Tensor.item() is only valid for scalars");
    return this.data[0];
  }
  detach() {
    return new at(this.data, { requires_grad: !1 }, { shape: this.shape });
  }
  detach_() {
    this.requires_grad = !1, this.grad = null, this.grad_fn = null;
  }
  zero_() {
    this.data = Array(this.dataLength()).fill(0);
  }
  is_retain_grad = !1;
  retain_grad() {
    this.grad_fn instanceof kt || this.is_retain_grad || (this.is_retain_grad = !0, this.grad_fn._retained_tensors.push(this));
  }
  backward(t) {
    if (this.requires_grad) {
      if (t)
        t.toArray_();
      else {
        if (this.dataLength() !== 1)
          throw new Error("Gradient is required for non-scalar tensors");
        t = new at(1);
      }
      this.grad_fn && (Z.dispatchEvent(new CustomEvent(tt.TENSOR_BEFORE_BACKWARD, { detail: { tensor: this } })), this.grad_fn.backward(t), Z.dispatchEvent(new CustomEvent(tt.TENSOR_AFTER_BACKWARD, { detail: { tensor: this } })));
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
};
c(at, "Tensor");
let l = at;
function Q(s) {
  return (...t) => new (J(s))().forward(...t);
}
c(Q, "generate_function$1");
function C(s) {
  return (t) => (typeof t == "number" && (t = new l(t)), new (J(s))().forward(t));
}
c(C, "generate_unary_function$1");
function R(s) {
  return (t, e) => (typeof t == "number" && (t = new l(t)), typeof e == "number" && (e = new l(e)), new (J(s))().forward(t, e));
}
c(R, "generate_binary_function$1");
const ws = R("__left_index__"), xs = R("__right_index__"), bs = R("add"), ys = R("sub"), vs = R("mul"), As = R("div"), ks = R("pow"), Os = R("fmod"), qs = R("maximum"), Es = R("minimum"), Rs = C("log"), Fs = C("sqrt"), Ms = C("exp"), Bs = C("square"), Ts = C("abs"), ns = C("sign"), Is = C("neg"), Ds = C("reciprocal"), Us = C("nan_to_num"), Ws = Q("reshape"), Ns = Q("squeeze"), Ps = Q("unsqueeze"), $s = Q("expand"), Ss = C("sin"), zs = C("cos"), Cs = C("tan"), js = Q("sum"), Ks = Q("mean"), Ls = Q("min"), Gs = Q("max"), Vs = Q("transpose"), Hs = R("matmul"), Js = R("lt"), Qs = R("gt"), Xs = R("le"), Ys = R("ge"), Zs = R("eq"), tn = R("ne");
function We(s) {
  const t = new Array(s.length).fill(1);
  for (let e = s.length - 2; e >= 0; e--)
    t[e] = t[e + 1] * s[e + 1];
  return t;
}
c(We, "_get_strides");
function rs(s, t) {
  return t.map((e) => {
    const n = Math.floor(s / e);
    return s %= e, n;
  });
}
c(rs, "_unravel_index");
function as(s, t) {
  return s.reduce((e, n, r) => e + n * t[r], 0);
}
c(as, "_ravel_index");
function Mt(s, t, e = !1) {
  if (t === void 0) return e ? s.map(() => 1) : [];
  const r = (Array.isArray(t) ? t : [t]).map((i) => i < 0 ? i + s.length : i);
  return e ? s.map((i, a) => r.includes(a) ? 1 : i) : s.filter((i, a) => !r.includes(a));
}
c(Mt, "_get_reduction_shape");
function B(s, t, e = null) {
  const n = /* @__PURE__ */ c((a, o, u, d, h, f) => {
    const _ = Array(f);
    for (let p = 0; p < f; p++) {
      const F = ht(o, h, p), w = ht(d, h, p);
      _[p] = s(a, u, F, w);
    }
    return _;
  }, "kernel"), r = /* @__PURE__ */ c((a, o, u = null) => {
    const d = Pe(a.shape, o.shape), h = dt(a.shape, d), f = dt(o.shape, d), _ = d.reduce((p, F) => p * F, 1);
    return new l(
      n(
        a.data,
        h,
        o.data,
        f,
        d,
        _
      ),
      { requires_grad: y(a, o) },
      { operation: u, shape: d }
    );
  }, "forward_tensor"), i = {
    [e]: class extends At {
      _forward(a, o) {
        const u = y(a, o);
        return u && (this.saved_tensors = [a, o]), this.next_functions.push(a.grad_fn ? a.grad_fn : b), this.next_functions.push(o.grad_fn ? o.grad_fn : b), r(a, o, u ? this : null);
      }
      _backward(a) {
        const [o, u] = this.saved_tensors, [d, h] = this.next_functions;
        t(o, u, d, h, a);
      }
    }
  }[e];
  return e && $(e, i), i;
}
c(B, "BinaryFunctionMixin");
function N(s, t, e = null) {
  const n = /* @__PURE__ */ c((a, o) => {
    const u = Array(o);
    for (let d = 0; d < o; d++)
      u[d] = s(a, d);
    return u;
  }, "kernel"), r = /* @__PURE__ */ c((a, o = null) => {
    const u = a.dataLength();
    return new l(
      n(a.data, u),
      { requires_grad: y(a) },
      { operation: o, shape: a.shape }
    );
  }, "forward_tensor"), i = {
    [e]: class extends vt {
      _forward(a) {
        const o = y(a);
        return o && (this.saved_tensors = [a]), this.next_functions.push(a.grad_fn ? a.grad_fn : b), r(a, o ? this : null);
      }
      _backward(a) {
        const [o] = this.saved_tensors, [u] = this.next_functions;
        t(o, u, a);
      }
    }
  }[e];
  return e && $(e, i), i;
}
c(N, "UnaryFunctionMixin");
function Et(s, t, e, n = null, r) {
  const i = {
    [n]: class extends D {
      dim;
      keepdim;
      _forward(a, o, u = !1) {
        this.dim = o, this.keepdim = u;
        const d = y(a);
        d && (this.saved_tensors = [a]), this.next_functions.push(a.grad_fn ? a.grad_fn : b);
        const h = Mt(a.shape, o, u), f = h.reduce((g, U) => g * U, 1), _ = new Array(f).fill(s), p = new Array(f).fill(0), F = We(a.shape), w = We(h), S = (o === void 0 ? [] : Array.isArray(o) ? o : [o]).map((g) => g < 0 ? g + a.shape.length : g), V = o === void 0;
        for (let g = 0; g < a.data.length; g++) {
          const U = rs(g, F);
          let T;
          if (V)
            T = u ? U.map(() => 0) : [];
          else {
            T = [];
            for (let O = 0; O < a.shape.length; O++)
              S.includes(O) ? u && T.push(0) : T.push(U[O]);
          }
          const W = as(T, w);
          _[W] = t(_[W], a.data[g]), p[W]++;
        }
        if (r)
          for (let g = 0; g < f; g++)
            _[g] = r(_[g], p[g]);
        return new l(
          _,
          { requires_grad: d },
          { operation: d ? this : null, shape: h }
        );
      }
      _backward(a) {
        const [o] = this.saved_tensors, [u] = this.next_functions;
        let d = a;
        const h = Mt(o.shape, this.dim, !0);
        a.shape.length !== h.length && (d = a.reshape(h));
        const f = d.expand(o.shape), _ = e(o, f, this.dim, this.keepdim);
        u.backward(_);
      }
    }
  }[n];
  return n && $(n, i), i;
}
c(Et, "ReductionFunctionMixin");
function ut(s, t) {
  const e = $e(s.shape, t, s.data);
  return new l(e, { requires_grad: s.requires_grad }, { shape: t });
}
c(ut, "unbroadcast");
function is(s, t) {
  return s.mul(Se(t));
}
c(is, "broadcast");
const en = B(
  (s, t, e, n) => e,
  (s, t, e, n, r) => {
  },
  "__left_index__"
), sn = B(
  (s, t, e, n) => n,
  (s, t, e, n, r) => {
  },
  "__right_index__"
), nn = B(
  (s, t, e, n) => s[e] + t[n],
  (s, t, e, n, r) => {
    e.backward(r), n.backward(r);
  },
  "add"
), rn = B(
  (s, t, e, n) => s[e] - t[n],
  (s, t, e, n, r) => {
    e.backward(r), n.backward(r.mul(new l(-1)));
  },
  "sub"
), an = B(
  (s, t, e, n) => s[e] * t[n],
  (s, t, e, n, r) => {
    e.backward(r.mul(t)), n.backward(r.mul(s));
  },
  "mul"
), on = B(
  (s, t, e, n) => s[e] / t[n],
  (s, t, e, n, r) => {
    e.backward(r.div(t)), n.backward(r.mul(s).mul(new l(-1)).div(t).div(t));
  },
  "div"
);
function Ne(s, t, e) {
  const n = typeof e == "number" ? e : null, r = new Array(t.dataLength());
  for (let i = 0; i < r.length; i++)
    r[i] = s.data[i] ? t.data[i] : n !== null ? n : e.data[i];
  return new l(r, {}, { shape: t.shape });
}
c(Ne, "_where");
const cn = B(
  (s, t, e, n) => Math.pow(s[e], t[n]),
  (s, t, e, n, r) => {
    const i = r.mul(t).mul(s.pow(t.sub(new l(1)))), a = r.mul(s.pow(t)).mul(s.log());
    e.backward(Ne(s.ne(0), i, i.nan_to_num())), n.backward(Ne(s.ne(0), a, 0));
  },
  "pow"
), un = B(
  (s, t, e, n) => s[e] % t[n],
  (s, t, e, n, r) => {
    e.backward(r);
  },
  "fmod"
), dn = B(
  (s, t, e, n) => Math.max(s[e], t[n]),
  (s, t, e, n, r) => {
    const i = s.eq(t), a = s.gt(t).add(i.mul(new l(0.5))), o = t.gt(s).add(i.mul(new l(0.5)));
    e.backward(r.mul(a)), n.backward(r.mul(o));
  },
  "maximum"
), hn = B(
  (s, t, e, n) => Math.min(s[e], t[n]),
  (s, t, e, n, r) => {
    const i = s.eq(t), a = s.lt(t).add(i.mul(new l(0.5))), o = t.lt(s).add(i.mul(new l(0.5)));
    e.backward(r.mul(a)), n.backward(r.mul(o));
  },
  "minimum"
);
function os(s, t, e = null) {
  const n = new Array(s.dataLength());
  for (let r = 0; r < n.length; r++)
    n[r] = Math.pow(s.data[r], t);
  return new l(
    n,
    { requires_grad: y(s) },
    { operation: e, shape: s.shape }
  );
}
c(os, "_powint_tensor");
const oe = class oe extends D {
  n;
  _forward(t, e) {
    const n = y(t);
    return n && (this.saved_tensors = [t], this.n = e), this.next_functions.push(t.grad_fn ? t.grad_fn : b), os(t, e, n ? this : null);
  }
  _backward(t) {
    const [e] = this.saved_tensors, n = this.n, [r] = this.next_functions;
    r.backward(t.mul(n).mul(e.pow(n - 1)));
  }
};
c(oe, "PowInt");
let Bt = oe;
$("powint", Bt);
const ln = N(
  (s, t) => Math.log(s[t]),
  (s, t, e) => {
    t.backward(e.mul(new l(1).div(s)));
  },
  "log"
), _n = N(
  (s, t) => Math.sqrt(s[t]),
  (s, t, e) => {
    t.backward(e.mul(new l(1).div(s.sqrt()).div(2)));
  },
  "sqrt"
), fn = N(
  (s, t) => Math.exp(s[t]),
  (s, t, e) => {
    t.backward(e.mul(e.mul(s.exp())));
  },
  "exp"
), pn = N(
  (s, t) => s[t] * s[t],
  (s, t, e) => {
    t.backward(e.mul(e.mul(s).mul(new l(2))));
  },
  "square"
), gn = N(
  (s, t) => Math.abs(s[t]),
  (s, t, e) => {
    t.backward(e.mul(e.mul(ns(s))));
  },
  "abs"
), mn = N(
  (s, t) => Math.sign(s[t]),
  (s, t, e) => {
    t.backward(0);
  },
  "sign"
), wn = N(
  (s, t) => -s[t],
  (s, t, e) => {
    t.backward(e.mul(e.mul(new l(-1))));
  },
  "neg"
), xn = N(
  (s, t) => 1 / s[t],
  (s, t, e) => {
    t.backward(e.mul(e.mul(s.pow(-2))).neg());
  },
  "reciprocal"
), bn = N(
  (s, t) => {
    const e = s[t];
    return Number.isNaN(e) ? 0 : e === 1 / 0 ? 34028235e31 : e === -1 / 0 ? -34028235e31 : e;
  },
  (s, t, e) => {
    t.backward(e);
  },
  "nan_to_num"
), ce = class ce extends D {
  _forward(t, e) {
    const n = t.dataLength(), r = e.reduce((a, o) => a * o, 1);
    if (n !== r)
      throw new Error("Shape mismatch: " + t.shape + " and " + e);
    const i = y(t);
    return i && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(b), new l(
      t.data,
      { requires_grad: i },
      { operation: i ? this : null, shape: e }
    );
  }
  _backward(t) {
    const [e] = this.saved_tensors, [n] = this.next_functions;
    n.backward(t.reshape(e.shape));
  }
};
c(ce, "Reshape");
let Tt = ce;
$("reshape", Tt);
const ue = class ue extends D {
  _forward(t, e) {
    const n = y(t);
    n && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(b);
    let r = [...t.shape];
    return e !== void 0 ? (e < 0 && (e += t.shape.length), r[e] === 1 && r.splice(e, 1)) : r = r.filter((i) => i !== 1), new l(
      t.data,
      { requires_grad: n },
      { operation: n ? this : null, shape: r }
    );
  }
  _backward(t) {
    const [e] = this.saved_tensors, [n] = this.next_functions;
    n.backward(t.reshape(e.shape));
  }
};
c(ue, "Squeeze");
let It = ue;
$("squeeze", It);
const de = class de extends D {
  _forward(t, e) {
    const n = y(t);
    n && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(b), e < 0 && (e += t.shape.length + 1);
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
};
c(de, "Unsqueeze");
let Dt = de;
$("unsqueeze", Dt);
const he = class he extends D {
  _forward(t, e) {
    const n = y(t);
    n && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(b);
    const r = e.length - t.shape.length, i = e.map((o, u) => {
      if (o === -1) {
        const d = u - r;
        return d >= 0 ? t.shape[d] : 1;
      }
      return o;
    }), a = is(t, i).data;
    return new l(
      a,
      { requires_grad: n },
      { operation: n ? this : null, shape: i }
    );
  }
  _backward(t) {
    const [e] = this.saved_tensors, [n] = this.next_functions;
    n.backward(ut(t, e.shape));
  }
};
c(he, "Expand");
let Ut = he;
$("expand", Ut);
const yn = N(
  (s, t) => Math.sin(s[t]),
  (s, t, e) => {
    t.backward(e.mul(e.mul(s.cos())));
  },
  "sin"
), vn = N(
  (s, t) => Math.cos(s[t]),
  (s, t, e) => {
    t.backward(e.mul(e.mul(s.sin().neg())));
  },
  "cos"
), An = N(
  (s, t) => Math.tan(s[t]),
  (s, t, e) => {
    t.backward(e.mul(e.mul(s.cos().pow(-2))));
  },
  "tan"
), kn = Et(
  0,
  (s, t) => s + t,
  (s, t) => t,
  "sum"
), On = Et(
  0,
  (s, t) => s + t,
  (s, t, e) => {
    const n = Mt(s.shape, e, !1), r = n.length > 0 ? n.reduce((a, o) => a * o, 1) : 1, i = s.dataLength() / r;
    return t.mul(new l([1 / i]));
  },
  "mean",
  (s, t) => s / t
), qn = Et(
  -1 / 0,
  (s, t) => Math.max(s, t),
  (s, t, e) => {
    const r = s.max(e, !0).expand(s.shape), i = s.eq(r).detach();
    return t.mul(i);
  },
  "max"
), En = Et(
  1 / 0,
  (s, t) => Math.min(s, t),
  (s, t, e) => {
    const r = s.min(e, !0).expand(s.shape), i = s.eq(r).detach();
    return t.mul(i);
  },
  "min"
);
function cs(s, t, e, n = null) {
  if (s.shape.length + t < 0 || s.shape.length + e < 0)
    throw new Error(`Transpose: Dimension out of range (${t} and ${e})`);
  t = t < 0 ? s.shape.length + t : t, e = e < 0 ? s.shape.length + e : e;
  const r = [...s.shape];
  [r[t], r[e]] = [r[e], r[t]];
  const i = s.dataLength(), a = new Array(i), o = new Array(s.shape.length), u = new Array(r.length);
  for (let d = s.shape.length - 1, h = 1; d >= 0; d--)
    o[d] = h, h *= s.shape[d];
  for (let d = r.length - 1, h = 1; d >= 0; d--)
    u[d] = h, h *= r[d];
  for (let d = 0; d < i; d++) {
    let h = d, f = 0;
    for (let _ = 0; _ < r.length; _++) {
      const p = u[_], F = Math.floor(h / p);
      h %= p;
      let w = _;
      _ === t ? w = e : _ === e && (w = t), f += F * o[w];
    }
    a[d] = s.data[f];
  }
  return new l(
    a,
    { requires_grad: y(s) },
    { operation: n, shape: r }
  );
}
c(cs, "_transpose_tensor");
const le = class le extends D {
  dim0;
  dim1;
  _forward(t, e, n) {
    const r = y(t);
    return r && (this.saved_tensors = [t], this.dim0 = e, this.dim1 = n), this.next_functions.push(t.grad_fn ? t.grad_fn : b), cs(t, e, n, r ? this : null);
  }
  _backward(t) {
    const [e] = this.saved_tensors, n = this.dim0, r = this.dim1, [i] = this.next_functions;
    i.backward(t.transpose(n, r));
  }
};
c(le, "Transpose");
let Wt = le;
$("transpose", Wt);
function us(s, t, e = null) {
  if (s.shape.length == 1 && t.shape.length == 1)
    return [s.mul(t).sum(), []];
  const n = s.shape.length == 1, r = t.shape.length == 1, i = n ? [1, s.shape[0]] : s.shape, a = r ? [t.shape[0], 1] : t.shape;
  if (i[i.length - 1] != a[a.length - 2])
    throw new Error("Shape mismatch: " + s.shape + " and " + t.shape);
  const o = Pe(i.slice(0, -2), a.slice(0, -2)).concat([
    i[i.length - 2],
    a[a.length - 1]
  ]), u = o.reduce((M, S) => M * S, 1), d = new Array(u).fill(0), h = dt(i, o), f = dt(a, o), _ = o[o.length - 2], p = o[o.length - 1], F = i[i.length - 1];
  for (let M = 0; M < u; M++) {
    const S = M % (_ * p), V = Math.floor(S / p), g = S % p, U = ht(h, o, M - g), T = ht(f, o, M - V * p);
    let W = 0;
    for (let O = 0; O < F; O++)
      W += s.data[U + O] * t.data[T + O * p];
    d[M] = W;
  }
  let w = [...o];
  return n && (w = w.slice(0, -2).concat([o[o.length - 1]])), r && (w = w.slice(0, -1)), [new l(
    d,
    { requires_grad: y(s, t) },
    { operation: e, shape: w }
  ), w];
}
c(us, "_matmul_tensor");
const _e = class _e extends At {
  shape;
  _forward(t, e) {
    const n = y(t, e);
    n && (this.saved_tensors = [t, e]), this.next_functions.push(t.grad_fn ? t.grad_fn : b), this.next_functions.push(e.grad_fn ? e.grad_fn : b);
    const r = us(t, e, n ? this : null);
    return this.shape = r[1], r[0];
  }
  _backward(t) {
    const [e, n] = this.saved_tensors, [r, i] = this.next_functions;
    if (e.shape.length === 1 && n.shape.length === 1) {
      r.backward(t.mul(n)), i.backward(t.mul(e));
      return;
    }
    if (e.shape.length === 1) {
      const u = t.unsqueeze(-2), d = e.unsqueeze(-2);
      let h = u.matmul(n.transpose(-2, -1)), f = d.transpose(-2, -1).matmul(u);
      h = h.squeeze(-2), f = ut(f, n.shape), r.backward(h), i.backward(f);
      return;
    }
    if (n.shape.length === 1) {
      const u = t.unsqueeze(-1), d = n.unsqueeze(-1);
      let h = u.matmul(d.transpose(-2, -1)), f = e.transpose(-2, -1).matmul(u);
      h = ut(h, e.shape), f = f.squeeze(-1), r.backward(h), i.backward(f);
      return;
    }
    let a = t.matmul(n.transpose(-2, -1)), o = e.transpose(-2, -1).matmul(t);
    a = ut(a, e.shape), o = ut(o, n.shape), r.backward(a), i.backward(o);
  }
};
c(_e, "Matmul");
let Nt = _e;
$("matmul", Nt);
function Zt(s, t, e, n, r, i, a, o) {
  const u = typeof n == "number" ? new Array(o).fill(n) : n, d = typeof r == "number" ? new Array(o).fill(r) : r, h = typeof i == "number" ? new Array(o).fill(i) : i, f = s.shape[0], _ = s.shape[1], p = t.shape[0], F = s.shape.slice(2), w = t.shape.slice(2);
  if (_ !== t.shape[1] * a)
    throw new Error(`in_channels (${_}) must be divisible by groups (${a}) and match weight.shape[1] * groups (${t.shape[1] * a})`);
  const M = F.map((I, k) => Math.floor((I + 2 * d[k] - h[k] * (w[k] - 1) - 1) / u[k] + 1)), S = [f, p, ...M], V = S.reduce((I, k) => I * k, 1), g = new Array(V).fill(0), U = /* @__PURE__ */ c((I) => {
    const k = new Array(I.length);
    let K = 1;
    for (let j = I.length - 1; j >= 0; j--)
      k[j] = K, K *= I[j];
    return k;
  }, "get_strides"), T = U(s.shape), W = U(t.shape), O = U(S), it = _ / a, ot = p / a;
  for (let I = 0; I < f; I++)
    for (let k = 0; k < a; k++)
      for (let K = 0; K < ot; K++) {
        const j = k * ot + K, pt = M.reduce((q, E) => q * E, 1);
        for (let q = 0; q < pt; q++) {
          const E = new Array(o);
          let L = q;
          for (let x = o - 1; x >= 0; x--)
            E[x] = L % M[x], L = Math.floor(L / M[x]);
          let G = e ? e.data[j] : 0;
          for (let x = 0; x < it; x++) {
            const nt = k * it + x, ct = w.reduce((X, Y) => X * Y, 1);
            for (let X = 0; X < ct; X++) {
              const Y = new Array(o);
              let v = X;
              for (let m = o - 1; m >= 0; m--)
                Y[m] = v % w[m], v = Math.floor(v / w[m]);
              let mt = !0;
              const wt = new Array(o);
              for (let m = 0; m < o; m++) {
                const z = E[m] * u[m] + Y[m] * h[m] - d[m];
                if (z < 0 || z >= F[m]) {
                  mt = !1;
                  break;
                }
                wt[m] = z;
              }
              if (mt) {
                let m = I * T[0] + nt * T[1];
                for (let P = 0; P < o; P++) m += wt[P] * T[P + 2];
                let z = j * W[0] + x * W[1];
                for (let P = 0; P < o; P++) z += Y[P] * W[P + 2];
                G += s.data[m] * t.data[z];
              }
            }
          }
          let gt = I * O[0] + j * O[1];
          for (let x = 0; x < o; x++) gt += E[x] * O[x + 2];
          g[gt] = G;
        }
      }
  return new l(g, { requires_grad: !1 }, { shape: S });
}
c(Zt, "_convNd_forward");
function te(s, t, e, n, r, i, a, o, u, d, h) {
  const f = typeof r == "number" ? new Array(u).fill(r) : r, _ = typeof i == "number" ? new Array(u).fill(i) : i, p = typeof a == "number" ? new Array(u).fill(a) : a, F = t.shape[0], w = t.shape[1], M = e.shape[0], S = t.shape.slice(2), V = e.shape.slice(2), g = s.shape.slice(2), U = /* @__PURE__ */ c((q) => {
    const E = new Array(q.length);
    let L = 1;
    for (let G = q.length - 1; G >= 0; G--)
      E[G] = L, L *= q[G];
    return E;
  }, "get_strides"), T = U(t.shape), W = U(e.shape), O = U(s.shape);
  let it = null, ot = null, I = null, k = null, K = null;
  d && (k = new Array(t.dataLength()).fill(0)), h && (K = new Array(e.dataLength()).fill(0));
  const j = w / o, pt = M / o;
  for (let q = 0; q < F; q++)
    for (let E = 0; E < o; E++)
      for (let L = 0; L < pt; L++) {
        const G = E * pt + L, gt = g.reduce((x, nt) => x * nt, 1);
        for (let x = 0; x < gt; x++) {
          const nt = new Array(u);
          let ct = x;
          for (let v = u - 1; v >= 0; v--)
            nt[v] = ct % g[v], ct = Math.floor(ct / g[v]);
          let X = q * O[0] + G * O[1];
          for (let v = 0; v < u; v++) X += nt[v] * O[v + 2];
          const Y = s.data[X];
          for (let v = 0; v < j; v++) {
            const mt = E * j + v, wt = V.reduce((m, z) => m * z, 1);
            for (let m = 0; m < wt; m++) {
              const z = new Array(u);
              let P = m;
              for (let A = u - 1; A >= 0; A--)
                z[A] = P % V[A], P = Math.floor(P / V[A]);
              let Ie = !0;
              const De = new Array(u);
              for (let A = 0; A < u; A++) {
                const rt = nt[A] * f[A] + z[A] * p[A] - _[A];
                if (rt < 0 || rt >= S[A]) {
                  Ie = !1;
                  break;
                }
                De[A] = rt;
              }
              if (Ie) {
                let A = q * T[0] + mt * T[1];
                for (let H = 0; H < u; H++) A += De[H] * T[H + 2];
                let rt = G * W[0] + v * W[1];
                for (let H = 0; H < u; H++) rt += z[H] * W[H + 2];
                d && (k[A] += Y * e.data[rt]), h && (K[rt] += Y * t.data[A]);
              }
            }
          }
        }
      }
  if (d && (it = new l(k, { requires_grad: !1 }, { shape: t.shape })), h && (ot = new l(K, { requires_grad: !1 }, { shape: e.shape })), n && n.requires_grad) {
    const q = [0];
    for (let E = 2; E < s.shape.length; E++) q.push(E);
    I = s.sum(q);
  }
  return [it, ot, I];
}
c(te, "_convNd_backward");
const fe = class fe extends D {
  stride;
  padding;
  dilation;
  groups;
  _forward(t, e, n, r = 1, i = 0, a = 1, o = 1) {
    const u = y(t, e, ...n ? [n] : []);
    u && (this.saved_tensors = [t, e], n && this.saved_tensors.push(n)), this.next_functions.push(t.grad_fn ? t.grad_fn : b), this.next_functions.push(e.grad_fn ? e.grad_fn : b), n && this.next_functions.push(n.grad_fn ? n.grad_fn : b), this.stride = r, this.padding = i, this.dilation = a, this.groups = o;
    const d = Zt(t, e, n, r, i, a, o, 1);
    return d.requires_grad = u, d.grad_fn = u ? this : null, d;
  }
  _backward(t) {
    const e = this.saved_tensors[0], n = this.saved_tensors[1], r = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [i, a, o] = this.next_functions, [u, d, h] = te(
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
    e.requires_grad && i.backward(u), n.requires_grad && a.backward(d), r && r.requires_grad && o.backward(h);
  }
};
c(fe, "Conv1dOp");
let Pt = fe;
$("conv1d", Pt);
const pe = class pe extends D {
  stride;
  padding;
  dilation;
  groups;
  _forward(t, e, n, r = 1, i = 0, a = 1, o = 1) {
    const u = y(t, e, ...n ? [n] : []);
    u && (this.saved_tensors = [t, e], n && this.saved_tensors.push(n)), this.next_functions.push(t.grad_fn ? t.grad_fn : b), this.next_functions.push(e.grad_fn ? e.grad_fn : b), n && this.next_functions.push(n.grad_fn ? n.grad_fn : b), this.stride = r, this.padding = i, this.dilation = a, this.groups = o;
    const d = Zt(t, e, n, r, i, a, o, 2);
    return d.requires_grad = u, d.grad_fn = u ? this : null, d;
  }
  _backward(t) {
    const e = this.saved_tensors[0], n = this.saved_tensors[1], r = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [i, a, o] = this.next_functions, [u, d, h] = te(
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
    e.requires_grad && i.backward(u), n.requires_grad && a.backward(d), r && r.requires_grad && o.backward(h);
  }
};
c(pe, "Conv2dOp");
let $t = pe;
$("conv2d", $t);
const ge = class ge extends D {
  stride;
  padding;
  dilation;
  groups;
  _forward(t, e, n, r = 1, i = 0, a = 1, o = 1) {
    const u = y(t, e, ...n ? [n] : []);
    u && (this.saved_tensors = [t, e], n && this.saved_tensors.push(n)), this.next_functions.push(t.grad_fn ? t.grad_fn : b), this.next_functions.push(e.grad_fn ? e.grad_fn : b), n && this.next_functions.push(n.grad_fn ? n.grad_fn : b), this.stride = r, this.padding = i, this.dilation = a, this.groups = o;
    const d = Zt(t, e, n, r, i, a, o, 3);
    return d.requires_grad = u, d.grad_fn = u ? this : null, d;
  }
  _backward(t) {
    const e = this.saved_tensors[0], n = this.saved_tensors[1], r = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [i, a, o] = this.next_functions, [u, d, h] = te(
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
    e.requires_grad && i.backward(u), n.requires_grad && a.backward(d), r && r.requires_grad && o.backward(h);
  }
};
c(ge, "Conv3dOp");
let St = ge;
$("conv3d", St);
const Rn = B(
  (s, t, e, n) => s[e] < t[n] ? 1 : 0,
  (s, t, e, n) => {
  },
  "lt"
), Fn = B(
  (s, t, e, n) => s[e] > t[n] ? 1 : 0,
  (s, t, e, n) => {
  },
  "gt"
), Mn = B(
  (s, t, e, n) => s[e] <= t[n] ? 1 : 0,
  (s, t, e, n) => {
  },
  "le"
), Bn = B(
  (s, t, e, n) => s[e] >= t[n] ? 1 : 0,
  (s, t, e, n) => {
  },
  "ge"
), Tn = B(
  (s, t, e, n) => s[e] == t[n] ? 1 : 0,
  (s, t, e, n) => {
  },
  "eq"
), In = B(
  (s, t, e, n) => s[e] != t[n] ? 1 : 0,
  (s, t, e, n) => {
  },
  "ne"
), Dn = N(
  (s, t) => Math.max(s[t], 0),
  (s, t, e) => {
    t.backward(e.mul(e.mul(s.gt(0))));
  },
  "relu"
), Un = N(
  (s, t) => 1 / (1 + Math.exp(-s[t])),
  (s, t, e) => {
    const n = s.sigmoid();
    t.backward(n.mul(n.mul(-1).add(1)).mul(e));
  },
  "sigmoid"
), Ot = class Ot extends l {
  constructor(t, e = {
    requires_grad: !0
  }, n = {}) {
    t instanceof l ? super(t.data, { requires_grad: !0 }, { shape: t.shape }) : t instanceof Ot ? super(t.data, { requires_grad: !0 }, { shape: t.shape }) : super(t, e, n);
  }
};
c(Ot, "Parameter");
let et = Ot;
const me = class me {
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
    e instanceof et ? this.register_parameter(t, e) : this.register_module(t, e);
  }
  parameters() {
    let t = Object.values(this._parameters);
    for (const e of Object.values(this._modules))
      t = t.concat(e.parameters());
    return t;
  }
};
c(me, "Module");
let st = me;
const we = class we extends st {
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
};
c(we, "Sequential");
let zt = we;
const xe = class xe {
};
c(xe, "Loss");
let lt = xe;
const be = class be extends lt {
  constructor() {
    super();
  }
  forward(t, e) {
    return t.sub(e).pow(2).mean();
  }
};
c(be, "MSELoss");
let Ct = be;
const ye = class ye extends lt {
  constructor() {
    super();
  }
  forward(t, e) {
    return t.sub(e).abs().mean();
  }
};
c(ye, "L1Loss");
let jt = ye;
const ve = class ve extends lt {
  weight;
  constructor(t = null) {
    super(), this.weight = t;
  }
  forward(t, e) {
    const n = e.mul(t.log()), r = e.neg().add(1).mul(t.neg().add(1).log()), i = n.add(r).neg().mean();
    return this.weight ? i.mul(this.weight) : i;
  }
};
c(ve, "BCELoss");
let Kt = ve;
function ee(s) {
  return (...t) => new (J(s))().forward(...t);
}
c(ee, "generate_function");
function Ke(s) {
  return (t) => (typeof t == "number" && (t = new l(t)), new (J(s))().forward(t));
}
c(Ke, "generate_unary_function");
const Le = Ke("relu"), Ge = Ke("sigmoid"), Ve = ee("conv1d"), He = ee("conv2d"), Je = ee("conv3d"), ds = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  conv1d: Ve,
  conv2d: He,
  conv3d: Je,
  relu: Le,
  sigmoid: Ge
}, Symbol.toStringTag, { value: "Module" })), Ae = class Ae extends st {
  weight;
  bias;
  constructor(t, e) {
    super();
    const n = Math.sqrt(1 / t);
    this.weight = new et(
      bt([e, t]).mul(2 * n).sub(n)
    ), this.bias = new et(
      bt([e]).mul(2 * n).sub(n)
    ), this.register("weight", this.weight), this.register("bias", this.bias);
  }
  forward(t) {
    return t.matmul(this.weight.transpose(0, 1)).add(this.bias);
  }
};
c(Ae, "Linear");
let Lt = Ae;
const ke = class ke extends st {
  constructor() {
    super();
  }
  forward(t) {
    return Le(t);
  }
};
c(ke, "ReLU");
let Gt = ke;
const Oe = class Oe extends st {
  constructor() {
    super();
  }
  forward(t) {
    return Ge(t);
  }
};
c(Oe, "Sigmoid");
let Vt = Oe;
const qe = class qe extends st {
  weight;
  bias;
  in_channels;
  out_channels;
  kernel_size;
  stride;
  padding;
  dilation;
  groups;
  constructor(t, e, n, r, i, a, o, u, d) {
    if (super(), this.in_channels = t, this.out_channels = e, this.kernel_size = n, this.stride = r, this.padding = i, this.dilation = a, this.groups = o, t % o !== 0)
      throw new Error("in_channels must be divisible by groups");
    if (e % o !== 0)
      throw new Error("out_channels must be divisible by groups");
    const h = typeof n == "number" ? new Array(d).fill(n) : n, f = h.reduce((p, F) => p * F, 1), _ = Math.sqrt(o / (t * f));
    this.weight = new et(
      bt([e, t / o, ...h]).mul(2 * _).sub(_)
    ), this.register("weight", this.weight), u ? (this.bias = new et(
      bt([e]).mul(2 * _).sub(_)
    ), this.register("bias", this.bias)) : this.bias = null;
  }
};
c(qe, "_ConvNd");
let _t = qe;
const Ee = class Ee extends _t {
  constructor(t, e, n, r = 1, i = 0, a = 1, o = 1, u = !0) {
    super(t, e, n, r, i, a, o, u, 1);
  }
  forward(t) {
    return Ve(t, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
};
c(Ee, "Conv1d");
let Ht = Ee;
const Re = class Re extends _t {
  constructor(t, e, n, r = 1, i = 0, a = 1, o = 1, u = !0) {
    super(t, e, n, r, i, a, o, u, 2);
  }
  forward(t) {
    return He(t, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
};
c(Re, "Conv2d");
let Jt = Re;
const Fe = class Fe extends _t {
  constructor(t, e, n, r = 1, i = 0, a = 1, o = 1, u = !0) {
    super(t, e, n, r, i, a, o, u, 3);
  }
  forward(t) {
    return Je(t, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
};
c(Fe, "Conv3d");
let Qt = Fe;
const Wn = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  BCELoss: Kt,
  Conv1d: Ht,
  Conv2d: Jt,
  Conv3d: Qt,
  L1Loss: jt,
  Linear: Lt,
  MSELoss: Ct,
  Module: st,
  Parameter: et,
  ReLU: Gt,
  Sequential: zt,
  Sigmoid: Vt,
  functional: ds
}, Symbol.toStringTag, { value: "Module" })), Me = class Me {
  params;
  defaults;
  constructor(t, e) {
    this.params = t, this.defaults = e;
  }
  zero_grad() {
    for (const t of this.params)
      t.grad = null;
  }
};
c(Me, "Optimizer");
let ft = Me;
const Be = class Be extends ft {
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
};
c(Be, "SGD");
let Xt = Be;
const Te = class Te extends ft {
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
        m: xt(t),
        v: xt(t),
        vmax: xt(t)
      });
      const n = this.state.get(t);
      n.m = n.m.mul(this.beta1).add(e.mul(1 - this.beta1)), n.v = n.v.mul(this.beta2).add(e.mul(e).mul(1 - this.beta2));
      const r = 1 - Math.pow(this.beta1, this.step_count), i = 1 - Math.pow(this.beta2, this.step_count);
      let a;
      const o = n.m.div(r);
      this.amsgrad ? (n.vmax = n.vmax.maximum(n.v), a = n.vmax.div(i)) : a = n.v.div(i);
      const u = o.div(a.sqrt().add(this.eps)).mul(this.lr), d = t.sub(u);
      t.data = d.data;
    }
  }
};
c(Te, "Adam");
let Yt = Te;
const Nn = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Adam: Yt,
  Optimizer: ft,
  SGD: Xt
}, Symbol.toStringTag, { value: "Module" }));
export {
  kt as AccumulateGrad,
  qn as Max,
  On as Mean,
  En as Min,
  kn as Sum,
  l as Tensor,
  D as TorchFunction,
  ws as __left_index__,
  xs as __right_index__,
  Ts as abs,
  bs as add,
  gs as arange,
  zs as cos,
  ts as disable_no_grad,
  As as div,
  Ze as enable_no_grad,
  Zs as eq,
  Z as eventBus,
  tt as events,
  Ms as exp,
  $s as expand,
  Os as fmod,
  Ys as ge,
  Qs as gt,
  Ye as is_grad_enabled,
  Xs as le,
  ps as linspace,
  Rs as log,
  Js as lt,
  Hs as matmul,
  Gs as max,
  qs as maximum,
  Ks as mean,
  Ls as min,
  Es as minimum,
  vs as mul,
  Us as nan_to_num,
  tn as ne,
  Is as neg,
  Wn as nn,
  ms as no_grad,
  Se as ones,
  fs as ones_like,
  Nn as optim,
  ks as pow,
  bt as rand,
  _s as randint,
  ls as randn,
  Ds as reciprocal,
  Ws as reshape,
  ns as sign,
  Ss as sin,
  Fs as sqrt,
  Bs as square,
  Ns as squeeze,
  ys as sub,
  js as sum,
  Cs as tan,
  Vs as transpose,
  Ps as unsqueeze,
  Xe as zeros,
  xt as zeros_like
};
//# sourceMappingURL=torch.browser.es.js.map
