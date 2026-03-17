var es = Object.defineProperty;
var u = (n, t) => es(n, "name", { value: t, configurable: !0 });
function je(n, t) {
  const e = Math.max(n.length, t.length), s = [...Array(e - n.length).fill(1), ...n], r = [...Array(e - t.length).fill(1), ...t], a = [];
  for (let o = 0; o < e; o++) {
    if (s[o] !== r[o] && s[o] !== 1 && r[o] !== 1)
      throw new Error(`Shape mismatch: ${n} and ${t}`);
    a.push(Math.max(s[o], r[o]));
  }
  return a;
}
u(je, "_broadcast_shape");
function Ce(n, t, e) {
  const s = dt(t, n), r = new Array(t.reduce((a, o) => a * o, 1)).fill(0);
  for (let a = 0; a < e.length; a++)
    r[lt(s, n, a)] += e[a];
  return r;
}
u(Ce, "_unbroadcast");
function dt(n, t) {
  return n.length >= t.length ? n : [...Array(t.length - n.length).fill(1), ...n];
}
u(dt, "_pad_shape");
function lt(n, t, e) {
  let s = 0, r = 1, a = e;
  for (let o = n.length - 1; o >= 0; o--) {
    if (n[o] > 1) {
      const i = a % t[o];
      s = s + i * r;
    }
    r *= n[o], a = Math.floor(a / t[o]);
  }
  return s;
}
u(lt, "_get_original_index");
function qt(n) {
  return Array.isArray(n[0]) ? n[0] : n;
}
u(qt, "get_shape_from_args");
function bs(...n) {
  const t = qt(n), e = new h(Array(t.reduce((s, r) => s * r, 1)).fill(Math.random()));
  return e.shape = t, e;
}
u(bs, "randn");
function bt(...n) {
  const t = qt(n), e = new h(Array(t.reduce((s, r) => s * r, 1)).fill(Math.random()));
  return e.shape = t, e;
}
u(bt, "rand");
function vs(n, t, e) {
  const s = new h(
    Array(e.reduce((r, a) => r * a, 1)).fill(Math.floor(Math.random() * (t - n) + n))
  );
  return s.shape = e, s;
}
u(vs, "randint");
function ys(n, t = !1) {
  return new h(n, { requires_grad: t });
}
u(ys, "tensor");
function Le(...n) {
  const t = qt(n), e = new h(Array(t.reduce((s, r) => s * r, 1)).fill(1));
  return e.shape = t, e;
}
u(Le, "ones");
function ss(...n) {
  const t = qt(n), e = new h(Array(t.reduce((s, r) => s * r, 1)).fill(0));
  return e.shape = t, e;
}
u(ss, "zeros");
function As(n) {
  return Le(n.shape);
}
u(As, "ones_like");
function xt(n) {
  return ss(n.shape);
}
u(xt, "zeros_like");
function ks(n, t, e) {
  const s = [], r = (t - n) / (e - 1);
  for (let a = 0; a < e - 1; a++)
    s.push(n + a * r);
  return s.push(t), new h(s);
}
u(ks, "linspace");
function Os(n, t = void 0, e = 1) {
  const s = [];
  for (let r = n; r < t; r += e)
    s.push(r);
  return new h(s);
}
u(Os, "arange");
let vt = !0;
function ns() {
  return vt;
}
u(ns, "is_grad_enabled");
function rs() {
  const n = vt;
  return vt = !1, n;
}
u(rs, "enable_no_grad");
function as(n) {
  vt = n;
}
u(as, "disable_no_grad");
function os(n) {
  const t = rs();
  try {
    return n();
  } finally {
    as(t);
  }
}
u(os, "no_grad");
let is = 0;
const Ke = /* @__PURE__ */ u(() => is++, "getNextId"), J = new EventTarget(), H = {
  TENSOR_BEFORE_BACKWARD: "tensor.beforeBackward",
  TENSOR_AFTER_BACKWARD: "tensor.afterBackward",
  OPERATION_BEFORE_FORWARD: "operation.beforeForward",
  OPERATION_AFTER_FORWARD: "operation.afterForward",
  OPERATION_BEFORE_BACKWARD: "operation.beforeBackward",
  OPERATION_AFTER_BACKWARD: "operation.afterBackward",
  OPERATION_BEFORE_ACCUMULATE_GRAD: "operation.beforeAccumulateGrad",
  OPERATION_AFTER_ACCUMULATE_GRAD: "operation.afterAccumulateGrad"
};
function A(...n) {
  if (!ns()) return !1;
  for (const t of n)
    if (t instanceof h && t.requires_grad)
      return !0;
  return !1;
}
u(A, "resultRequiresGrad");
const ne = class ne {
  id = Ke();
  opName = "";
  next_functions = [];
  saved_tensors = [];
  _retained_tensors = [];
  forward(...t) {
    const e = A(...t);
    J.dispatchEvent(new CustomEvent(H.OPERATION_BEFORE_FORWARD, {
      detail: {
        operation: this,
        requires_grad: e,
        args: t
      }
    }));
    const s = this._forward(...t);
    return J.dispatchEvent(new CustomEvent(H.OPERATION_AFTER_FORWARD, {
      detail: {
        operation: this,
        requires_grad: e,
        args: t,
        result: s
      }
    })), s;
  }
  backward(t) {
    J.dispatchEvent(new CustomEvent(H.OPERATION_BEFORE_BACKWARD, { detail: { operation: this, dz: t } }));
    for (const e of this._retained_tensors)
      e.grad || (e.grad = new h(new Array(e.dataLength()).fill(0))), e.grad = e.grad.add(t);
    this._backward(t), J.dispatchEvent(new CustomEvent(H.OPERATION_AFTER_BACKWARD, { detail: { operation: this, dz: t } }));
  }
};
u(ne, "TorchFunction");
let z = ne;
const re = class re extends z {
  _forward(...t) {
    throw new Error("NullOp should not be called");
  }
  _backward(t) {
  }
};
u(re, "NullOp");
let Rt = re;
const y = new Rt(), ae = class ae extends z {
};
u(ae, "UnaryFunction");
let yt = ae;
const oe = class oe extends z {
};
u(oe, "BinaryFunction");
let At = oe;
const ie = class ie extends yt {
  variable;
  _forward(t) {
    return this.variable = t, t;
  }
  _backward(t) {
    if (this.variable.grad || (this.variable.grad = xt(this.variable)), J.dispatchEvent(new CustomEvent(H.OPERATION_BEFORE_ACCUMULATE_GRAD, { detail: { operation: this, dz: t } })), typeof t == "number")
      this.variable.grad = this.variable.grad.add(t);
    else {
      const e = Ce(t.shape, this.variable.shape, t.data);
      this.variable.grad = this.variable.grad.add(new h(e, {}, { shape: this.variable.shape }));
    }
    J.dispatchEvent(new CustomEvent(H.OPERATION_AFTER_ACCUMULATE_GRAD, { detail: { operation: this, dz: t } }));
  }
};
u(ie, "AccumulateGrad");
let kt = ie;
const Ge = /* @__PURE__ */ new Map(), Pe = /* @__PURE__ */ new Map();
function D(n, t) {
  Ge.set(n, t);
}
u(D, "registerOperation");
function Ve(n) {
  const t = Ge.get(n);
  if (!t)
    throw new Error(`Operation '${n}' is not registered.`);
  return t;
}
u(Ve, "getOperation");
function Se(n) {
  const t = Pe.get(n);
  if (!t) {
    const e = new (Ve(n))();
    return e.opName = n, Pe.set(n, e), e;
  }
  return t;
}
u(Se, "getOperationCache");
function tt(n) {
  const t = new (Ve(n))();
  return t.opName = n, t;
}
u(tt, "createOperation");
function us(n) {
  if (ArrayBuffer.isView(n))
    return [n.length];
  const t = [];
  for (; Array.isArray(n); )
    t.push(n.length), n = n[0];
  return t;
}
u(us, "_get_shape");
function Je(n) {
  return Array.isArray(n) ? n.flatMap((t) => Je(t)) : ArrayBuffer.isView(n) ? Array.from(n) : [n];
}
u(Je, "_flatten");
const at = class at {
  // Auto-generated ID
  id = Ke();
  // Optional user-defined name
  name = null;
  data;
  shape;
  grad_fn = null;
  grad = null;
  requires_grad;
  constructor(t, e = {}, s = {}) {
    if (this.data = Je(t), this.requires_grad = e.requires_grad ?? !1, e.name && (this.name = e.name), this.shape = s.shape ?? us(t), this.grad_fn = s.operation ?? null, this.requires_grad && !this.grad_fn) {
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
    const e = this.data, s = /* @__PURE__ */ u((r) => {
      const a = this.shape[r], o = new Array(a), i = r === this.shape.length - 1;
      for (let c = 0; c < a; c++)
        i ? o[c] = e[t++] : o[c] = s(r + 1);
      return o;
    }, "buildDimension");
    return s(0);
  }
  dataLength() {
    return this.data.length;
  }
  _executeUnaryOp(t) {
    return (A(this) ? tt(t) : Se(t)).forward(this);
  }
  _executeBinaryOp(t, e) {
    return typeof e == "number" && (e = new at(e)), (A(this, e) ? tt(t) : Se(t)).forward(this, e);
  }
  _executeOpRaw(t, ...e) {
    return tt(t).forward(this, ...e);
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
      this.grad_fn && (J.dispatchEvent(new CustomEvent(H.TENSOR_BEFORE_BACKWARD, { detail: { tensor: this } })), this.grad_fn.backward(t), J.dispatchEvent(new CustomEvent(H.TENSOR_AFTER_BACKWARD, { detail: { tensor: this } })));
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
  allclose(t, e = 1e-5, s = 1e-8, r = !1) {
    if (this.data.length !== t.data.length) return !1;
    for (let a = 0; a < this.data.length; a++) {
      const o = this.data[a], i = t.data[a];
      if (!(r && isNaN(o) && isNaN(i)) && (isNaN(o) || isNaN(i) || Math.abs(o - i) > s + e * Math.abs(i)))
        return !1;
    }
    return !0;
  }
  // other
  sigmoid() {
    return this._executeUnaryOp("sigmoid");
  }
};
u(at, "Tensor");
let h = at;
function X(n) {
  return (...t) => tt(n).forward(...t);
}
u(X, "generate_function$1");
function C(n) {
  return (t) => (typeof t == "number" && (t = new h(t)), tt(n).forward(t));
}
u(C, "generate_unary_function$1");
function B(n) {
  return (t, e) => (typeof t == "number" && (t = new h(t)), typeof e == "number" && (e = new h(e)), tt(n).forward(t, e));
}
u(B, "generate_binary_function");
const qs = B("__left_index__"), Es = B("__right_index__"), Rs = B("add"), Ms = B("sub"), Ts = B("mul"), Fs = B("div"), Bs = B("pow"), Ns = B("fmod"), $s = B("maximum"), Is = B("minimum"), zs = C("log"), Us = C("sqrt"), Ps = C("exp"), Ss = C("square"), Ds = C("abs"), cs = C("sign"), Ws = C("neg"), js = C("reciprocal"), Cs = C("nan_to_num"), Ls = X("reshape"), Ks = X("squeeze"), Gs = X("unsqueeze"), Vs = X("expand"), Js = C("sin"), Hs = C("cos"), Qs = C("tan"), Xs = X("sum"), Ys = X("mean"), Zs = X("min"), tn = X("max"), en = X("transpose"), sn = B("matmul"), nn = B("lt"), rn = B("gt"), an = B("le"), on = B("ge"), un = B("eq"), cn = B("ne");
function dn(n, t, e = 1e-5, s = 1e-8, r = !1) {
  return n.allclose(t, e, s, r);
}
u(dn, "allclose");
function De(n) {
  const t = new Array(n.length).fill(1);
  for (let e = n.length - 2; e >= 0; e--)
    t[e] = t[e + 1] * n[e + 1];
  return t;
}
u(De, "_get_strides");
function ds(n, t) {
  return t.map((e) => {
    const s = Math.floor(n / e);
    return n %= e, s;
  });
}
u(ds, "_unravel_index");
function ls(n, t) {
  return n.reduce((e, s, r) => e + s * t[r], 0);
}
u(ls, "_ravel_index");
function Mt(n, t, e = !1) {
  if (t === void 0) return e ? n.map(() => 1) : [];
  const r = (Array.isArray(t) ? t : [t]).map((a) => a < 0 ? a + n.length : a);
  return e ? n.map((a, o) => r.includes(o) ? 1 : a) : n.filter((a, o) => !r.includes(o));
}
u(Mt, "_get_reduction_shape");
function $(n, t, e = null) {
  const s = /* @__PURE__ */ u((o, i, c, d, l, p) => {
    const f = Array(p);
    for (let _ = 0; _ < p; _++) {
      const w = lt(i, l, _), g = lt(d, l, _);
      f[_] = n(o, c, w, g);
    }
    return f;
  }, "kernel"), r = /* @__PURE__ */ u((o, i, c = null) => {
    const d = je(o.shape, i.shape), l = dt(o.shape, d), p = dt(i.shape, d), f = d.reduce((_, w) => _ * w, 1);
    return new h(
      s(
        o.data,
        l,
        i.data,
        p,
        d,
        f
      ),
      { requires_grad: A(o, i) },
      { operation: c, shape: d }
    );
  }, "forward_tensor"), a = {
    [e]: class extends At {
      _forward(o, i) {
        const c = A(o, i);
        return c && (this.saved_tensors = [o, i]), this.next_functions.push(o.grad_fn ? o.grad_fn : y), this.next_functions.push(i.grad_fn ? i.grad_fn : y), r(o, i, c ? this : null);
      }
      _backward(o) {
        const [i, c] = this.saved_tensors, [d, l] = this.next_functions;
        t(i, c, d, l, o);
      }
    }
  }[e];
  return e && D(e, a), a;
}
u($, "BinaryFunctionMixin");
function P(n, t, e = null) {
  const s = /* @__PURE__ */ u((o, i) => {
    const c = Array(i);
    for (let d = 0; d < i; d++)
      c[d] = n(o, d);
    return c;
  }, "kernel"), r = /* @__PURE__ */ u((o, i = null) => {
    const c = o.dataLength();
    return new h(
      s(o.data, c),
      { requires_grad: A(o) },
      { operation: i, shape: o.shape }
    );
  }, "forward_tensor"), a = {
    [e]: class extends yt {
      _forward(o) {
        const i = A(o);
        return i && (this.saved_tensors = [o]), this.next_functions.push(o.grad_fn ? o.grad_fn : y), r(o, i ? this : null);
      }
      _backward(o) {
        const [i] = this.saved_tensors, [c] = this.next_functions;
        t(i, c, o);
      }
    }
  }[e];
  return e && D(e, a), a;
}
u(P, "UnaryFunctionMixin");
function Et(n, t, e, s = null, r) {
  const a = {
    [s]: class extends z {
      dim;
      keepdim;
      _forward(o, i, c = !1) {
        this.dim = i, this.keepdim = c;
        const d = A(o);
        d && (this.saved_tensors = [o]), this.next_functions.push(o.grad_fn ? o.grad_fn : y);
        const l = Mt(o.shape, i, c), p = l.reduce((m, R) => m * R, 1), f = new Array(p).fill(n), _ = new Array(p).fill(0), w = De(o.shape), g = De(l), N = (i === void 0 ? [] : Array.isArray(i) ? i : [i]).map((m) => m < 0 ? m + o.shape.length : m), W = i === void 0;
        for (let m = 0; m < o.data.length; m++) {
          const R = ds(m, w);
          let q;
          if (W)
            q = c ? R.map(() => 0) : [];
          else {
            q = [];
            for (let M = 0; M < o.shape.length; M++)
              N.includes(M) ? c && q.push(0) : q.push(R[M]);
          }
          const U = ls(q, g);
          f[U] = t(f[U], o.data[m]), _[U]++;
        }
        if (r)
          for (let m = 0; m < p; m++)
            f[m] = r(f[m], _[m]);
        return new h(
          f,
          { requires_grad: d },
          { operation: d ? this : null, shape: l }
        );
      }
      _backward(o) {
        const [i] = this.saved_tensors, [c] = this.next_functions;
        let d = o;
        const l = Mt(i.shape, this.dim, !0);
        o.shape.length !== l.length && (d = o.reshape(l));
        const p = d.expand(i.shape), f = e(i, p, this.dim, this.keepdim);
        c.backward(f);
      }
    }
  }[s];
  return s && D(s, a), a;
}
u(Et, "ReductionFunctionMixin");
function ct(n, t) {
  const e = Ce(n.shape, t, n.data);
  return new h(e, { requires_grad: n.requires_grad }, { shape: t });
}
u(ct, "unbroadcast");
function hs(n, t) {
  return n.mul(Le(t));
}
u(hs, "broadcast");
const ln = $(
  (n, t, e, s) => e,
  () => {
  },
  "__left_index__"
), hn = $(
  (n, t, e, s) => s,
  () => {
  },
  "__right_index__"
), _n = $(
  (n, t, e, s) => n[e] + t[s],
  (n, t, e, s, r) => {
    e.backward(r), s.backward(r);
  },
  "add"
), fn = $(
  (n, t, e, s) => n[e] - t[s],
  (n, t, e, s, r) => {
    e.backward(r), s.backward(r.mul(new h(-1)));
  },
  "sub"
), pn = $(
  (n, t, e, s) => n[e] * t[s],
  (n, t, e, s, r) => {
    e.backward(r.mul(t)), s.backward(r.mul(n));
  },
  "mul"
), gn = $(
  (n, t, e, s) => n[e] / t[s],
  (n, t, e, s, r) => {
    e.backward(r.div(t)), s.backward(r.mul(n).mul(new h(-1)).div(t).div(t));
  },
  "div"
);
function We(n, t, e) {
  const s = typeof e == "number" ? e : null, r = new Array(t.dataLength());
  for (let a = 0; a < r.length; a++)
    r[a] = n.data[a] ? t.data[a] : s !== null ? s : e.data[a];
  return new h(r, {}, { shape: t.shape });
}
u(We, "_where");
const mn = $(
  (n, t, e, s) => Math.pow(n[e], t[s]),
  (n, t, e, s, r) => {
    const a = r.mul(t).mul(n.pow(t.sub(new h(1)))), o = r.mul(n.pow(t)).mul(n.log());
    e.backward(We(n.ne(0), a, a.nan_to_num())), s.backward(We(n.ne(0), o, 0));
  },
  "pow"
), wn = $(
  (n, t, e, s) => n[e] % t[s],
  (n, t, e, s, r) => {
    e.backward(r);
  },
  "fmod"
), xn = $(
  (n, t, e, s) => Math.max(n[e], t[s]),
  (n, t, e, s, r) => {
    const a = n.eq(t), o = n.gt(t).add(a.mul(new h(0.5))), i = t.gt(n).add(a.mul(new h(0.5)));
    e.backward(r.mul(o)), s.backward(r.mul(i));
  },
  "maximum"
), bn = $(
  (n, t, e, s) => Math.min(n[e], t[s]),
  (n, t, e, s, r) => {
    const a = n.eq(t), o = n.lt(t).add(a.mul(new h(0.5))), i = t.lt(n).add(a.mul(new h(0.5)));
    e.backward(r.mul(o)), s.backward(r.mul(i));
  },
  "minimum"
);
function _s(n, t, e = null) {
  const s = new Array(n.dataLength());
  for (let r = 0; r < s.length; r++)
    s[r] = Math.pow(n.data[r], t);
  return new h(
    s,
    { requires_grad: A(n) },
    { operation: e, shape: n.shape }
  );
}
u(_s, "_powint_tensor");
const ue = class ue extends z {
  n;
  _forward(t, e) {
    const s = A(t);
    return s && (this.saved_tensors = [t], this.n = e), this.next_functions.push(t.grad_fn ? t.grad_fn : y), _s(t, e, s ? this : null);
  }
  _backward(t) {
    const [e] = this.saved_tensors, s = this.n, [r] = this.next_functions;
    r.backward(t.mul(s).mul(e.pow(s - 1)));
  }
};
u(ue, "PowInt");
let Tt = ue;
D("powint", Tt);
const vn = P(
  (n, t) => Math.log(n[t]),
  (n, t, e) => {
    t.backward(e.mul(new h(1).div(n)));
  },
  "log"
), yn = P(
  (n, t) => Math.sqrt(n[t]),
  (n, t, e) => {
    t.backward(e.mul(new h(1).div(n.sqrt()).div(2)));
  },
  "sqrt"
), An = P(
  (n, t) => Math.exp(n[t]),
  (n, t, e) => {
    t.backward(e.mul(n.exp()));
  },
  "exp"
), kn = P(
  (n, t) => n[t] * n[t],
  (n, t, e) => {
    t.backward(e.mul(n).mul(new h(2)));
  },
  "square"
), On = P(
  (n, t) => Math.abs(n[t]),
  (n, t, e) => {
    t.backward(e.mul(cs(n)));
  },
  "abs"
), qn = P(
  (n, t) => Math.sign(n[t]),
  (n, t) => {
    t.backward(0);
  },
  "sign"
), En = P(
  (n, t) => -n[t],
  (n, t, e) => {
    t.backward(e.mul(new h(-1)));
  },
  "neg"
), Rn = P(
  (n, t) => 1 / n[t],
  (n, t, e) => {
    t.backward(e.mul(n.pow(-2)).neg());
  },
  "reciprocal"
), Mn = P(
  (n, t) => {
    const e = n[t];
    return Number.isNaN(e) ? 0 : e === 1 / 0 ? 34028235e31 : e === -1 / 0 ? -34028235e31 : e;
  },
  (n, t, e) => {
    t.backward(e);
  },
  "nan_to_num"
), ce = class ce extends z {
  _forward(t, e) {
    const s = t.dataLength(), r = e.reduce((o, i) => o * i, 1);
    if (s !== r)
      throw new Error("Shape mismatch: " + t.shape + " and " + e);
    const a = A(t);
    return a && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(y), new h(
      t.data,
      { requires_grad: a },
      { operation: a ? this : null, shape: e }
    );
  }
  _backward(t) {
    const [e] = this.saved_tensors, [s] = this.next_functions;
    s.backward(t.reshape(e.shape));
  }
};
u(ce, "Reshape");
let Ft = ce;
D("reshape", Ft);
const de = class de extends z {
  _forward(t, e) {
    const s = A(t);
    s && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(y);
    let r = [...t.shape];
    return e !== void 0 ? (e < 0 && (e += t.shape.length), r[e] === 1 && r.splice(e, 1)) : r = r.filter((a) => a !== 1), new h(
      t.data,
      { requires_grad: s },
      { operation: s ? this : null, shape: r }
    );
  }
  _backward(t) {
    const [e] = this.saved_tensors, [s] = this.next_functions;
    s.backward(t.reshape(e.shape));
  }
};
u(de, "Squeeze");
let Bt = de;
D("squeeze", Bt);
const le = class le extends z {
  _forward(t, e) {
    const s = A(t);
    s && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(y), e < 0 && (e += t.shape.length + 1);
    const r = [...t.shape];
    return r.splice(e, 0, 1), new h(
      t.data,
      { requires_grad: s },
      { operation: s ? this : null, shape: r }
    );
  }
  _backward(t) {
    const [e] = this.saved_tensors, [s] = this.next_functions;
    s.backward(t.reshape(e.shape));
  }
};
u(le, "Unsqueeze");
let Nt = le;
D("unsqueeze", Nt);
const he = class he extends z {
  _forward(t, e) {
    const s = A(t);
    s && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(y);
    const r = e.length - t.shape.length, a = e.map((i, c) => {
      if (i === -1) {
        const d = c - r;
        return d >= 0 ? t.shape[d] : 1;
      }
      return i;
    }), o = hs(t, a).data;
    return new h(
      o,
      { requires_grad: s },
      { operation: s ? this : null, shape: a }
    );
  }
  _backward(t) {
    const [e] = this.saved_tensors, [s] = this.next_functions;
    s.backward(ct(t, e.shape));
  }
};
u(he, "Expand");
let $t = he;
D("expand", $t);
const Tn = P(
  (n, t) => Math.sin(n[t]),
  (n, t, e) => {
    t.backward(e.mul(n.cos()));
  },
  "sin"
), Fn = P(
  (n, t) => Math.cos(n[t]),
  (n, t, e) => {
    t.backward(e.mul(n.sin().neg()));
  },
  "cos"
), Bn = P(
  (n, t) => Math.tan(n[t]),
  (n, t, e) => {
    t.backward(e.mul(n.cos().pow(-2)));
  },
  "tan"
), Nn = Et(
  0,
  (n, t) => n + t,
  (n, t) => t,
  "sum"
), $n = Et(
  0,
  (n, t) => n + t,
  (n, t, e) => {
    const s = Mt(n.shape, e, !1), r = s.length > 0 ? s.reduce((o, i) => o * i, 1) : 1, a = n.dataLength() / r;
    return t.mul(new h([1 / a]));
  },
  "mean",
  (n, t) => n / t
), In = Et(
  -1 / 0,
  (n, t) => Math.max(n, t),
  (n, t, e) => {
    const r = n.max(e, !0).expand(n.shape), a = n.eq(r).detach();
    return t.mul(a);
  },
  "max"
), zn = Et(
  1 / 0,
  (n, t) => Math.min(n, t),
  (n, t, e) => {
    const r = n.min(e, !0).expand(n.shape), a = n.eq(r).detach();
    return t.mul(a);
  },
  "min"
);
function fs(n, t, e, s = null) {
  if (n.shape.length + t < 0 || n.shape.length + e < 0)
    throw new Error(`Transpose: Dimension out of range (${t} and ${e})`);
  t = t < 0 ? n.shape.length + t : t, e = e < 0 ? n.shape.length + e : e;
  const r = [...n.shape];
  [r[t], r[e]] = [r[e], r[t]];
  const a = n.dataLength(), o = new Array(a), i = new Array(n.shape.length), c = new Array(r.length);
  for (let d = n.shape.length - 1, l = 1; d >= 0; d--)
    i[d] = l, l *= n.shape[d];
  for (let d = r.length - 1, l = 1; d >= 0; d--)
    c[d] = l, l *= r[d];
  for (let d = 0; d < a; d++) {
    let l = d, p = 0;
    for (let f = 0; f < r.length; f++) {
      const _ = c[f], w = Math.floor(l / _);
      l %= _;
      let g = f;
      f === t ? g = e : f === e && (g = t), p += w * i[g];
    }
    o[d] = n.data[p];
  }
  return new h(
    o,
    { requires_grad: A(n) },
    { operation: s, shape: r }
  );
}
u(fs, "_transpose_tensor");
const _e = class _e extends z {
  dim0;
  dim1;
  _forward(t, e, s) {
    const r = A(t);
    return r && (this.saved_tensors = [t], this.dim0 = e, this.dim1 = s), this.next_functions.push(t.grad_fn ? t.grad_fn : y), fs(t, e, s, r ? this : null);
  }
  _backward(t) {
    const e = this.dim0, s = this.dim1, [r] = this.next_functions;
    r.backward(t.transpose(e, s));
  }
};
u(_e, "Transpose");
let It = _e;
D("transpose", It);
function ps(n, t, e = null) {
  if (n.shape.length == 1 && t.shape.length == 1)
    return [n.mul(t).sum(), []];
  const s = n.shape.length == 1, r = t.shape.length == 1, a = s ? [1, n.shape[0]] : n.shape, o = r ? [t.shape[0], 1] : t.shape;
  if (a[a.length - 1] != o[o.length - 2])
    throw new Error("Shape mismatch: " + n.shape + " and " + t.shape);
  const i = je(a.slice(0, -2), o.slice(0, -2)).concat([
    a[a.length - 2],
    o[o.length - 1]
  ]), c = i.reduce((x, N) => x * N, 1), d = new Array(c).fill(0), l = dt(a, i), p = dt(o, i), f = i[i.length - 2], _ = i[i.length - 1], w = a[a.length - 1];
  for (let x = 0; x < c; x++) {
    const N = x % (f * _), W = Math.floor(N / _), m = N % _, R = lt(l, i, x - m), q = lt(p, i, x - W * _);
    let U = 0;
    for (let M = 0; M < w; M++)
      U += n.data[R + M] * t.data[q + M * _];
    d[x] = U;
  }
  let g = [...i];
  return s && (g = g.slice(0, -2).concat([i[i.length - 1]])), r && (g = g.slice(0, -1)), [new h(
    d,
    { requires_grad: A(n, t) },
    { operation: e, shape: g }
  ), g];
}
u(ps, "_matmul_tensor");
const fe = class fe extends At {
  shape;
  _forward(t, e) {
    const s = A(t, e);
    s && (this.saved_tensors = [t, e]), this.next_functions.push(t.grad_fn ? t.grad_fn : y), this.next_functions.push(e.grad_fn ? e.grad_fn : y);
    const r = ps(t, e, s ? this : null);
    return this.shape = r[1], r[0];
  }
  _backward(t) {
    const [e, s] = this.saved_tensors, [r, a] = this.next_functions;
    if (e.shape.length === 1 && s.shape.length === 1) {
      r.backward(t.mul(s)), a.backward(t.mul(e));
      return;
    }
    if (e.shape.length === 1) {
      const c = t.unsqueeze(-2), d = e.unsqueeze(-2);
      let l = c.matmul(s.transpose(-2, -1)), p = d.transpose(-2, -1).matmul(c);
      l = l.squeeze(-2), p = ct(p, s.shape), r.backward(l), a.backward(p);
      return;
    }
    if (s.shape.length === 1) {
      const c = t.unsqueeze(-1), d = s.unsqueeze(-1);
      let l = c.matmul(d.transpose(-2, -1)), p = e.transpose(-2, -1).matmul(c);
      l = ct(l, e.shape), p = p.squeeze(-1), r.backward(l), a.backward(p);
      return;
    }
    let o = t.matmul(s.transpose(-2, -1)), i = e.transpose(-2, -1).matmul(t);
    o = ct(o, e.shape), i = ct(i, s.shape), r.backward(o), a.backward(i);
  }
};
u(fe, "Matmul");
let zt = fe;
D("matmul", zt);
function te(n, t, e, s, r, a, o, i) {
  const c = typeof s == "number" ? new Array(i).fill(s) : s, d = typeof r == "number" ? new Array(i).fill(r) : r, l = typeof a == "number" ? new Array(i).fill(a) : a, p = n.shape[0], f = n.shape[1], _ = t.shape[0], w = n.shape.slice(2), g = t.shape.slice(2);
  if (f !== t.shape[1] * o)
    throw new Error(`in_channels (${f}) must be divisible by groups (${o}) and match weight.shape[1] * groups (${t.shape[1] * o})`);
  const x = w.map((I, E) => Math.floor((I + 2 * d[E] - l[E] * (g[E] - 1) - 1) / c[E] + 1)), N = [p, _, ...x], W = N.reduce((I, E) => I * E, 1), m = new Array(W).fill(0), R = /* @__PURE__ */ u((I) => {
    const E = new Array(I.length);
    let K = 1;
    for (let L = I.length - 1; L >= 0; L--)
      E[L] = K, K *= I[L];
    return E;
  }, "get_strides"), q = R(n.shape), U = R(t.shape), M = R(N), ot = f / o, it = _ / o;
  for (let I = 0; I < p; I++)
    for (let E = 0; E < o; E++)
      for (let K = 0; K < it; K++) {
        const L = E * it + K, pt = x.reduce((T, F) => T * F, 1);
        for (let T = 0; T < pt; T++) {
          const F = new Array(i);
          let G = T;
          for (let v = i - 1; v >= 0; v--)
            F[v] = G % x[v], G = Math.floor(G / x[v]);
          let V = e ? e.data[L] : 0;
          for (let v = 0; v < ot; v++) {
            const nt = E * ot + v, ut = g.reduce((Y, Z) => Y * Z, 1);
            for (let Y = 0; Y < ut; Y++) {
              const Z = new Array(i);
              let k = Y;
              for (let b = i - 1; b >= 0; b--)
                Z[b] = k % g[b], k = Math.floor(k / g[b]);
              let mt = !0;
              const wt = new Array(i);
              for (let b = 0; b < i; b++) {
                const j = F[b] * c[b] + Z[b] * l[b] - d[b];
                if (j < 0 || j >= w[b]) {
                  mt = !1;
                  break;
                }
                wt[b] = j;
              }
              if (mt) {
                let b = I * q[0] + nt * q[1];
                for (let S = 0; S < i; S++) b += wt[S] * q[S + 2];
                let j = L * U[0] + v * U[1];
                for (let S = 0; S < i; S++) j += Z[S] * U[S + 2];
                V += n.data[b] * t.data[j];
              }
            }
          }
          let gt = I * M[0] + L * M[1];
          for (let v = 0; v < i; v++) gt += F[v] * M[v + 2];
          m[gt] = V;
        }
      }
  return new h(m, { requires_grad: !1 }, { shape: N });
}
u(te, "_convNd_forward");
function ee(n, t, e, s, r, a, o, i, c, d, l) {
  const p = typeof r == "number" ? new Array(c).fill(r) : r, f = typeof a == "number" ? new Array(c).fill(a) : a, _ = typeof o == "number" ? new Array(c).fill(o) : o, w = t.shape[0], g = t.shape[1], x = e.shape[0], N = t.shape.slice(2), W = e.shape.slice(2), m = n.shape.slice(2), R = /* @__PURE__ */ u((T) => {
    const F = new Array(T.length);
    let G = 1;
    for (let V = T.length - 1; V >= 0; V--)
      F[V] = G, G *= T[V];
    return F;
  }, "get_strides"), q = R(t.shape), U = R(e.shape), M = R(n.shape);
  let ot = null, it = null, I = null, E = null, K = null;
  d && (E = new Array(t.dataLength()).fill(0)), l && (K = new Array(e.dataLength()).fill(0));
  const L = g / i, pt = x / i;
  for (let T = 0; T < w; T++)
    for (let F = 0; F < i; F++)
      for (let G = 0; G < pt; G++) {
        const V = F * pt + G, gt = m.reduce((v, nt) => v * nt, 1);
        for (let v = 0; v < gt; v++) {
          const nt = new Array(c);
          let ut = v;
          for (let k = c - 1; k >= 0; k--)
            nt[k] = ut % m[k], ut = Math.floor(ut / m[k]);
          let Y = T * M[0] + V * M[1];
          for (let k = 0; k < c; k++) Y += nt[k] * M[k + 2];
          const Z = n.data[Y];
          for (let k = 0; k < L; k++) {
            const mt = F * L + k, wt = W.reduce((b, j) => b * j, 1);
            for (let b = 0; b < wt; b++) {
              const j = new Array(c);
              let S = b;
              for (let O = c - 1; O >= 0; O--)
                j[O] = S % W[O], S = Math.floor(S / W[O]);
              let ze = !0;
              const Ue = new Array(c);
              for (let O = 0; O < c; O++) {
                const rt = nt[O] * p[O] + j[O] * _[O] - f[O];
                if (rt < 0 || rt >= N[O]) {
                  ze = !1;
                  break;
                }
                Ue[O] = rt;
              }
              if (ze) {
                let O = T * q[0] + mt * q[1];
                for (let Q = 0; Q < c; Q++) O += Ue[Q] * q[Q + 2];
                let rt = V * U[0] + k * U[1];
                for (let Q = 0; Q < c; Q++) rt += j[Q] * U[Q + 2];
                d && (E[O] += Z * e.data[rt]), l && (K[rt] += Z * t.data[O]);
              }
            }
          }
        }
      }
  if (d && (ot = new h(E, { requires_grad: !1 }, { shape: t.shape })), l && (it = new h(K, { requires_grad: !1 }, { shape: e.shape })), s && s.requires_grad) {
    const T = [0];
    for (let F = 2; F < n.shape.length; F++) T.push(F);
    I = n.sum(T);
  }
  return [ot, it, I];
}
u(ee, "_convNd_backward");
const pe = class pe extends z {
  stride;
  padding;
  dilation;
  groups;
  _forward(t, e, s, r = 1, a = 0, o = 1, i = 1) {
    const c = A(t, e, ...s ? [s] : []);
    c && (this.saved_tensors = [t, e], s && this.saved_tensors.push(s)), this.next_functions.push(t.grad_fn ? t.grad_fn : y), this.next_functions.push(e.grad_fn ? e.grad_fn : y), s && this.next_functions.push(s.grad_fn ? s.grad_fn : y), this.stride = r, this.padding = a, this.dilation = o, this.groups = i;
    const d = te(t, e, s, r, a, o, i, 1);
    return d.requires_grad = c, d.grad_fn = c ? this : null, d;
  }
  _backward(t) {
    const e = this.saved_tensors[0], s = this.saved_tensors[1], r = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [a, o, i] = this.next_functions, [c, d, l] = ee(
      t,
      e,
      s,
      r,
      this.stride,
      this.padding,
      this.dilation,
      this.groups,
      1,
      e.requires_grad,
      s.requires_grad
    );
    e.requires_grad && a.backward(c), s.requires_grad && o.backward(d), r && r.requires_grad && i.backward(l);
  }
};
u(pe, "Conv1dOp");
let Ut = pe;
D("conv1d", Ut);
const ge = class ge extends z {
  stride;
  padding;
  dilation;
  groups;
  _forward(t, e, s, r = 1, a = 0, o = 1, i = 1) {
    const c = A(t, e, ...s ? [s] : []);
    c && (this.saved_tensors = [t, e], s && this.saved_tensors.push(s)), this.next_functions.push(t.grad_fn ? t.grad_fn : y), this.next_functions.push(e.grad_fn ? e.grad_fn : y), s && this.next_functions.push(s.grad_fn ? s.grad_fn : y), this.stride = r, this.padding = a, this.dilation = o, this.groups = i;
    const d = te(t, e, s, r, a, o, i, 2);
    return d.requires_grad = c, d.grad_fn = c ? this : null, d;
  }
  _backward(t) {
    const e = this.saved_tensors[0], s = this.saved_tensors[1], r = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [a, o, i] = this.next_functions, [c, d, l] = ee(
      t,
      e,
      s,
      r,
      this.stride,
      this.padding,
      this.dilation,
      this.groups,
      2,
      e.requires_grad,
      s.requires_grad
    );
    e.requires_grad && a.backward(c), s.requires_grad && o.backward(d), r && r.requires_grad && i.backward(l);
  }
};
u(ge, "Conv2dOp");
let Pt = ge;
D("conv2d", Pt);
const me = class me extends z {
  stride;
  padding;
  dilation;
  groups;
  _forward(t, e, s, r = 1, a = 0, o = 1, i = 1) {
    const c = A(t, e, ...s ? [s] : []);
    c && (this.saved_tensors = [t, e], s && this.saved_tensors.push(s)), this.next_functions.push(t.grad_fn ? t.grad_fn : y), this.next_functions.push(e.grad_fn ? e.grad_fn : y), s && this.next_functions.push(s.grad_fn ? s.grad_fn : y), this.stride = r, this.padding = a, this.dilation = o, this.groups = i;
    const d = te(t, e, s, r, a, o, i, 3);
    return d.requires_grad = c, d.grad_fn = c ? this : null, d;
  }
  _backward(t) {
    const e = this.saved_tensors[0], s = this.saved_tensors[1], r = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [a, o, i] = this.next_functions, [c, d, l] = ee(
      t,
      e,
      s,
      r,
      this.stride,
      this.padding,
      this.dilation,
      this.groups,
      3,
      e.requires_grad,
      s.requires_grad
    );
    e.requires_grad && a.backward(c), s.requires_grad && o.backward(d), r && r.requires_grad && i.backward(l);
  }
};
u(me, "Conv3dOp");
let St = me;
D("conv3d", St);
const Un = $(
  (n, t, e, s) => n[e] < t[s] ? 1 : 0,
  () => {
  },
  "lt"
), Pn = $(
  (n, t, e, s) => n[e] > t[s] ? 1 : 0,
  () => {
  },
  "gt"
), Sn = $(
  (n, t, e, s) => n[e] <= t[s] ? 1 : 0,
  () => {
  },
  "le"
), Dn = $(
  (n, t, e, s) => n[e] >= t[s] ? 1 : 0,
  () => {
  },
  "ge"
), Wn = $(
  (n, t, e, s) => n[e] == t[s] ? 1 : 0,
  () => {
  },
  "eq"
), jn = $(
  (n, t, e, s) => n[e] != t[s] ? 1 : 0,
  () => {
  },
  "ne"
), Cn = P(
  (n, t) => Math.max(n[t], 0),
  (n, t, e) => {
    t.backward(e.mul(n.gt(0)));
  },
  "relu"
), Ln = P(
  (n, t) => 1 / (1 + Math.exp(-n[t])),
  (n, t, e) => {
    const s = n.sigmoid();
    t.backward(s.mul(s.mul(-1).add(1)).mul(e));
  },
  "sigmoid"
), Ot = class Ot extends h {
  constructor(t, e = {
    requires_grad: !0
  }, s = {}) {
    t instanceof h ? super(t.data, { requires_grad: !0 }, { shape: t.shape }) : t instanceof Ot ? super(t.data, { requires_grad: !0 }, { shape: t.shape }) : super(t, e, s);
  }
};
u(Ot, "Parameter");
let et = Ot;
const we = class we {
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
  named_parameters(t = "") {
    const e = [];
    for (const [s, r] of Object.entries(this._parameters)) {
      const a = t ? `${t}.${s}` : s;
      e.push([a, r]);
    }
    for (const [s, r] of Object.entries(this._modules)) {
      const a = t ? `${t}.${s}` : s;
      e.push(...r.named_parameters(a));
    }
    return e;
  }
};
u(we, "Module");
let st = we;
const xe = class xe extends st {
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
    for (let s = t; s < this._modulesArr.length; s++)
      this.register(s.toString(), this._modulesArr[s]);
    return this;
  }
  forward(t) {
    let e = t;
    for (const s of this._modulesArr)
      e = s.forward(e);
    return e;
  }
};
u(xe, "Sequential");
let Dt = xe;
const be = class be {
};
u(be, "Loss");
let ht = be;
const ve = class ve extends ht {
  constructor() {
    super();
  }
  forward(t, e) {
    return t.sub(e).pow(2).mean();
  }
};
u(ve, "MSELoss");
let Wt = ve;
const ye = class ye extends ht {
  constructor() {
    super();
  }
  forward(t, e) {
    return t.sub(e).abs().mean();
  }
};
u(ye, "L1Loss");
let jt = ye;
const Ae = class Ae extends ht {
  weight;
  constructor(t = null) {
    super(), this.weight = t;
  }
  forward(t, e) {
    const s = e.mul(t.log()), r = e.neg().add(1).mul(t.neg().add(1).log()), a = s.add(r).neg().mean();
    return this.weight ? a.mul(this.weight) : a;
  }
};
u(Ae, "BCELoss");
let Ct = Ae;
function se(n) {
  return (...t) => tt(n).forward(...t);
}
u(se, "generate_function");
function He(n) {
  return (t) => (typeof t == "number" && (t = new h(t)), tt(n).forward(t));
}
u(He, "generate_unary_function");
const Qe = He("relu"), Xe = He("sigmoid"), Ye = se("conv1d"), Ze = se("conv2d"), ts = se("conv3d"), gs = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  conv1d: Ye,
  conv2d: Ze,
  conv3d: ts,
  relu: Qe,
  sigmoid: Xe
}, Symbol.toStringTag, { value: "Module" })), ke = class ke extends st {
  weight;
  bias;
  constructor(t, e) {
    super();
    const s = Math.sqrt(1 / t);
    this.weight = new et(
      bt([e, t]).mul(2 * s).sub(s)
    ), this.bias = new et(
      bt([e]).mul(2 * s).sub(s)
    ), this.register("weight", this.weight), this.register("bias", this.bias);
  }
  forward(t) {
    return t.matmul(this.weight.transpose(0, 1)).add(this.bias);
  }
};
u(ke, "Linear");
let Lt = ke;
const Oe = class Oe extends st {
  constructor() {
    super();
  }
  forward(t) {
    return Qe(t);
  }
};
u(Oe, "ReLU");
let Kt = Oe;
const qe = class qe extends st {
  constructor() {
    super();
  }
  forward(t) {
    return Xe(t);
  }
};
u(qe, "Sigmoid");
let Gt = qe;
const Ee = class Ee extends st {
  weight;
  bias;
  in_channels;
  out_channels;
  kernel_size;
  stride;
  padding;
  dilation;
  groups;
  constructor(t, e, s, r, a, o, i, c, d) {
    if (super(), this.in_channels = t, this.out_channels = e, this.kernel_size = s, this.stride = r, this.padding = a, this.dilation = o, this.groups = i, t % i !== 0)
      throw new Error("in_channels must be divisible by groups");
    if (e % i !== 0)
      throw new Error("out_channels must be divisible by groups");
    const l = typeof s == "number" ? new Array(d).fill(s) : s, p = l.reduce((_, w) => _ * w, 1), f = Math.sqrt(i / (t * p));
    this.weight = new et(
      bt([e, t / i, ...l]).mul(2 * f).sub(f)
    ), this.register("weight", this.weight), c ? (this.bias = new et(
      bt([e]).mul(2 * f).sub(f)
    ), this.register("bias", this.bias)) : this.bias = null;
  }
};
u(Ee, "_ConvNd");
let _t = Ee;
const Re = class Re extends _t {
  constructor(t, e, s, r = 1, a = 0, o = 1, i = 1, c = !0) {
    super(t, e, s, r, a, o, i, c, 1);
  }
  forward(t) {
    return Ye(t, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
};
u(Re, "Conv1d");
let Vt = Re;
const Me = class Me extends _t {
  constructor(t, e, s, r = 1, a = 0, o = 1, i = 1, c = !0) {
    super(t, e, s, r, a, o, i, c, 2);
  }
  forward(t) {
    return Ze(t, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
};
u(Me, "Conv2d");
let Jt = Me;
const Te = class Te extends _t {
  constructor(t, e, s, r = 1, a = 0, o = 1, i = 1, c = !0) {
    super(t, e, s, r, a, o, i, c, 3);
  }
  forward(t) {
    return ts(t, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
};
u(Te, "Conv3d");
let Ht = Te;
const Kn = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  BCELoss: Ct,
  Conv1d: Vt,
  Conv2d: Jt,
  Conv3d: Ht,
  L1Loss: jt,
  Linear: Lt,
  MSELoss: Wt,
  Module: st,
  Parameter: et,
  ReLU: Kt,
  Sequential: Dt,
  Sigmoid: Gt,
  functional: gs
}, Symbol.toStringTag, { value: "Module" })), Fe = class Fe {
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
u(Fe, "Optimizer");
let ft = Fe;
const Be = class Be extends ft {
  state = /* @__PURE__ */ new Map();
  lr;
  momentum;
  dampening;
  weight_decay;
  nesterov;
  maximize;
  constructor(t, e = 1e-3, s = 0, r = 0, a = 0, o = !1, i = !1) {
    super(t, {}), this.lr = e, this.momentum = s, this.dampening = r, this.weight_decay = a, this.nesterov = o, this.maximize = i;
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
      const s = t.sub(e.mul(this.lr));
      t.data = s.data;
    }
  }
};
u(Be, "SGD");
let Qt = Be;
const Ne = class Ne extends ft {
  state = /* @__PURE__ */ new Map();
  step_count = 0;
  lr;
  beta1;
  beta2;
  eps;
  weight_decay;
  amsgrad;
  maximize;
  constructor(t, e = 1e-3, s = [0.9, 0.999], r = 1e-8, a = 0, o = !1, i = !1) {
    super(t, {}), this.lr = e, this.beta1 = s[0], this.beta2 = s[1], this.eps = r, this.weight_decay = a, this.amsgrad = o, this.maximize = i;
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
      const s = this.state.get(t);
      s.m = s.m.mul(this.beta1).add(e.mul(1 - this.beta1)), s.v = s.v.mul(this.beta2).add(e.mul(e).mul(1 - this.beta2));
      const r = 1 - Math.pow(this.beta1, this.step_count), a = 1 - Math.pow(this.beta2, this.step_count);
      let o;
      const i = s.m.div(r);
      this.amsgrad ? (s.vmax = s.vmax.maximum(s.v), o = s.vmax.div(a)) : o = s.v.div(a);
      const c = i.div(o.sqrt().add(this.eps)).mul(this.lr), d = t.sub(c);
      t.data = d.data;
    }
  }
};
u(Ne, "Adam");
let Xt = Ne;
const Gn = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Adam: Xt,
  Optimizer: ft,
  SGD: Qt
}, Symbol.toStringTag, { value: "Module" })), ms = {
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
function ws(n) {
  return ms[n] || `aten.${n}.default`;
}
u(ws, "toAtenTarget");
const $e = class $e {
  counts = /* @__PURE__ */ new Map();
  generate(t) {
    const e = this.counts.get(t) || 0;
    return this.counts.set(t, e + 1), e === 0 ? t : `${t}_${e}`;
  }
};
u($e, "NameGenerator");
let Yt = $e;
const Ie = class Ie {
  constructor(t, e, s) {
    this.graph = t, this.graph_signature = e, this.parameters = s;
  }
  toString() {
    const t = ["ExportedProgram:"], e = this.graph.filter((s) => s.op === "placeholder").map((s) => {
      const r = s.val_shape ? JSON.stringify(s.val_shape) : "?";
      return `${s.name}: "${r}"`;
    }).join(", ");
    t.push("    class GraphModule(torch.nn.Module):"), t.push(`        def forward(self, ${e}):`);
    for (const s of this.graph)
      if (s.op === "call_function") {
        const r = s.args.join(", ");
        t.push(`            ${s.name} = ${s.target}(${r})`);
      } else s.op === "output" && t.push(`            return (${s.args.join(", ")},)`);
    t.push(""), t.push("Graph signature:"), t.push("    # inputs");
    for (const s of this.graph_signature.input_specs) {
      const r = s.target ? ` target='${s.target}'` : "";
      t.push(`    ${s.name}: ${s.kind}${r}`);
    }
    t.push("    # outputs");
    for (const s of this.graph_signature.output_specs)
      t.push(`    ${s.name}: ${s.kind}`);
    return t.join(`
`);
  }
};
u(Ie, "ExportedProgram");
let Zt = Ie;
function Vn(n, t) {
  const e = [], s = new Yt(), r = /* @__PURE__ */ new Map(), a = n.named_parameters(), o = /* @__PURE__ */ new Set(), i = [];
  for (const [_, w] of a) {
    const g = "p_" + _.replace(/\./g, "_"), x = s.generate(g);
    r.set(w.id, x), o.add(w.id), e.push({
      op: "placeholder",
      name: x,
      target: x,
      args: [],
      val_shape: w.shape
    }), i.push({
      kind: "PARAMETER",
      name: x,
      target: _
    });
  }
  for (let _ = 0; _ < t.length; _++) {
    const g = s.generate("input");
    r.set(t[_].id, g), e.push({
      op: "placeholder",
      name: g,
      target: g,
      args: [],
      val_shape: t[_].shape
    }), i.push({
      kind: "USER_INPUT",
      name: g
    });
  }
  const c = /* @__PURE__ */ u((_) => {
    const { operation: w, args: g, result: x } = _.detail, N = w.opName;
    if (!N) return;
    const W = [];
    for (const R of g)
      if (R instanceof h) {
        const q = r.get(R.id);
        q && W.push(q);
      }
    const m = s.generate(N);
    r.set(x.id, m), e.push({
      op: "call_function",
      name: m,
      target: ws(N),
      args: W,
      val_shape: x.shape
    });
  }, "handler");
  J.addEventListener(
    H.OPERATION_AFTER_FORWARD,
    c
  );
  let d;
  try {
    d = os(() => n.forward(...t));
  } finally {
    J.removeEventListener(
      H.OPERATION_AFTER_FORWARD,
      c
    );
  }
  const l = r.get(d.id) || "output";
  e.push({
    op: "output",
    name: "output",
    target: "output",
    args: [l]
  });
  const p = [{
    kind: "USER_OUTPUT",
    name: l
  }], f = /* @__PURE__ */ new Map();
  for (const [_, w] of a)
    f.set(_, {
      data: [...w.data],
      shape: [...w.shape]
    });
  return new Zt(
    e,
    { input_specs: i, output_specs: p },
    f
  );
}
u(Vn, "export_");
export {
  kt as AccumulateGrad,
  Zt as ExportedProgram,
  In as Max,
  $n as Mean,
  zn as Min,
  Nn as Sum,
  h as Tensor,
  z as TorchFunction,
  qs as __left_index__,
  Es as __right_index__,
  Ds as abs,
  Rs as add,
  dn as allclose,
  Os as arange,
  Hs as cos,
  as as disable_no_grad,
  Fs as div,
  rs as enable_no_grad,
  un as eq,
  J as eventBus,
  H as events,
  Ps as exp,
  Vs as expand,
  Vn as export_,
  Ns as fmod,
  on as ge,
  rn as gt,
  ns as is_grad_enabled,
  an as le,
  ks as linspace,
  zs as log,
  nn as lt,
  sn as matmul,
  tn as max,
  $s as maximum,
  Ys as mean,
  Zs as min,
  Is as minimum,
  Ts as mul,
  Cs as nan_to_num,
  cn as ne,
  Ws as neg,
  Kn as nn,
  os as no_grad,
  Le as ones,
  As as ones_like,
  Gn as optim,
  Bs as pow,
  bt as rand,
  vs as randint,
  bs as randn,
  js as reciprocal,
  Ls as reshape,
  cs as sign,
  Js as sin,
  Us as sqrt,
  Ss as square,
  Ks as squeeze,
  Ms as sub,
  Xs as sum,
  Qs as tan,
  ys as tensor,
  en as transpose,
  Gs as unsqueeze,
  ss as zeros,
  xt as zeros_like
};
//# sourceMappingURL=torch.browser.es.js.map
