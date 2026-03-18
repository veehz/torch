var ss = Object.defineProperty;
var c = (s, t) => ss(s, "name", { value: t, configurable: !0 });
function je(s, t) {
  const e = Math.max(s.length, t.length), n = [...Array(e - s.length).fill(1), ...s], r = [...Array(e - t.length).fill(1), ...t], a = [];
  for (let o = 0; o < e; o++) {
    if (n[o] !== r[o] && n[o] !== 1 && r[o] !== 1)
      throw new Error(`Shape mismatch: ${s} and ${t}`);
    a.push(Math.max(n[o], r[o]));
  }
  return a;
}
c(je, "_broadcast_shape");
function Ce(s, t, e) {
  const n = ht(t, s), r = new Array(t.reduce((a, o) => a * o, 1)).fill(0);
  for (let a = 0; a < e.length; a++)
    r[lt(n, s, a)] += e[a];
  return r;
}
c(Ce, "_unbroadcast");
function ht(s, t) {
  return s.length >= t.length ? s : [...Array(t.length - s.length).fill(1), ...s];
}
c(ht, "_pad_shape");
function lt(s, t, e) {
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
c(lt, "_get_original_index");
function qt(s) {
  return Array.isArray(s[0]) ? s[0] : s;
}
c(qt, "get_shape_from_args");
function vs(...s) {
  const t = qt(s), e = new d(Array(t.reduce((n, r) => n * r, 1)).fill(Math.random()));
  return e.shape = t, e;
}
c(vs, "randn");
function bt(...s) {
  const t = qt(s), e = new d(Array(t.reduce((n, r) => n * r, 1)).fill(Math.random()));
  return e.shape = t, e;
}
c(bt, "rand");
function As(s, t, e) {
  const n = new d(
    Array(e.reduce((r, a) => r * a, 1)).fill(Math.floor(Math.random() * (t - s) + s))
  );
  return n.shape = e, n;
}
c(As, "randint");
function ks(s, t = !1) {
  return new d(s, { requires_grad: t });
}
c(ks, "tensor");
function Le(...s) {
  const t = qt(s), e = new d(Array(t.reduce((n, r) => n * r, 1)).fill(1));
  return e.shape = t, e;
}
c(Le, "ones");
function ns(...s) {
  const t = qt(s), e = new d(Array(t.reduce((n, r) => n * r, 1)).fill(0));
  return e.shape = t, e;
}
c(ns, "zeros");
function Os(s) {
  return Le(s.shape);
}
c(Os, "ones_like");
function xt(s) {
  return ns(s.shape);
}
c(xt, "zeros_like");
function qs(s, t, e) {
  const n = [], r = (t - s) / (e - 1);
  for (let a = 0; a < e - 1; a++)
    n.push(s + a * r);
  return n.push(t), new d(n);
}
c(qs, "linspace");
function Es(s, t = void 0, e = 1) {
  const n = [];
  for (let r = s; r < t; r += e)
    n.push(r);
  return new d(n);
}
c(Es, "arange");
let yt = !0;
function rs() {
  return yt;
}
c(rs, "is_grad_enabled");
function as() {
  const s = yt;
  return yt = !1, s;
}
c(as, "enable_no_grad");
function os(s) {
  yt = s;
}
c(os, "disable_no_grad");
function is(s) {
  const t = as();
  try {
    return s();
  } finally {
    os(t);
  }
}
c(is, "no_grad");
let cs = 0;
const Ke = /* @__PURE__ */ c(() => cs++, "getNextId"), J = new EventTarget(), H = {
  TENSOR_BEFORE_BACKWARD: "tensor.beforeBackward",
  TENSOR_AFTER_BACKWARD: "tensor.afterBackward",
  OPERATION_BEFORE_FORWARD: "operation.beforeForward",
  OPERATION_AFTER_FORWARD: "operation.afterForward",
  OPERATION_BEFORE_BACKWARD: "operation.beforeBackward",
  OPERATION_AFTER_BACKWARD: "operation.afterBackward",
  OPERATION_BEFORE_ACCUMULATE_GRAD: "operation.beforeAccumulateGrad",
  OPERATION_AFTER_ACCUMULATE_GRAD: "operation.afterAccumulateGrad"
};
function A(...s) {
  if (!rs()) return !1;
  for (const t of s)
    if (t instanceof d && t.requires_grad)
      return !0;
  return !1;
}
c(A, "resultRequiresGrad");
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
    const n = this._forward(...t);
    return J.dispatchEvent(new CustomEvent(H.OPERATION_AFTER_FORWARD, {
      detail: {
        operation: this,
        requires_grad: e,
        args: t,
        result: n
      }
    })), n;
  }
  backward(t) {
    J.dispatchEvent(new CustomEvent(H.OPERATION_BEFORE_BACKWARD, { detail: { operation: this, dz: t } }));
    for (const e of this._retained_tensors)
      e.grad || (e.grad = new d(new Array(e.dataLength()).fill(0))), e.grad = e.grad.add(t);
    this._backward(t), J.dispatchEvent(new CustomEvent(H.OPERATION_AFTER_BACKWARD, { detail: { operation: this, dz: t } }));
  }
};
c(ne, "TorchFunction");
let z = ne;
const re = class re extends z {
  _forward(...t) {
    throw new Error("NullOp should not be called");
  }
  _backward(t) {
  }
};
c(re, "NullOp");
let Rt = re;
const v = new Rt(), ae = class ae extends z {
};
c(ae, "UnaryFunction");
let vt = ae;
const oe = class oe extends z {
};
c(oe, "BinaryFunction");
let At = oe;
const ie = class ie extends vt {
  variable;
  _forward(t) {
    return this.variable = t, t;
  }
  _backward(t) {
    if (this.variable.grad || (this.variable.grad = xt(this.variable)), J.dispatchEvent(new CustomEvent(H.OPERATION_BEFORE_ACCUMULATE_GRAD, { detail: { operation: this, dz: t } })), typeof t == "number")
      this.variable.grad = this.variable.grad.add(t);
    else {
      const e = Ce(t.shape, this.variable.shape, t.data);
      this.variable.grad = this.variable.grad.add(new d(e, {}, { shape: this.variable.shape }));
    }
    J.dispatchEvent(new CustomEvent(H.OPERATION_AFTER_ACCUMULATE_GRAD, { detail: { operation: this, dz: t } }));
  }
};
c(ie, "AccumulateGrad");
let kt = ie;
const Ge = /* @__PURE__ */ new Map(), De = /* @__PURE__ */ new Map();
function P(s, t) {
  Ge.set(s, t);
}
c(P, "registerOperation");
function Ve(s) {
  const t = Ge.get(s);
  if (!t)
    throw new Error(`Operation '${s}' is not registered.`);
  return t;
}
c(Ve, "getOperation");
function Ue(s) {
  const t = De.get(s);
  if (!t) {
    const e = new (Ve(s))();
    return e.opName = s, De.set(s, e), e;
  }
  return t;
}
c(Ue, "getOperationCache");
function tt(s) {
  const t = new (Ve(s))();
  return t.opName = s, t;
}
c(tt, "createOperation");
function us(s) {
  if (ArrayBuffer.isView(s))
    return [s.length];
  const t = [];
  for (; Array.isArray(s); )
    t.push(s.length), s = s[0];
  return t;
}
c(us, "_get_shape");
function Je(s, t) {
  if (Array.isArray(s)) {
    if (s.length !== t[0])
      throw new Error(`Shape mismatch at dim ${t.length}: expected ${t[0]}, got ${s.length}`);
    for (let e = 0; e < s.length; e++)
      Je(s[e], t.slice(1));
  } else if (ArrayBuffer.isView(s)) {
    if (t.length !== 1)
      throw new Error(`Shape mismatch at dim ${t.length}: expected 1D, got ${t}`);
    if (s.length !== t[0])
      throw new Error(`Shape mismatch at dim ${t.length}: expected ${t[0]}, got ${s.length}`);
  } else if (t.length !== 0)
    throw new Error(`Shape mismatch at dim ${t.length}: expected scalar, got ${s}`);
}
c(Je, "_assert_shape");
function hs(s) {
  const t = us(s);
  return Je(s, t), t;
}
c(hs, "_get_and_assert_shape");
function He(s) {
  return Array.isArray(s) ? s.flatMap((t) => He(t)) : ArrayBuffer.isView(s) ? Array.from(s) : [s];
}
c(He, "_flatten");
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
  constructor(t, e = {}, n = {}) {
    if (this.data = He(t), this.requires_grad = e.requires_grad ?? !1, e.name && (this.name = e.name), this.shape = n.shape ?? hs(t), this.grad_fn = n.operation ?? null, this.requires_grad && !this.grad_fn) {
      const r = new kt();
      r.variable = this, this.grad_fn = r;
    }
  }
  size(t) {
    if (t !== void 0) {
      if (t < 0 && (t += this.shape.length), t < 0 || t >= this.shape.length)
        throw new Error(`Dimension out of range (expected to be in range of [${-this.shape.length}, ${this.shape.length - 1}], but got ${t})`);
      return this.shape[t];
    }
    return this.shape;
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
      const a = this.shape[r], o = new Array(a), i = r === this.shape.length - 1;
      for (let u = 0; u < a; u++)
        i ? o[u] = e[t++] : o[u] = n(r + 1);
      return o;
    }, "buildDimension");
    return n(0);
  }
  toString() {
    let t = "";
    return this.name && (t += `, name="${this.name}"`), this.dataLength() == 0 && this.shape.length > 0 && (t += `, size=(${this.shape.join(", ")})`), this.requires_grad && (t += ", requires_grad=True"), `Tensor(${JSON.stringify(this.toArray())}${t})`;
  }
  dataLength() {
    return this.data.length;
  }
  _executeUnaryOp(t) {
    return (A(this) ? tt(t) : Ue(t)).forward(this);
  }
  _executeBinaryOp(t, e) {
    return typeof e == "number" && (e = new at(e)), (A(this, e) ? tt(t) : Ue(t)).forward(this, e);
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
  allclose(t, e = 1e-5, n = 1e-8, r = !1) {
    if (this.data.length !== t.data.length) return !1;
    for (let a = 0; a < this.data.length; a++) {
      const o = this.data[a], i = t.data[a];
      if (!(r && isNaN(o) && isNaN(i)) && (isNaN(o) || isNaN(i) || Math.abs(o - i) > n + e * Math.abs(i)))
        return !1;
    }
    return !0;
  }
  // other
  sigmoid() {
    return this._executeUnaryOp("sigmoid");
  }
};
c(at, "Tensor");
let d = at;
function X(s) {
  return (...t) => tt(s).forward(...t);
}
c(X, "generate_function$1");
function C(s) {
  return (t) => (typeof t == "number" && (t = new d(t)), tt(s).forward(t));
}
c(C, "generate_unary_function$1");
function F(s) {
  return (t, e) => (typeof t == "number" && (t = new d(t)), typeof e == "number" && (e = new d(e)), tt(s).forward(t, e));
}
c(F, "generate_binary_function");
const Rs = F("__left_index__"), Ms = F("__right_index__"), Ts = F("add"), $s = F("sub"), Fs = F("mul"), Bs = F("div"), Ns = F("pow"), Ss = F("fmod"), zs = F("maximum"), Is = F("minimum"), Ds = C("log"), Us = C("sqrt"), Ps = C("exp"), Ws = C("square"), js = C("abs"), ls = C("sign"), Cs = C("neg"), Ls = C("reciprocal"), Ks = C("nan_to_num"), Gs = X("reshape"), Vs = X("squeeze"), Js = X("unsqueeze"), Hs = X("expand"), Qs = C("sin"), Xs = C("cos"), Ys = C("tan"), Zs = X("sum"), tn = X("mean"), en = X("min"), sn = X("max"), nn = X("transpose"), rn = F("matmul"), an = F("lt"), on = F("gt"), cn = F("le"), un = F("ge"), hn = F("eq"), ln = F("ne");
function dn(s, t, e = 1e-5, n = 1e-8, r = !1) {
  return s.allclose(t, e, n, r);
}
c(dn, "allclose");
function Pe(s) {
  const t = new Array(s.length).fill(1);
  for (let e = s.length - 2; e >= 0; e--)
    t[e] = t[e + 1] * s[e + 1];
  return t;
}
c(Pe, "_get_strides");
function ds(s, t) {
  return t.map((e) => {
    const n = Math.floor(s / e);
    return s %= e, n;
  });
}
c(ds, "_unravel_index");
function _s(s, t) {
  return s.reduce((e, n, r) => e + n * t[r], 0);
}
c(_s, "_ravel_index");
function Mt(s, t, e = !1) {
  if (t === void 0) return e ? s.map(() => 1) : [];
  const r = (Array.isArray(t) ? t : [t]).map((a) => a < 0 ? a + s.length : a);
  return e ? s.map((a, o) => r.includes(o) ? 1 : a) : s.filter((a, o) => !r.includes(o));
}
c(Mt, "_get_reduction_shape");
function N(s, t, e = null) {
  const n = /* @__PURE__ */ c((o, i, u, h, l, p) => {
    const f = Array(p);
    for (let _ = 0; _ < p; _++) {
      const w = lt(i, l, _), g = lt(h, l, _);
      f[_] = s(o, u, w, g);
    }
    return f;
  }, "kernel"), r = /* @__PURE__ */ c((o, i, u = null) => {
    const h = je(o.shape, i.shape), l = ht(o.shape, h), p = ht(i.shape, h), f = h.reduce((_, w) => _ * w, 1);
    return new d(
      n(
        o.data,
        l,
        i.data,
        p,
        h,
        f
      ),
      { requires_grad: A(o, i) },
      { operation: u, shape: h }
    );
  }, "forward_tensor"), a = {
    [e]: class extends At {
      _forward(o, i) {
        const u = A(o, i);
        return u && (this.saved_tensors = [o, i]), this.next_functions.push(o.grad_fn ? o.grad_fn : v), this.next_functions.push(i.grad_fn ? i.grad_fn : v), r(o, i, u ? this : null);
      }
      _backward(o) {
        const [i, u] = this.saved_tensors, [h, l] = this.next_functions;
        t(i, u, h, l, o);
      }
    }
  }[e];
  return e && P(e, a), a;
}
c(N, "BinaryFunctionMixin");
function D(s, t, e = null) {
  const n = /* @__PURE__ */ c((o, i) => {
    const u = Array(i);
    for (let h = 0; h < i; h++)
      u[h] = s(o, h);
    return u;
  }, "kernel"), r = /* @__PURE__ */ c((o, i = null) => {
    const u = o.dataLength();
    return new d(
      n(o.data, u),
      { requires_grad: A(o) },
      { operation: i, shape: o.shape }
    );
  }, "forward_tensor"), a = {
    [e]: class extends vt {
      _forward(o) {
        const i = A(o);
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
c(D, "UnaryFunctionMixin");
function Et(s, t, e, n = null, r) {
  const a = {
    [n]: class extends z {
      dim;
      keepdim;
      _forward(o, i, u = !1) {
        this.dim = i, this.keepdim = u;
        const h = A(o);
        h && (this.saved_tensors = [o]), this.next_functions.push(o.grad_fn ? o.grad_fn : v);
        const l = Mt(o.shape, i, u), p = l.reduce((m, R) => m * R, 1), f = new Array(p).fill(s), _ = new Array(p).fill(0), w = Pe(o.shape), g = Pe(l), B = (i === void 0 ? [] : Array.isArray(i) ? i : [i]).map((m) => m < 0 ? m + o.shape.length : m), W = i === void 0;
        for (let m = 0; m < o.data.length; m++) {
          const R = ds(m, w);
          let q;
          if (W)
            q = u ? R.map(() => 0) : [];
          else {
            q = [];
            for (let M = 0; M < o.shape.length; M++)
              B.includes(M) ? u && q.push(0) : q.push(R[M]);
          }
          const I = _s(q, g);
          f[I] = t(f[I], o.data[m]), _[I]++;
        }
        if (r)
          for (let m = 0; m < p; m++)
            f[m] = r(f[m], _[m]);
        return new d(
          f,
          { requires_grad: h },
          { operation: h ? this : null, shape: l }
        );
      }
      _backward(o) {
        const [i] = this.saved_tensors, [u] = this.next_functions;
        let h = o;
        const l = Mt(i.shape, this.dim, !0);
        o.shape.length !== l.length && (h = o.reshape(l));
        const p = h.expand(i.shape), f = e(i, p, this.dim, this.keepdim);
        u.backward(f);
      }
    }
  }[n];
  return n && P(n, a), a;
}
c(Et, "ReductionFunctionMixin");
function ut(s, t) {
  const e = Ce(s.shape, t, s.data);
  return new d(e, { requires_grad: s.requires_grad }, { shape: t });
}
c(ut, "unbroadcast");
function fs(s, t) {
  return s.mul(Le(t));
}
c(fs, "broadcast");
const _n = N(
  (s, t, e, n) => e,
  () => {
  },
  "__left_index__"
), fn = N(
  (s, t, e, n) => n,
  () => {
  },
  "__right_index__"
), pn = N(
  (s, t, e, n) => s[e] + t[n],
  (s, t, e, n, r) => {
    e.backward(r), n.backward(r);
  },
  "add"
), gn = N(
  (s, t, e, n) => s[e] - t[n],
  (s, t, e, n, r) => {
    e.backward(r), n.backward(r.mul(new d(-1)));
  },
  "sub"
), mn = N(
  (s, t, e, n) => s[e] * t[n],
  (s, t, e, n, r) => {
    e.backward(r.mul(t)), n.backward(r.mul(s));
  },
  "mul"
), wn = N(
  (s, t, e, n) => s[e] / t[n],
  (s, t, e, n, r) => {
    e.backward(r.div(t)), n.backward(r.mul(s).mul(new d(-1)).div(t).div(t));
  },
  "div"
);
function We(s, t, e) {
  const n = typeof e == "number" ? e : null, r = new Array(t.dataLength());
  for (let a = 0; a < r.length; a++)
    r[a] = s.data[a] ? t.data[a] : n !== null ? n : e.data[a];
  return new d(r, {}, { shape: t.shape });
}
c(We, "_where");
const xn = N(
  (s, t, e, n) => Math.pow(s[e], t[n]),
  (s, t, e, n, r) => {
    const a = r.mul(t).mul(s.pow(t.sub(new d(1)))), o = r.mul(s.pow(t)).mul(s.log());
    e.backward(We(s.ne(0), a, a.nan_to_num())), n.backward(We(s.ne(0), o, 0));
  },
  "pow"
), bn = N(
  (s, t, e, n) => s[e] % t[n],
  (s, t, e, n, r) => {
    e.backward(r);
  },
  "fmod"
), yn = N(
  (s, t, e, n) => Math.max(s[e], t[n]),
  (s, t, e, n, r) => {
    const a = s.eq(t), o = s.gt(t).add(a.mul(new d(0.5))), i = t.gt(s).add(a.mul(new d(0.5)));
    e.backward(r.mul(o)), n.backward(r.mul(i));
  },
  "maximum"
), vn = N(
  (s, t, e, n) => Math.min(s[e], t[n]),
  (s, t, e, n, r) => {
    const a = s.eq(t), o = s.lt(t).add(a.mul(new d(0.5))), i = t.lt(s).add(a.mul(new d(0.5)));
    e.backward(r.mul(o)), n.backward(r.mul(i));
  },
  "minimum"
);
function ps(s, t, e = null) {
  const n = new Array(s.dataLength());
  for (let r = 0; r < n.length; r++)
    n[r] = Math.pow(s.data[r], t);
  return new d(
    n,
    { requires_grad: A(s) },
    { operation: e, shape: s.shape }
  );
}
c(ps, "_powint_tensor");
const ce = class ce extends z {
  n;
  _forward(t, e) {
    const n = A(t);
    return n && (this.saved_tensors = [t], this.n = e), this.next_functions.push(t.grad_fn ? t.grad_fn : v), ps(t, e, n ? this : null);
  }
  _backward(t) {
    const [e] = this.saved_tensors, n = this.n, [r] = this.next_functions;
    r.backward(t.mul(n).mul(e.pow(n - 1)));
  }
};
c(ce, "PowInt");
let Tt = ce;
P("powint", Tt);
const An = D(
  (s, t) => Math.log(s[t]),
  (s, t, e) => {
    t.backward(e.mul(new d(1).div(s)));
  },
  "log"
), kn = D(
  (s, t) => Math.sqrt(s[t]),
  (s, t, e) => {
    t.backward(e.mul(new d(1).div(s.sqrt()).div(2)));
  },
  "sqrt"
), On = D(
  (s, t) => Math.exp(s[t]),
  (s, t, e) => {
    t.backward(e.mul(s.exp()));
  },
  "exp"
), qn = D(
  (s, t) => s[t] * s[t],
  (s, t, e) => {
    t.backward(e.mul(s).mul(new d(2)));
  },
  "square"
), En = D(
  (s, t) => Math.abs(s[t]),
  (s, t, e) => {
    t.backward(e.mul(ls(s)));
  },
  "abs"
), Rn = D(
  (s, t) => Math.sign(s[t]),
  (s, t) => {
    t.backward(0);
  },
  "sign"
), Mn = D(
  (s, t) => -s[t],
  (s, t, e) => {
    t.backward(e.mul(new d(-1)));
  },
  "neg"
), Tn = D(
  (s, t) => 1 / s[t],
  (s, t, e) => {
    t.backward(e.mul(s.pow(-2)).neg());
  },
  "reciprocal"
), $n = D(
  (s, t) => {
    const e = s[t];
    return Number.isNaN(e) ? 0 : e === 1 / 0 ? 34028235e31 : e === -1 / 0 ? -34028235e31 : e;
  },
  (s, t, e) => {
    t.backward(e);
  },
  "nan_to_num"
), ue = class ue extends z {
  _forward(t, e) {
    const n = t.dataLength(), r = e.reduce((o, i) => o * i, 1);
    if (n !== r)
      throw new Error("Shape mismatch: " + t.shape + " and " + e);
    const a = A(t);
    return a && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(v), new d(
      t.data,
      { requires_grad: a },
      { operation: a ? this : null, shape: e }
    );
  }
  _backward(t) {
    const [e] = this.saved_tensors, [n] = this.next_functions;
    n.backward(t.reshape(e.shape));
  }
};
c(ue, "Reshape");
let $t = ue;
P("reshape", $t);
const he = class he extends z {
  _forward(t, e) {
    const n = A(t);
    n && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(v);
    let r = [...t.shape];
    return e !== void 0 ? (e < 0 && (e += t.shape.length), r[e] === 1 && r.splice(e, 1)) : r = r.filter((a) => a !== 1), new d(
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
c(he, "Squeeze");
let Ft = he;
P("squeeze", Ft);
const le = class le extends z {
  _forward(t, e) {
    const n = A(t);
    n && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(v), e < 0 && (e += t.shape.length + 1);
    const r = [...t.shape];
    return r.splice(e, 0, 1), new d(
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
c(le, "Unsqueeze");
let Bt = le;
P("unsqueeze", Bt);
const de = class de extends z {
  _forward(t, e) {
    const n = A(t);
    n && (this.saved_tensors = [t]), t.grad_fn ? this.next_functions.push(t.grad_fn) : this.next_functions.push(v);
    const r = e.length - t.shape.length, a = e.map((i, u) => {
      if (i === -1) {
        const h = u - r;
        return h >= 0 ? t.shape[h] : 1;
      }
      return i;
    }), o = fs(t, a).data;
    return new d(
      o,
      { requires_grad: n },
      { operation: n ? this : null, shape: a }
    );
  }
  _backward(t) {
    const [e] = this.saved_tensors, [n] = this.next_functions;
    n.backward(ut(t, e.shape));
  }
};
c(de, "Expand");
let Nt = de;
P("expand", Nt);
const Fn = D(
  (s, t) => Math.sin(s[t]),
  (s, t, e) => {
    t.backward(e.mul(s.cos()));
  },
  "sin"
), Bn = D(
  (s, t) => Math.cos(s[t]),
  (s, t, e) => {
    t.backward(e.mul(s.sin().neg()));
  },
  "cos"
), Nn = D(
  (s, t) => Math.tan(s[t]),
  (s, t, e) => {
    t.backward(e.mul(s.cos().pow(-2)));
  },
  "tan"
), Sn = Et(
  0,
  (s, t) => s + t,
  (s, t) => t,
  "sum"
), zn = Et(
  0,
  (s, t) => s + t,
  (s, t, e) => {
    const n = Mt(s.shape, e, !1), r = n.length > 0 ? n.reduce((o, i) => o * i, 1) : 1, a = s.dataLength() / r;
    return t.mul(new d([1 / a]));
  },
  "mean",
  (s, t) => s / t
), In = Et(
  -1 / 0,
  (s, t) => Math.max(s, t),
  (s, t, e) => {
    const r = s.max(e, !0).expand(s.shape), a = s.eq(r).detach();
    return t.mul(a);
  },
  "max"
), Dn = Et(
  1 / 0,
  (s, t) => Math.min(s, t),
  (s, t, e) => {
    const r = s.min(e, !0).expand(s.shape), a = s.eq(r).detach();
    return t.mul(a);
  },
  "min"
);
function gs(s, t, e, n = null) {
  if (s.shape.length + t < 0 || s.shape.length + e < 0)
    throw new Error(`Transpose: Dimension out of range (${t} and ${e})`);
  t = t < 0 ? s.shape.length + t : t, e = e < 0 ? s.shape.length + e : e;
  const r = [...s.shape];
  [r[t], r[e]] = [r[e], r[t]];
  const a = s.dataLength(), o = new Array(a), i = new Array(s.shape.length), u = new Array(r.length);
  for (let h = s.shape.length - 1, l = 1; h >= 0; h--)
    i[h] = l, l *= s.shape[h];
  for (let h = r.length - 1, l = 1; h >= 0; h--)
    u[h] = l, l *= r[h];
  for (let h = 0; h < a; h++) {
    let l = h, p = 0;
    for (let f = 0; f < r.length; f++) {
      const _ = u[f], w = Math.floor(l / _);
      l %= _;
      let g = f;
      f === t ? g = e : f === e && (g = t), p += w * i[g];
    }
    o[h] = s.data[p];
  }
  return new d(
    o,
    { requires_grad: A(s) },
    { operation: n, shape: r }
  );
}
c(gs, "_transpose_tensor");
const _e = class _e extends z {
  dim0;
  dim1;
  _forward(t, e, n) {
    const r = A(t);
    return r && (this.saved_tensors = [t], this.dim0 = e, this.dim1 = n), this.next_functions.push(t.grad_fn ? t.grad_fn : v), gs(t, e, n, r ? this : null);
  }
  _backward(t) {
    const e = this.dim0, n = this.dim1, [r] = this.next_functions;
    r.backward(t.transpose(e, n));
  }
};
c(_e, "Transpose");
let St = _e;
P("transpose", St);
function ms(s, t, e = null) {
  if (s.shape.length == 1 && t.shape.length == 1)
    return [s.mul(t).sum(), []];
  const n = s.shape.length == 1, r = t.shape.length == 1, a = n ? [1, s.shape[0]] : s.shape, o = r ? [t.shape[0], 1] : t.shape;
  if (a[a.length - 1] != o[o.length - 2])
    throw new Error("Shape mismatch: " + s.shape + " and " + t.shape);
  const i = je(a.slice(0, -2), o.slice(0, -2)).concat([
    a[a.length - 2],
    o[o.length - 1]
  ]), u = i.reduce((x, B) => x * B, 1), h = new Array(u).fill(0), l = ht(a, i), p = ht(o, i), f = i[i.length - 2], _ = i[i.length - 1], w = a[a.length - 1];
  for (let x = 0; x < u; x++) {
    const B = x % (f * _), W = Math.floor(B / _), m = B % _, R = lt(l, i, x - m), q = lt(p, i, x - W * _);
    let I = 0;
    for (let M = 0; M < w; M++)
      I += s.data[R + M] * t.data[q + M * _];
    h[x] = I;
  }
  let g = [...i];
  return n && (g = g.slice(0, -2).concat([i[i.length - 1]])), r && (g = g.slice(0, -1)), [new d(
    h,
    { requires_grad: A(s, t) },
    { operation: e, shape: g }
  ), g];
}
c(ms, "_matmul_tensor");
const fe = class fe extends At {
  shape;
  _forward(t, e) {
    const n = A(t, e);
    n && (this.saved_tensors = [t, e]), this.next_functions.push(t.grad_fn ? t.grad_fn : v), this.next_functions.push(e.grad_fn ? e.grad_fn : v);
    const r = ms(t, e, n ? this : null);
    return this.shape = r[1], r[0];
  }
  _backward(t) {
    const [e, n] = this.saved_tensors, [r, a] = this.next_functions;
    if (e.shape.length === 1 && n.shape.length === 1) {
      r.backward(t.mul(n)), a.backward(t.mul(e));
      return;
    }
    if (e.shape.length === 1) {
      const u = t.unsqueeze(-2), h = e.unsqueeze(-2);
      let l = u.matmul(n.transpose(-2, -1)), p = h.transpose(-2, -1).matmul(u);
      l = l.squeeze(-2), p = ut(p, n.shape), r.backward(l), a.backward(p);
      return;
    }
    if (n.shape.length === 1) {
      const u = t.unsqueeze(-1), h = n.unsqueeze(-1);
      let l = u.matmul(h.transpose(-2, -1)), p = e.transpose(-2, -1).matmul(u);
      l = ut(l, e.shape), p = p.squeeze(-1), r.backward(l), a.backward(p);
      return;
    }
    let o = t.matmul(n.transpose(-2, -1)), i = e.transpose(-2, -1).matmul(t);
    o = ut(o, e.shape), i = ut(i, n.shape), r.backward(o), a.backward(i);
  }
};
c(fe, "Matmul");
let zt = fe;
P("matmul", zt);
function te(s, t, e, n, r, a, o, i) {
  const u = typeof n == "number" ? new Array(i).fill(n) : n, h = typeof r == "number" ? new Array(i).fill(r) : r, l = typeof a == "number" ? new Array(i).fill(a) : a, p = s.shape[0], f = s.shape[1], _ = t.shape[0], w = s.shape.slice(2), g = t.shape.slice(2);
  if (f !== t.shape[1] * o)
    throw new Error(`in_channels (${f}) must be divisible by groups (${o}) and match weight.shape[1] * groups (${t.shape[1] * o})`);
  const x = w.map((S, E) => Math.floor((S + 2 * h[E] - l[E] * (g[E] - 1) - 1) / u[E] + 1)), B = [p, _, ...x], W = B.reduce((S, E) => S * E, 1), m = new Array(W).fill(0), R = /* @__PURE__ */ c((S) => {
    const E = new Array(S.length);
    let K = 1;
    for (let L = S.length - 1; L >= 0; L--)
      E[L] = K, K *= S[L];
    return E;
  }, "get_strides"), q = R(s.shape), I = R(t.shape), M = R(B), ot = f / o, it = _ / o;
  for (let S = 0; S < p; S++)
    for (let E = 0; E < o; E++)
      for (let K = 0; K < it; K++) {
        const L = E * it + K, pt = x.reduce((T, $) => T * $, 1);
        for (let T = 0; T < pt; T++) {
          const $ = new Array(i);
          let G = T;
          for (let y = i - 1; y >= 0; y--)
            $[y] = G % x[y], G = Math.floor(G / x[y]);
          let V = e ? e.data[L] : 0;
          for (let y = 0; y < ot; y++) {
            const nt = E * ot + y, ct = g.reduce((Y, Z) => Y * Z, 1);
            for (let Y = 0; Y < ct; Y++) {
              const Z = new Array(i);
              let k = Y;
              for (let b = i - 1; b >= 0; b--)
                Z[b] = k % g[b], k = Math.floor(k / g[b]);
              let mt = !0;
              const wt = new Array(i);
              for (let b = 0; b < i; b++) {
                const j = $[b] * u[b] + Z[b] * l[b] - h[b];
                if (j < 0 || j >= w[b]) {
                  mt = !1;
                  break;
                }
                wt[b] = j;
              }
              if (mt) {
                let b = S * q[0] + nt * q[1];
                for (let U = 0; U < i; U++) b += wt[U] * q[U + 2];
                let j = L * I[0] + y * I[1];
                for (let U = 0; U < i; U++) j += Z[U] * I[U + 2];
                V += s.data[b] * t.data[j];
              }
            }
          }
          let gt = S * M[0] + L * M[1];
          for (let y = 0; y < i; y++) gt += $[y] * M[y + 2];
          m[gt] = V;
        }
      }
  return new d(m, { requires_grad: !1 }, { shape: B });
}
c(te, "_convNd_forward");
function ee(s, t, e, n, r, a, o, i, u, h, l) {
  const p = typeof r == "number" ? new Array(u).fill(r) : r, f = typeof a == "number" ? new Array(u).fill(a) : a, _ = typeof o == "number" ? new Array(u).fill(o) : o, w = t.shape[0], g = t.shape[1], x = e.shape[0], B = t.shape.slice(2), W = e.shape.slice(2), m = s.shape.slice(2), R = /* @__PURE__ */ c((T) => {
    const $ = new Array(T.length);
    let G = 1;
    for (let V = T.length - 1; V >= 0; V--)
      $[V] = G, G *= T[V];
    return $;
  }, "get_strides"), q = R(t.shape), I = R(e.shape), M = R(s.shape);
  let ot = null, it = null, S = null, E = null, K = null;
  h && (E = new Array(t.dataLength()).fill(0)), l && (K = new Array(e.dataLength()).fill(0));
  const L = g / i, pt = x / i;
  for (let T = 0; T < w; T++)
    for (let $ = 0; $ < i; $++)
      for (let G = 0; G < pt; G++) {
        const V = $ * pt + G, gt = m.reduce((y, nt) => y * nt, 1);
        for (let y = 0; y < gt; y++) {
          const nt = new Array(u);
          let ct = y;
          for (let k = u - 1; k >= 0; k--)
            nt[k] = ct % m[k], ct = Math.floor(ct / m[k]);
          let Y = T * M[0] + V * M[1];
          for (let k = 0; k < u; k++) Y += nt[k] * M[k + 2];
          const Z = s.data[Y];
          for (let k = 0; k < L; k++) {
            const mt = $ * L + k, wt = W.reduce((b, j) => b * j, 1);
            for (let b = 0; b < wt; b++) {
              const j = new Array(u);
              let U = b;
              for (let O = u - 1; O >= 0; O--)
                j[O] = U % W[O], U = Math.floor(U / W[O]);
              let ze = !0;
              const Ie = new Array(u);
              for (let O = 0; O < u; O++) {
                const rt = nt[O] * p[O] + j[O] * _[O] - f[O];
                if (rt < 0 || rt >= B[O]) {
                  ze = !1;
                  break;
                }
                Ie[O] = rt;
              }
              if (ze) {
                let O = T * q[0] + mt * q[1];
                for (let Q = 0; Q < u; Q++) O += Ie[Q] * q[Q + 2];
                let rt = V * I[0] + k * I[1];
                for (let Q = 0; Q < u; Q++) rt += j[Q] * I[Q + 2];
                h && (E[O] += Z * e.data[rt]), l && (K[rt] += Z * t.data[O]);
              }
            }
          }
        }
      }
  if (h && (ot = new d(E, { requires_grad: !1 }, { shape: t.shape })), l && (it = new d(K, { requires_grad: !1 }, { shape: e.shape })), n && n.requires_grad) {
    const T = [0];
    for (let $ = 2; $ < s.shape.length; $++) T.push($);
    S = s.sum(T);
  }
  return [ot, it, S];
}
c(ee, "_convNd_backward");
const pe = class pe extends z {
  stride;
  padding;
  dilation;
  groups;
  _forward(t, e, n, r = 1, a = 0, o = 1, i = 1) {
    const u = A(t, e, ...n ? [n] : []);
    u && (this.saved_tensors = [t, e], n && this.saved_tensors.push(n)), this.next_functions.push(t.grad_fn ? t.grad_fn : v), this.next_functions.push(e.grad_fn ? e.grad_fn : v), n && this.next_functions.push(n.grad_fn ? n.grad_fn : v), this.stride = r, this.padding = a, this.dilation = o, this.groups = i;
    const h = te(t, e, n, r, a, o, i, 1);
    return h.requires_grad = u, h.grad_fn = u ? this : null, h;
  }
  _backward(t) {
    const e = this.saved_tensors[0], n = this.saved_tensors[1], r = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [a, o, i] = this.next_functions, [u, h, l] = ee(
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
    e.requires_grad && a.backward(u), n.requires_grad && o.backward(h), r && r.requires_grad && i.backward(l);
  }
};
c(pe, "Conv1dOp");
let It = pe;
P("conv1d", It);
const ge = class ge extends z {
  stride;
  padding;
  dilation;
  groups;
  _forward(t, e, n, r = 1, a = 0, o = 1, i = 1) {
    const u = A(t, e, ...n ? [n] : []);
    u && (this.saved_tensors = [t, e], n && this.saved_tensors.push(n)), this.next_functions.push(t.grad_fn ? t.grad_fn : v), this.next_functions.push(e.grad_fn ? e.grad_fn : v), n && this.next_functions.push(n.grad_fn ? n.grad_fn : v), this.stride = r, this.padding = a, this.dilation = o, this.groups = i;
    const h = te(t, e, n, r, a, o, i, 2);
    return h.requires_grad = u, h.grad_fn = u ? this : null, h;
  }
  _backward(t) {
    const e = this.saved_tensors[0], n = this.saved_tensors[1], r = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [a, o, i] = this.next_functions, [u, h, l] = ee(
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
    e.requires_grad && a.backward(u), n.requires_grad && o.backward(h), r && r.requires_grad && i.backward(l);
  }
};
c(ge, "Conv2dOp");
let Dt = ge;
P("conv2d", Dt);
const me = class me extends z {
  stride;
  padding;
  dilation;
  groups;
  _forward(t, e, n, r = 1, a = 0, o = 1, i = 1) {
    const u = A(t, e, ...n ? [n] : []);
    u && (this.saved_tensors = [t, e], n && this.saved_tensors.push(n)), this.next_functions.push(t.grad_fn ? t.grad_fn : v), this.next_functions.push(e.grad_fn ? e.grad_fn : v), n && this.next_functions.push(n.grad_fn ? n.grad_fn : v), this.stride = r, this.padding = a, this.dilation = o, this.groups = i;
    const h = te(t, e, n, r, a, o, i, 3);
    return h.requires_grad = u, h.grad_fn = u ? this : null, h;
  }
  _backward(t) {
    const e = this.saved_tensors[0], n = this.saved_tensors[1], r = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [a, o, i] = this.next_functions, [u, h, l] = ee(
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
    e.requires_grad && a.backward(u), n.requires_grad && o.backward(h), r && r.requires_grad && i.backward(l);
  }
};
c(me, "Conv3dOp");
let Ut = me;
P("conv3d", Ut);
const Un = N(
  (s, t, e, n) => s[e] < t[n] ? 1 : 0,
  () => {
  },
  "lt"
), Pn = N(
  (s, t, e, n) => s[e] > t[n] ? 1 : 0,
  () => {
  },
  "gt"
), Wn = N(
  (s, t, e, n) => s[e] <= t[n] ? 1 : 0,
  () => {
  },
  "le"
), jn = N(
  (s, t, e, n) => s[e] >= t[n] ? 1 : 0,
  () => {
  },
  "ge"
), Cn = N(
  (s, t, e, n) => s[e] == t[n] ? 1 : 0,
  () => {
  },
  "eq"
), Ln = N(
  (s, t, e, n) => s[e] != t[n] ? 1 : 0,
  () => {
  },
  "ne"
), Kn = D(
  (s, t) => Math.max(s[t], 0),
  (s, t, e) => {
    t.backward(e.mul(s.gt(0)));
  },
  "relu"
), Gn = D(
  (s, t) => 1 / (1 + Math.exp(-s[t])),
  (s, t, e) => {
    const n = s.sigmoid();
    t.backward(n.mul(n.mul(-1).add(1)).mul(e));
  },
  "sigmoid"
), Ot = class Ot extends d {
  constructor(t, e = {
    requires_grad: !0
  }, n = {}) {
    t instanceof d ? super(t.data, { requires_grad: !0 }, { shape: t.shape }) : t instanceof Ot ? super(t.data, { requires_grad: !0 }, { shape: t.shape }) : super(t, e, n);
  }
};
c(Ot, "Parameter");
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
};
c(we, "Module");
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
c(xe, "Sequential");
let Pt = xe;
const be = class be {
};
c(be, "Loss");
let dt = be;
const ye = class ye extends dt {
  constructor() {
    super();
  }
  forward(t, e) {
    return t.sub(e).pow(2).mean();
  }
};
c(ye, "MSELoss");
let Wt = ye;
const ve = class ve extends dt {
  constructor() {
    super();
  }
  forward(t, e) {
    return t.sub(e).abs().mean();
  }
};
c(ve, "L1Loss");
let jt = ve;
const Ae = class Ae extends dt {
  weight;
  constructor(t = null) {
    super(), this.weight = t;
  }
  forward(t, e) {
    const n = e.mul(t.log()), r = e.neg().add(1).mul(t.neg().add(1).log()), a = n.add(r).neg().mean();
    return this.weight ? a.mul(this.weight) : a;
  }
};
c(Ae, "BCELoss");
let Ct = Ae;
function se(s) {
  return (...t) => tt(s).forward(...t);
}
c(se, "generate_function");
function Qe(s) {
  return (t) => (typeof t == "number" && (t = new d(t)), tt(s).forward(t));
}
c(Qe, "generate_unary_function");
const Xe = Qe("relu"), Ye = Qe("sigmoid"), Ze = se("conv1d"), ts = se("conv2d"), es = se("conv3d"), ws = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  conv1d: Ze,
  conv2d: ts,
  conv3d: es,
  relu: Xe,
  sigmoid: Ye
}, Symbol.toStringTag, { value: "Module" })), ke = class ke extends st {
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
c(ke, "Linear");
let Lt = ke;
const Oe = class Oe extends st {
  constructor() {
    super();
  }
  forward(t) {
    return Xe(t);
  }
};
c(Oe, "ReLU");
let Kt = Oe;
const qe = class qe extends st {
  constructor() {
    super();
  }
  forward(t) {
    return Ye(t);
  }
};
c(qe, "Sigmoid");
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
  constructor(t, e, n, r, a, o, i, u, h) {
    if (super(), this.in_channels = t, this.out_channels = e, this.kernel_size = n, this.stride = r, this.padding = a, this.dilation = o, this.groups = i, t % i !== 0)
      throw new Error("in_channels must be divisible by groups");
    if (e % i !== 0)
      throw new Error("out_channels must be divisible by groups");
    const l = typeof n == "number" ? new Array(h).fill(n) : n, p = l.reduce((_, w) => _ * w, 1), f = Math.sqrt(i / (t * p));
    this.weight = new et(
      bt([e, t / i, ...l]).mul(2 * f).sub(f)
    ), this.register("weight", this.weight), u ? (this.bias = new et(
      bt([e]).mul(2 * f).sub(f)
    ), this.register("bias", this.bias)) : this.bias = null;
  }
};
c(Ee, "_ConvNd");
let _t = Ee;
const Re = class Re extends _t {
  constructor(t, e, n, r = 1, a = 0, o = 1, i = 1, u = !0) {
    super(t, e, n, r, a, o, i, u, 1);
  }
  forward(t) {
    return Ze(t, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
};
c(Re, "Conv1d");
let Vt = Re;
const Me = class Me extends _t {
  constructor(t, e, n, r = 1, a = 0, o = 1, i = 1, u = !0) {
    super(t, e, n, r, a, o, i, u, 2);
  }
  forward(t) {
    return ts(t, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
};
c(Me, "Conv2d");
let Jt = Me;
const Te = class Te extends _t {
  constructor(t, e, n, r = 1, a = 0, o = 1, i = 1, u = !0) {
    super(t, e, n, r, a, o, i, u, 3);
  }
  forward(t) {
    return es(t, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
};
c(Te, "Conv3d");
let Ht = Te;
const Vn = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
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
  Sequential: Pt,
  Sigmoid: Gt,
  functional: ws
}, Symbol.toStringTag, { value: "Module" })), $e = class $e {
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
c($e, "Optimizer");
let ft = $e;
const Fe = class Fe extends ft {
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
};
c(Fe, "SGD");
let Qt = Fe;
const Be = class Be extends ft {
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
        m: xt(t),
        v: xt(t),
        vmax: xt(t)
      });
      const n = this.state.get(t);
      n.m = n.m.mul(this.beta1).add(e.mul(1 - this.beta1)), n.v = n.v.mul(this.beta2).add(e.mul(e).mul(1 - this.beta2));
      const r = 1 - Math.pow(this.beta1, this.step_count), a = 1 - Math.pow(this.beta2, this.step_count);
      let o;
      const i = n.m.div(r);
      this.amsgrad ? (n.vmax = n.vmax.maximum(n.v), o = n.vmax.div(a)) : o = n.v.div(a);
      const u = i.div(o.sqrt().add(this.eps)).mul(this.lr), h = t.sub(u);
      t.data = h.data;
    }
  }
};
c(Be, "Adam");
let Xt = Be;
const Jn = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Adam: Xt,
  Optimizer: ft,
  SGD: Qt
}, Symbol.toStringTag, { value: "Module" })), xs = {
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
function bs(s) {
  return xs[s] || `aten.${s}.default`;
}
c(bs, "toAtenTarget");
const Ne = class Ne {
  counts = /* @__PURE__ */ new Map();
  generate(t) {
    const e = this.counts.get(t) || 0;
    return this.counts.set(t, e + 1), e === 0 ? t : `${t}_${e}`;
  }
};
c(Ne, "NameGenerator");
let Yt = Ne;
const Se = class Se {
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
};
c(Se, "ExportedProgram");
let Zt = Se;
function Hn(s, t) {
  const e = [], n = new Yt(), r = /* @__PURE__ */ new Map(), a = s.named_parameters(), o = /* @__PURE__ */ new Set(), i = [];
  for (const [_, w] of a) {
    const g = "p_" + _.replace(/\./g, "_"), x = n.generate(g);
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
    const g = n.generate("input");
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
  const u = /* @__PURE__ */ c((_) => {
    const { operation: w, args: g, result: x } = _.detail, B = w.opName;
    if (!B) return;
    const W = [];
    for (const R of g)
      if (R instanceof d) {
        const q = r.get(R.id);
        q && W.push(q);
      }
    const m = n.generate(B);
    r.set(x.id, m), e.push({
      op: "call_function",
      name: m,
      target: bs(B),
      args: W,
      val_shape: x.shape
    });
  }, "handler");
  J.addEventListener(
    H.OPERATION_AFTER_FORWARD,
    u
  );
  let h;
  try {
    h = is(() => s.forward(...t));
  } finally {
    J.removeEventListener(
      H.OPERATION_AFTER_FORWARD,
      u
    );
  }
  const l = r.get(h.id) || "output";
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
c(Hn, "export_");
export {
  kt as AccumulateGrad,
  Zt as ExportedProgram,
  In as Max,
  zn as Mean,
  Dn as Min,
  Sn as Sum,
  d as Tensor,
  z as TorchFunction,
  Rs as __left_index__,
  Ms as __right_index__,
  js as abs,
  Ts as add,
  dn as allclose,
  Es as arange,
  Xs as cos,
  os as disable_no_grad,
  Bs as div,
  as as enable_no_grad,
  hn as eq,
  J as eventBus,
  H as events,
  Ps as exp,
  Hs as expand,
  Hn as export_,
  Ss as fmod,
  un as ge,
  on as gt,
  rs as is_grad_enabled,
  cn as le,
  qs as linspace,
  Ds as log,
  an as lt,
  rn as matmul,
  sn as max,
  zs as maximum,
  tn as mean,
  en as min,
  Is as minimum,
  Fs as mul,
  Ks as nan_to_num,
  ln as ne,
  Cs as neg,
  Vn as nn,
  is as no_grad,
  Le as ones,
  Os as ones_like,
  Jn as optim,
  Ns as pow,
  bt as rand,
  As as randint,
  vs as randn,
  Ls as reciprocal,
  Gs as reshape,
  ls as sign,
  Qs as sin,
  Us as sqrt,
  Ws as square,
  Vs as squeeze,
  $s as sub,
  Zs as sum,
  Ys as tan,
  ks as tensor,
  nn as transpose,
  Js as unsqueeze,
  ns as zeros,
  xt as zeros_like
};
//# sourceMappingURL=torch.browser.es.js.map
