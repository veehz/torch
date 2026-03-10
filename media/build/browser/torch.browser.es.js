var Ht = Object.defineProperty;
var c = (s, e) => Ht(s, "name", { value: e, configurable: !0 });
function Wt(s, e) {
  const t = Math.max(s.length, e.length), r = [...Array(t - s.length).fill(1), ...s], n = [...Array(t - e.length).fill(1), ...e], i = [];
  for (let a = 0; a < t; a++) {
    if (r[a] !== n[a] && r[a] !== 1 && n[a] !== 1)
      throw new Error(`Shape mismatch: ${s} and ${e}`);
    i.push(Math.max(r[a], n[a]));
  }
  return i;
}
c(Wt, "_broadcast_shape");
function Nt(s, e, t) {
  const r = ce(e, s), n = new Array(e.reduce((i, a) => i * a, 1)).fill(0);
  for (let i = 0; i < t.length; i++)
    n[de(r, s, i)] += t[i];
  return n;
}
c(Nt, "_unbroadcast");
function ce(s, e) {
  return s.length >= e.length ? s : [...Array(e.length - s.length).fill(1), ...s];
}
c(ce, "_pad_shape");
function de(s, e, t) {
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
c(de, "_get_original_index");
function Ae(s) {
  return Array.isArray(s[0]) ? s[0] : s;
}
c(Ae, "get_shape_from_args");
function us(...s) {
  const e = Ae(s), t = new l(Array(e.reduce((r, n) => r * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
c(us, "randn");
function xe(...s) {
  const e = Ae(s), t = new l(Array(e.reduce((r, n) => r * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
c(xe, "rand");
function cs(s, e, t) {
  const r = new l(
    Array(t.reduce((n, i) => n * i, 1)).fill(Math.floor(Math.random() * (e - s) + s))
  );
  return r.shape = t, r;
}
c(cs, "randint");
function Pt(...s) {
  const e = Ae(s), t = new l(Array(e.reduce((r, n) => r * n, 1)).fill(1));
  return t.shape = e, t;
}
c(Pt, "ones");
function Jt(...s) {
  const e = Ae(s), t = new l(Array(e.reduce((r, n) => r * n, 1)).fill(0));
  return t.shape = e, t;
}
c(Jt, "zeros");
function ds(s) {
  return Pt(s.shape);
}
c(ds, "ones_like");
function we(s) {
  return Jt(s.shape);
}
c(we, "zeros_like");
function hs(s, e, t) {
  const r = [], n = (e - s) / (t - 1);
  for (let i = 0; i < t - 1; i++)
    r.push(s + i * n);
  return r.push(e), new l(r);
}
c(hs, "linspace");
function ls(s, e = void 0, t = 1) {
  const r = [];
  for (let n = s; n < e; n += t)
    r.push(n);
  return new l(r);
}
c(ls, "arange");
let Qt = 0;
const $t = /* @__PURE__ */ c(() => Qt++, "getNextId"), Y = new EventTarget(), Z = {
  TENSOR_BEFORE_BACKWARD: "tensor.beforeBackward",
  TENSOR_AFTER_BACKWARD: "tensor.afterBackward",
  OPERATION_BEFORE_FORWARD: "operation.beforeForward",
  OPERATION_AFTER_FORWARD: "operation.afterForward",
  OPERATION_BEFORE_BACKWARD: "operation.beforeBackward",
  OPERATION_AFTER_BACKWARD: "operation.afterBackward",
  OPERATION_BEFORE_ACCUMULATE_GRAD: "operation.beforeAccumulateGrad",
  OPERATION_AFTER_ACCUMULATE_GRAD: "operation.afterAccumulateGrad"
};
function Xt(...s) {
  for (const e of s)
    if (e instanceof l && e.requires_grad)
      return !0;
  return !1;
}
c(Xt, "resultRequiresGrad");
const et = class et {
  id = $t();
  next_functions = [];
  saved_tensors = [];
  _retained_tensors = [];
  forward(...e) {
    const t = Xt(...e);
    Y.dispatchEvent(new CustomEvent(Z.OPERATION_BEFORE_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e
      }
    }));
    const r = this._forward(...e);
    return Y.dispatchEvent(new CustomEvent(Z.OPERATION_AFTER_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e,
        result: r
      }
    })), r;
  }
  backward(e) {
    Y.dispatchEvent(new CustomEvent(Z.OPERATION_BEFORE_BACKWARD, { detail: { operation: this, dz: e } }));
    for (const t of this._retained_tensors)
      t.grad || (t.grad = new l(new Array(t.dataLength()).fill(0))), t.grad = t.grad.add(e);
    this._backward(e), Y.dispatchEvent(new CustomEvent(Z.OPERATION_AFTER_BACKWARD, { detail: { operation: this, dz: e } }));
  }
};
c(et, "TorchFunction");
let I = et;
const tt = class tt extends I {
  _forward(...e) {
    throw new Error("NullOp should not be called");
  }
  _backward(e) {
  }
};
c(tt, "NullOp");
let Ee = tt;
const x = new Ee(), st = class st extends I {
};
c(st, "UnaryFunction");
let be = st;
const rt = class rt extends I {
};
c(rt, "BinaryFunction");
let qe = rt;
const nt = class nt extends be {
  variable;
  _forward(e) {
    return this.variable = e, e;
  }
  _backward(e) {
    if (this.variable.grad || (this.variable.grad = we(this.variable)), Y.dispatchEvent(new CustomEvent(Z.OPERATION_BEFORE_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } })), typeof e == "number")
      this.variable.grad = this.variable.grad.add(e);
    else {
      const t = Nt(e.shape, this.variable.shape, e.data);
      this.variable.grad = this.variable.grad.add(new l(t, {}, { shape: this.variable.shape }));
    }
    Y.dispatchEvent(new CustomEvent(Z.OPERATION_AFTER_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } }));
  }
};
c(nt, "AccumulateGrad");
let ye = nt;
const St = /* @__PURE__ */ new Map(), Oe = /* @__PURE__ */ new Map();
function P(s, e) {
  St.set(s, e);
}
c(P, "registerOperation");
function H(s) {
  const e = St.get(s);
  if (!e)
    throw new Error(`Operation '${s}' is not registered.`);
  return e;
}
c(H, "getOperation");
function It(s) {
  const e = Oe.get(s);
  return e || (Oe.set(s, new (H(s))()), Oe.get(s));
}
c(It, "getOperationCache");
function Yt(s) {
  if (ArrayBuffer.isView(s))
    return [s.length];
  const e = [];
  for (; Array.isArray(s); )
    e.push(s.length), s = s[0];
  return e;
}
c(Yt, "_get_shape");
function zt(s) {
  return Array.isArray(s) ? s.flatMap((e) => zt(e)) : ArrayBuffer.isView(s) ? Array.from(s) : [s];
}
c(zt, "_flatten");
const ne = class ne {
  // Auto-generated ID
  id = $t();
  // Optional user-defined name
  name = null;
  data;
  shape;
  grad_fn = null;
  grad = null;
  requires_grad;
  constructor(e, t = {}, r = {}) {
    if (this.data = zt(e), this.requires_grad = t.requires_grad ?? !1, t.name && (this.name = t.name), this.shape = r.shape ?? Yt(e), this.grad_fn = r.operation ?? null, this.requires_grad && !this.grad_fn) {
      const n = new ye();
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
    const t = this.data, r = /* @__PURE__ */ c((n) => {
      const i = this.shape[n], a = new Array(i), o = n === this.shape.length - 1;
      for (let u = 0; u < i; u++)
        o ? a[u] = t[e++] : a[u] = r(n + 1);
      return a;
    }, "buildDimension");
    return r(0);
  }
  dataLength() {
    return this.data.length;
  }
  _executeUnaryOp(e) {
    return (this.requires_grad ? new (H(e))() : It(e)).forward(this);
  }
  _executeBinaryOp(e, t) {
    return typeof t == "number" && (t = new ne(t)), (this.requires_grad || t.requires_grad ? new (H(e))() : It(e)).forward(this, t);
  }
  _executeOpRaw(e, ...t) {
    return new (H(e))().forward(this, ...t);
  }
  item() {
    if (this.dataLength() !== 1)
      throw new Error("Tensor.item() is only valid for scalars");
    return this.data[0];
  }
  detach() {
    return new ne(this.data, { requires_grad: !1 }, { shape: this.shape });
  }
  detach_() {
    this.requires_grad = !1, this.grad = null, this.grad_fn = null;
  }
  zero_() {
    this.data = Array(this.dataLength()).fill(0);
  }
  is_retain_grad = !1;
  retain_grad() {
    this.grad_fn instanceof ye || this.is_retain_grad || (this.is_retain_grad = !0, this.grad_fn._retained_tensors.push(this));
  }
  backward(e) {
    if (this.requires_grad) {
      if (e)
        e.toArray_();
      else {
        if (this.dataLength() !== 1)
          throw new Error("Gradient is required for non-scalar tensors");
        e = new ne(1);
      }
      this.grad_fn && (Y.dispatchEvent(new CustomEvent(Z.TENSOR_BEFORE_BACKWARD, { detail: { tensor: this } })), this.grad_fn.backward(e), Y.dispatchEvent(new CustomEvent(Z.TENSOR_AFTER_BACKWARD, { detail: { tensor: this } })));
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
  nan_to_num() {
    return this._executeUnaryOp("nan_to_num");
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
};
c(ne, "Tensor");
let l = ne;
function J(s) {
  return (...e) => new (H(s))().forward(...e);
}
c(J, "generate_function$1");
function C(s) {
  return (e) => (typeof e == "number" && (e = new l(e)), new (H(s))().forward(e));
}
c(C, "generate_unary_function$1");
function E(s) {
  return (e, t) => (typeof e == "number" && (e = new l(e)), typeof t == "number" && (t = new l(t)), new (H(s))().forward(e, t));
}
c(E, "generate_binary_function$1");
const _s = E("__left_index__"), fs = E("__right_index__"), ps = E("add"), gs = E("sub"), ms = E("mul"), ws = E("div"), xs = E("pow"), bs = E("fmod"), qs = E("maximum"), ys = E("minimum"), vs = C("log"), As = C("sqrt"), ks = C("exp"), Os = C("square"), Es = C("abs"), Zt = C("sign"), Rs = C("neg"), Fs = C("reciprocal"), Ms = C("nan_to_num"), Bs = J("reshape"), Ts = J("squeeze"), Is = J("unsqueeze"), Ds = J("expand"), Us = C("sin"), Ws = C("cos"), Ns = C("tan"), Ps = J("sum"), $s = J("mean"), Ss = J("min"), zs = J("max"), Cs = J("transpose"), js = E("matmul"), Ks = E("lt"), Ls = E("gt"), Gs = E("le"), Vs = E("ge"), Hs = E("eq"), Js = E("ne");
function Dt(s) {
  const e = new Array(s.length).fill(1);
  for (let t = s.length - 2; t >= 0; t--)
    e[t] = e[t + 1] * s[t + 1];
  return e;
}
c(Dt, "_get_strides");
function es(s, e) {
  return e.map((t) => {
    const r = Math.floor(s / t);
    return s %= t, r;
  });
}
c(es, "_unravel_index");
function ts(s, e) {
  return s.reduce((t, r, n) => t + r * e[n], 0);
}
c(ts, "_ravel_index");
function Re(s, e, t = !1) {
  if (e === void 0) return t ? s.map(() => 1) : [];
  const n = (Array.isArray(e) ? e : [e]).map((i) => i < 0 ? i + s.length : i);
  return t ? s.map((i, a) => n.includes(a) ? 1 : i) : s.filter((i, a) => !n.includes(a));
}
c(Re, "_get_reduction_shape");
function F(s, e, t = null) {
  const r = /* @__PURE__ */ c((a, o, u, d, h, _) => {
    const f = Array(_);
    for (let p = 0; p < _; p++) {
      const R = de(o, h, p), b = de(d, h, p);
      f[p] = s(a, u, R, b);
    }
    return f;
  }, "kernel"), n = /* @__PURE__ */ c((a, o, u = null) => {
    const d = Wt(a.shape, o.shape), h = ce(a.shape, d), _ = ce(o.shape, d), f = d.reduce((p, R) => p * R, 1);
    return new l(
      r(
        a.data,
        h,
        o.data,
        _,
        d,
        f
      ),
      { requires_grad: a.requires_grad || o.requires_grad },
      { operation: u, shape: d }
    );
  }, "forward_tensor"), i = {
    [t]: class extends qe {
      _forward(a, o) {
        return (a.requires_grad || o.requires_grad) && (this.saved_tensors = [a, o]), this.next_functions.push(a.grad_fn ? a.grad_fn : x), this.next_functions.push(o.grad_fn ? o.grad_fn : x), n(a, o, a.requires_grad || o.requires_grad ? this : null);
      }
      _backward(a) {
        const [o, u] = this.saved_tensors, [d, h] = this.next_functions;
        e(o, u, d, h, a);
      }
    }
  }[t];
  return t && P(t, i), i;
}
c(F, "BinaryFunctionMixin");
function D(s, e, t = null) {
  const r = /* @__PURE__ */ c((a, o) => {
    const u = Array(o);
    for (let d = 0; d < o; d++)
      u[d] = s(a, d);
    return u;
  }, "kernel"), n = /* @__PURE__ */ c((a, o = null) => {
    const u = a.dataLength();
    return new l(
      r(a.data, u),
      { requires_grad: a.requires_grad },
      { operation: o, shape: a.shape }
    );
  }, "forward_tensor"), i = {
    [t]: class extends be {
      _forward(a) {
        return a.requires_grad && (this.saved_tensors = [a]), this.next_functions.push(a.grad_fn ? a.grad_fn : x), n(a, a.requires_grad ? this : null);
      }
      _backward(a) {
        const [o] = this.saved_tensors, [u] = this.next_functions;
        e(o, u, a);
      }
    }
  }[t];
  return t && P(t, i), i;
}
c(D, "UnaryFunctionMixin");
function ke(s, e, t, r = null, n) {
  const i = {
    [r]: class extends I {
      dim;
      keepdim;
      _forward(a, o, u = !1) {
        this.dim = o, this.keepdim = u, a.requires_grad && (this.saved_tensors = [a]), this.next_functions.push(a.grad_fn ? a.grad_fn : x);
        const d = Re(a.shape, o, u), h = d.reduce((g, U) => g * U, 1), _ = new Array(h).fill(s), f = new Array(h).fill(0), p = Dt(a.shape), R = Dt(d), A = (o === void 0 ? [] : Array.isArray(o) ? o : [o]).map((g) => g < 0 ? g + a.shape.length : g), $ = o === void 0;
        for (let g = 0; g < a.data.length; g++) {
          const U = es(g, p);
          let M;
          if ($)
            M = u ? U.map(() => 0) : [];
          else {
            M = [];
            for (let B = 0; B < a.shape.length; B++)
              A.includes(B) ? u && M.push(0) : M.push(U[B]);
          }
          const W = ts(M, R);
          _[W] = e(_[W], a.data[g]), f[W]++;
        }
        if (n)
          for (let g = 0; g < h; g++)
            _[g] = n(_[g], f[g]);
        return new l(
          _,
          { requires_grad: a.requires_grad },
          { operation: a.requires_grad ? this : null, shape: d }
        );
      }
      _backward(a) {
        const [o] = this.saved_tensors, [u] = this.next_functions;
        let d = a;
        const h = Re(o.shape, this.dim, !0);
        a.shape.length !== h.length && (d = a.reshape(h));
        const _ = d.expand(o.shape), f = t(o, _, this.dim, this.keepdim);
        u.backward(f);
      }
    }
  }[r];
  return r && P(r, i), i;
}
c(ke, "ReductionFunctionMixin");
function ue(s, e) {
  const t = Nt(s.shape, e, s.data);
  return new l(t, { requires_grad: s.requires_grad }, { shape: e });
}
c(ue, "unbroadcast");
function ss(s, e) {
  return s.mul(Pt(e));
}
c(ss, "broadcast");
const Qs = F(
  (s, e, t, r) => t,
  (s, e, t, r, n) => {
  },
  "__left_index__"
), Xs = F(
  (s, e, t, r) => r,
  (s, e, t, r, n) => {
  },
  "__right_index__"
), Ys = F(
  (s, e, t, r) => s[t] + e[r],
  (s, e, t, r, n) => {
    t.backward(n), r.backward(n);
  },
  "add"
), Zs = F(
  (s, e, t, r) => s[t] - e[r],
  (s, e, t, r, n) => {
    t.backward(n), r.backward(n.mul(new l(-1)));
  },
  "sub"
), er = F(
  (s, e, t, r) => s[t] * e[r],
  (s, e, t, r, n) => {
    t.backward(n.mul(e)), r.backward(n.mul(s));
  },
  "mul"
), tr = F(
  (s, e, t, r) => s[t] / e[r],
  (s, e, t, r, n) => {
    t.backward(n.div(e)), r.backward(n.mul(s).mul(new l(-1)).div(e).div(e));
  },
  "div"
);
function Ut(s, e, t) {
  const r = typeof t == "number" ? t : null, n = new Array(e.dataLength());
  for (let i = 0; i < n.length; i++)
    n[i] = s.data[i] ? e.data[i] : r !== null ? r : t.data[i];
  return new l(n, {}, { shape: e.shape });
}
c(Ut, "_where");
const sr = F(
  (s, e, t, r) => Math.pow(s[t], e[r]),
  (s, e, t, r, n) => {
    const i = n.mul(e).mul(s.pow(e.sub(new l(1)))), a = n.mul(s.pow(e)).mul(s.log());
    t.backward(Ut(s.ne(0), i, i.nan_to_num())), r.backward(Ut(s.ne(0), a, 0));
  },
  "pow"
), rr = F(
  (s, e, t, r) => s[t] % e[r],
  (s, e, t, r, n) => {
    t.backward(n);
  },
  "fmod"
), nr = F(
  (s, e, t, r) => Math.max(s[t], e[r]),
  (s, e, t, r, n) => {
    const i = s.eq(e), a = s.gt(e).add(i.mul(new l(0.5))), o = e.gt(s).add(i.mul(new l(0.5)));
    t.backward(n.mul(a)), r.backward(n.mul(o));
  },
  "maximum"
), ar = F(
  (s, e, t, r) => Math.min(s[t], e[r]),
  (s, e, t, r, n) => {
    const i = s.eq(e), a = s.lt(e).add(i.mul(new l(0.5))), o = e.lt(s).add(i.mul(new l(0.5)));
    t.backward(n.mul(a)), r.backward(n.mul(o));
  },
  "minimum"
);
function rs(s, e, t = null) {
  const r = new Array(s.dataLength());
  for (let n = 0; n < r.length; n++)
    r[n] = Math.pow(s.data[n], e);
  return new l(
    r,
    { requires_grad: s.requires_grad },
    { operation: t, shape: s.shape }
  );
}
c(rs, "_powint_tensor");
const at = class at extends I {
  n;
  _forward(e, t) {
    return e.requires_grad && (this.saved_tensors = [e], this.n = t), this.next_functions.push(e.grad_fn ? e.grad_fn : x), rs(e, t, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, r = this.n, [n] = this.next_functions;
    n.backward(e.mul(r).mul(t.pow(r - 1)));
  }
};
c(at, "PowInt");
let Fe = at;
P("powint", Fe);
const ir = D(
  (s, e) => Math.log(s[e]),
  (s, e, t) => {
    e.backward(t.mul(new l(1).div(s)));
  },
  "log"
), or = D(
  (s, e) => Math.sqrt(s[e]),
  (s, e, t) => {
    e.backward(t.mul(new l(1).div(s.sqrt()).div(2)));
  },
  "sqrt"
), ur = D(
  (s, e) => Math.exp(s[e]),
  (s, e, t) => {
    e.backward(t.mul(t.mul(s.exp())));
  },
  "exp"
), cr = D(
  (s, e) => s[e] * s[e],
  (s, e, t) => {
    e.backward(t.mul(t.mul(s).mul(new l(2))));
  },
  "square"
), dr = D(
  (s, e) => Math.abs(s[e]),
  (s, e, t) => {
    e.backward(t.mul(t.mul(Zt(s))));
  },
  "abs"
), hr = D(
  (s, e) => Math.sign(s[e]),
  (s, e, t) => {
    e.backward(0);
  },
  "sign"
), lr = D(
  (s, e) => -s[e],
  (s, e, t) => {
    e.backward(t.mul(t.mul(new l(-1))));
  },
  "neg"
), _r = D(
  (s, e) => 1 / s[e],
  (s, e, t) => {
    e.backward(t.mul(t.mul(s.pow(-2))).neg());
  },
  "reciprocal"
), fr = D(
  (s, e) => {
    const t = s[e];
    return Number.isNaN(t) ? 0 : t === 1 / 0 ? 34028235e31 : t === -1 / 0 ? -34028235e31 : t;
  },
  (s, e, t) => {
    e.backward(t);
  },
  "nan_to_num"
), it = class it extends I {
  _forward(e, t) {
    const r = e.dataLength(), n = t.reduce((i, a) => i * a, 1);
    if (r !== n)
      throw new Error("Shape mismatch: " + e.shape + " and " + t);
    return e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(x), new l(
      e.data,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: t }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [r] = this.next_functions;
    r.backward(e.reshape(t.shape));
  }
};
c(it, "Reshape");
let Me = it;
P("reshape", Me);
const ot = class ot extends I {
  _forward(e, t) {
    e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(x);
    let r = [...e.shape];
    return t !== void 0 ? (t < 0 && (t += e.shape.length), r[t] === 1 && r.splice(t, 1)) : r = r.filter((n) => n !== 1), new l(
      e.data,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: r }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [r] = this.next_functions;
    r.backward(e.reshape(t.shape));
  }
};
c(ot, "Squeeze");
let Be = ot;
P("squeeze", Be);
const ut = class ut extends I {
  _forward(e, t) {
    e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(x), t < 0 && (t += e.shape.length + 1);
    const r = [...e.shape];
    return r.splice(t, 0, 1), new l(
      e.data,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: r }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [r] = this.next_functions;
    r.backward(e.reshape(t.shape));
  }
};
c(ut, "Unsqueeze");
let Te = ut;
P("unsqueeze", Te);
const ct = class ct extends I {
  _forward(e, t) {
    e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(x);
    const r = t.length - e.shape.length, n = t.map((a, o) => {
      if (a === -1) {
        const u = o - r;
        return u >= 0 ? e.shape[u] : 1;
      }
      return a;
    }), i = ss(e, n).data;
    return new l(
      i,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: n }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [r] = this.next_functions;
    r.backward(ue(e, t.shape));
  }
};
c(ct, "Expand");
let Ie = ct;
P("expand", Ie);
const pr = D(
  (s, e) => Math.sin(s[e]),
  (s, e, t) => {
    e.backward(t.mul(t.mul(s.cos())));
  },
  "sin"
), gr = D(
  (s, e) => Math.cos(s[e]),
  (s, e, t) => {
    e.backward(t.mul(t.mul(s.sin().neg())));
  },
  "cos"
), mr = D(
  (s, e) => Math.tan(s[e]),
  (s, e, t) => {
    e.backward(t.mul(t.mul(s.cos().pow(-2))));
  },
  "tan"
), wr = ke(
  0,
  (s, e) => s + e,
  (s, e) => e,
  "sum"
), xr = ke(
  0,
  (s, e) => s + e,
  (s, e, t) => {
    const r = Re(s.shape, t, !1), n = r.length > 0 ? r.reduce((a, o) => a * o, 1) : 1, i = s.dataLength() / n;
    return e.mul(new l([1 / i]));
  },
  "mean",
  (s, e) => s / e
), br = ke(
  -1 / 0,
  (s, e) => Math.max(s, e),
  (s, e, t) => {
    const n = s.max(t, !0).expand(s.shape), i = s.eq(n).detach();
    return e.mul(i);
  },
  "max"
), qr = ke(
  1 / 0,
  (s, e) => Math.min(s, e),
  (s, e, t) => {
    const n = s.min(t, !0).expand(s.shape), i = s.eq(n).detach();
    return e.mul(i);
  },
  "min"
);
function ns(s, e, t, r = null) {
  if (s.shape.length + e < 0 || s.shape.length + t < 0)
    throw new Error(`Transpose: Dimension out of range (${e} and ${t})`);
  e = e < 0 ? s.shape.length + e : e, t = t < 0 ? s.shape.length + t : t;
  const n = [...s.shape];
  [n[e], n[t]] = [n[t], n[e]];
  const i = s.dataLength(), a = new Array(i), o = new Array(s.shape.length), u = new Array(n.length);
  for (let d = s.shape.length - 1, h = 1; d >= 0; d--)
    o[d] = h, h *= s.shape[d];
  for (let d = n.length - 1, h = 1; d >= 0; d--)
    u[d] = h, h *= n[d];
  for (let d = 0; d < i; d++) {
    let h = d, _ = 0;
    for (let f = 0; f < n.length; f++) {
      const p = u[f], R = Math.floor(h / p);
      h %= p;
      let b = f;
      f === e ? b = t : f === t && (b = e), _ += R * o[b];
    }
    a[d] = s.data[_];
  }
  return new l(
    a,
    { requires_grad: s.requires_grad },
    { operation: r, shape: n }
  );
}
c(ns, "_transpose_tensor");
const dt = class dt extends I {
  dim0;
  dim1;
  _forward(e, t, r) {
    return e.requires_grad && (this.saved_tensors = [e], this.dim0 = t, this.dim1 = r), this.next_functions.push(e.grad_fn ? e.grad_fn : x), ns(e, t, r, this);
  }
  _backward(e) {
    const [t] = this.saved_tensors, r = this.dim0, n = this.dim1, [i] = this.next_functions;
    i.backward(e.transpose(r, n));
  }
};
c(dt, "Transpose");
let De = dt;
P("transpose", De);
function as(s, e, t = null) {
  if (s.shape.length == 1 && e.shape.length == 1)
    return [s.mul(e).sum(), []];
  const r = s.shape.length == 1, n = e.shape.length == 1, i = r ? [1, s.shape[0]] : s.shape, a = n ? [e.shape[0], 1] : e.shape;
  if (i[i.length - 1] != a[a.length - 2])
    throw new Error("Shape mismatch: " + s.shape + " and " + e.shape);
  const o = Wt(i.slice(0, -2), a.slice(0, -2)).concat([
    i[i.length - 2],
    a[a.length - 1]
  ]), u = o.reduce((A, $) => A * $, 1), d = new Array(u).fill(0), h = ce(i, o), _ = ce(a, o), f = o[o.length - 2], p = o[o.length - 1], R = i[i.length - 1];
  for (let A = 0; A < u; A++) {
    const $ = A % (f * p), g = Math.floor($ / p), U = $ % p, M = de(h, o, A - U), W = de(_, o, A - g * p);
    let B = 0;
    for (let S = 0; S < R; S++)
      B += s.data[M + S] * e.data[W + S * p];
    d[A] = B;
  }
  let b = [...o];
  return r && (b = b.slice(0, -2).concat([o[o.length - 1]])), n && (b = b.slice(0, -1)), [new l(
    d,
    { requires_grad: s.requires_grad || e.requires_grad },
    { operation: t, shape: b }
  ), b];
}
c(as, "_matmul_tensor");
const ht = class ht extends qe {
  shape;
  _forward(e, t) {
    (e.requires_grad || t.requires_grad) && (this.saved_tensors = [e, t]), this.next_functions.push(e.grad_fn ? e.grad_fn : x), this.next_functions.push(t.grad_fn ? t.grad_fn : x);
    const r = as(e, t, e.requires_grad || t.requires_grad ? this : null);
    return this.shape = r[1], r[0];
  }
  _backward(e) {
    const [t, r] = this.saved_tensors, [n, i] = this.next_functions;
    if (t.shape.length === 1 && r.shape.length === 1) {
      n.backward(e.mul(r)), i.backward(e.mul(t));
      return;
    }
    if (t.shape.length === 1) {
      const u = e.unsqueeze(-2), d = t.unsqueeze(-2);
      let h = u.matmul(r.transpose(-2, -1)), _ = d.transpose(-2, -1).matmul(u);
      h = h.squeeze(-2), _ = ue(_, r.shape), n.backward(h), i.backward(_);
      return;
    }
    if (r.shape.length === 1) {
      const u = e.unsqueeze(-1), d = r.unsqueeze(-1);
      let h = u.matmul(d.transpose(-2, -1)), _ = t.transpose(-2, -1).matmul(u);
      h = ue(h, t.shape), _ = _.squeeze(-1), n.backward(h), i.backward(_);
      return;
    }
    let a = e.matmul(r.transpose(-2, -1)), o = t.transpose(-2, -1).matmul(e);
    a = ue(a, t.shape), o = ue(o, r.shape), n.backward(a), i.backward(o);
  }
};
c(ht, "Matmul");
let Ue = ht;
P("matmul", Ue);
function Xe(s, e, t, r, n, i, a, o) {
  const u = typeof r == "number" ? new Array(o).fill(r) : r, d = typeof n == "number" ? new Array(o).fill(n) : n, h = typeof i == "number" ? new Array(o).fill(i) : i, _ = s.shape[0], f = s.shape[1], p = e.shape[0], R = s.shape.slice(2), b = e.shape.slice(2);
  if (f !== e.shape[1] * a)
    throw new Error(`in_channels (${f}) must be divisible by groups (${a}) and match weight.shape[1] * groups (${e.shape[1] * a})`);
  const A = R.map((T, v) => Math.floor((T + 2 * d[v] - h[v] * (b[v] - 1) - 1) / u[v] + 1)), $ = [_, p, ...A], g = $.reduce((T, v) => T * v, 1), U = new Array(g).fill(0), M = /* @__PURE__ */ c((T) => {
    const v = new Array(T.length);
    let K = 1;
    for (let j = T.length - 1; j >= 0; j--)
      v[j] = K, K *= T[j];
    return v;
  }, "get_strides"), W = M(s.shape), B = M(e.shape), S = M($), ae = f / a, ie = p / a;
  for (let T = 0; T < _; T++)
    for (let v = 0; v < a; v++)
      for (let K = 0; K < ie; K++) {
        const j = v * ie + K, fe = A.reduce((k, O) => k * O, 1);
        for (let k = 0; k < fe; k++) {
          const O = new Array(o);
          let L = k;
          for (let w = o - 1; w >= 0; w--)
            O[w] = L % A[w], L = Math.floor(L / A[w]);
          let G = t ? t.data[j] : 0;
          for (let w = 0; w < ae; w++) {
            const se = v * ae + w, oe = b.reduce((Q, X) => Q * X, 1);
            for (let Q = 0; Q < oe; Q++) {
              const X = new Array(o);
              let q = Q;
              for (let m = o - 1; m >= 0; m--)
                X[m] = q % b[m], q = Math.floor(q / b[m]);
              let ge = !0;
              const me = new Array(o);
              for (let m = 0; m < o; m++) {
                const z = O[m] * u[m] + X[m] * h[m] - d[m];
                if (z < 0 || z >= R[m]) {
                  ge = !1;
                  break;
                }
                me[m] = z;
              }
              if (ge) {
                let m = T * W[0] + se * W[1];
                for (let N = 0; N < o; N++) m += me[N] * W[N + 2];
                let z = j * B[0] + w * B[1];
                for (let N = 0; N < o; N++) z += X[N] * B[N + 2];
                G += s.data[m] * e.data[z];
              }
            }
          }
          let pe = T * S[0] + j * S[1];
          for (let w = 0; w < o; w++) pe += O[w] * S[w + 2];
          U[pe] = G;
        }
      }
  return new l(U, { requires_grad: !1 }, { shape: $ });
}
c(Xe, "_convNd_forward");
function Ye(s, e, t, r, n, i, a, o, u, d, h) {
  const _ = typeof n == "number" ? new Array(u).fill(n) : n, f = typeof i == "number" ? new Array(u).fill(i) : i, p = typeof a == "number" ? new Array(u).fill(a) : a, R = e.shape[0], b = e.shape[1], A = t.shape[0], $ = e.shape.slice(2), g = t.shape.slice(2), U = s.shape.slice(2), M = /* @__PURE__ */ c((k) => {
    const O = new Array(k.length);
    let L = 1;
    for (let G = k.length - 1; G >= 0; G--)
      O[G] = L, L *= k[G];
    return O;
  }, "get_strides"), W = M(e.shape), B = M(t.shape), S = M(s.shape);
  let ae = null, ie = null, T = null, v = null, K = null;
  d && (v = new Array(e.dataLength()).fill(0)), h && (K = new Array(t.dataLength()).fill(0));
  const j = b / o, fe = A / o;
  for (let k = 0; k < R; k++)
    for (let O = 0; O < o; O++)
      for (let L = 0; L < fe; L++) {
        const G = O * fe + L, pe = U.reduce((w, se) => w * se, 1);
        for (let w = 0; w < pe; w++) {
          const se = new Array(u);
          let oe = w;
          for (let q = u - 1; q >= 0; q--)
            se[q] = oe % U[q], oe = Math.floor(oe / U[q]);
          let Q = k * S[0] + G * S[1];
          for (let q = 0; q < u; q++) Q += se[q] * S[q + 2];
          const X = s.data[Q];
          for (let q = 0; q < j; q++) {
            const ge = O * j + q, me = g.reduce((m, z) => m * z, 1);
            for (let m = 0; m < me; m++) {
              const z = new Array(u);
              let N = m;
              for (let y = u - 1; y >= 0; y--)
                z[y] = N % g[y], N = Math.floor(N / g[y]);
              let Bt = !0;
              const Tt = new Array(u);
              for (let y = 0; y < u; y++) {
                const re = se[y] * _[y] + z[y] * p[y] - f[y];
                if (re < 0 || re >= $[y]) {
                  Bt = !1;
                  break;
                }
                Tt[y] = re;
              }
              if (Bt) {
                let y = k * W[0] + ge * W[1];
                for (let V = 0; V < u; V++) y += Tt[V] * W[V + 2];
                let re = G * B[0] + q * B[1];
                for (let V = 0; V < u; V++) re += z[V] * B[V + 2];
                d && (v[y] += X * t.data[re]), h && (K[re] += X * e.data[y]);
              }
            }
          }
        }
      }
  if (d && (ae = new l(v, { requires_grad: !1 }, { shape: e.shape })), h && (ie = new l(K, { requires_grad: !1 }, { shape: t.shape })), r && r.requires_grad) {
    const k = [0];
    for (let O = 2; O < s.shape.length; O++) k.push(O);
    T = s.sum(k);
  }
  return [ae, ie, T];
}
c(Ye, "_convNd_backward");
const lt = class lt extends I {
  stride;
  padding;
  dilation;
  groups;
  _forward(e, t, r, n = 1, i = 0, a = 1, o = 1) {
    (e.requires_grad || t.requires_grad || r?.requires_grad) && (this.saved_tensors = [e, t], r && this.saved_tensors.push(r)), this.next_functions.push(e.grad_fn ? e.grad_fn : x), this.next_functions.push(t.grad_fn ? t.grad_fn : x), r && this.next_functions.push(r.grad_fn ? r.grad_fn : x), this.stride = n, this.padding = i, this.dilation = a, this.groups = o;
    const u = Xe(e, t, r, n, i, a, o, 1);
    return u.requires_grad = e.requires_grad || t.requires_grad || (r?.requires_grad ?? !1), u.grad_fn = u.requires_grad ? this : null, u;
  }
  _backward(e) {
    const t = this.saved_tensors[0], r = this.saved_tensors[1], n = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [i, a, o] = this.next_functions, [u, d, h] = Ye(
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
    t.requires_grad && i.backward(u), r.requires_grad && a.backward(d), n && n.requires_grad && o.backward(h);
  }
};
c(lt, "Conv1dOp");
let We = lt;
P("conv1d", We);
const _t = class _t extends I {
  stride;
  padding;
  dilation;
  groups;
  _forward(e, t, r, n = 1, i = 0, a = 1, o = 1) {
    (e.requires_grad || t.requires_grad || r?.requires_grad) && (this.saved_tensors = [e, t], r && this.saved_tensors.push(r)), this.next_functions.push(e.grad_fn ? e.grad_fn : x), this.next_functions.push(t.grad_fn ? t.grad_fn : x), r && this.next_functions.push(r.grad_fn ? r.grad_fn : x), this.stride = n, this.padding = i, this.dilation = a, this.groups = o;
    const u = Xe(e, t, r, n, i, a, o, 2);
    return u.requires_grad = e.requires_grad || t.requires_grad || (r?.requires_grad ?? !1), u.grad_fn = u.requires_grad ? this : null, u;
  }
  _backward(e) {
    const t = this.saved_tensors[0], r = this.saved_tensors[1], n = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [i, a, o] = this.next_functions, [u, d, h] = Ye(
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
    t.requires_grad && i.backward(u), r.requires_grad && a.backward(d), n && n.requires_grad && o.backward(h);
  }
};
c(_t, "Conv2dOp");
let Ne = _t;
P("conv2d", Ne);
const ft = class ft extends I {
  stride;
  padding;
  dilation;
  groups;
  _forward(e, t, r, n = 1, i = 0, a = 1, o = 1) {
    (e.requires_grad || t.requires_grad || r?.requires_grad) && (this.saved_tensors = [e, t], r && this.saved_tensors.push(r)), this.next_functions.push(e.grad_fn ? e.grad_fn : x), this.next_functions.push(t.grad_fn ? t.grad_fn : x), r && this.next_functions.push(r.grad_fn ? r.grad_fn : x), this.stride = n, this.padding = i, this.dilation = a, this.groups = o;
    const u = Xe(e, t, r, n, i, a, o, 3);
    return u.requires_grad = e.requires_grad || t.requires_grad || (r?.requires_grad ?? !1), u.grad_fn = u.requires_grad ? this : null, u;
  }
  _backward(e) {
    const t = this.saved_tensors[0], r = this.saved_tensors[1], n = this.saved_tensors.length > 2 ? this.saved_tensors[2] : null, [i, a, o] = this.next_functions, [u, d, h] = Ye(
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
    t.requires_grad && i.backward(u), r.requires_grad && a.backward(d), n && n.requires_grad && o.backward(h);
  }
};
c(ft, "Conv3dOp");
let Pe = ft;
P("conv3d", Pe);
const yr = F(
  (s, e, t, r) => s[t] < e[r] ? 1 : 0,
  (s, e, t, r) => {
  },
  "lt"
), vr = F(
  (s, e, t, r) => s[t] > e[r] ? 1 : 0,
  (s, e, t, r) => {
  },
  "gt"
), Ar = F(
  (s, e, t, r) => s[t] <= e[r] ? 1 : 0,
  (s, e, t, r) => {
  },
  "le"
), kr = F(
  (s, e, t, r) => s[t] >= e[r] ? 1 : 0,
  (s, e, t, r) => {
  },
  "ge"
), Or = F(
  (s, e, t, r) => s[t] == e[r] ? 1 : 0,
  (s, e, t, r) => {
  },
  "eq"
), Er = F(
  (s, e, t, r) => s[t] != e[r] ? 1 : 0,
  (s, e, t, r) => {
  },
  "ne"
), Rr = D(
  (s, e) => Math.max(s[e], 0),
  (s, e, t) => {
    e.backward(t.mul(t.mul(s.gt(0))));
  },
  "relu"
), Fr = D(
  (s, e) => 1 / (1 + Math.exp(-s[e])),
  (s, e, t) => {
    const r = s.sigmoid();
    e.backward(r.mul(r.mul(-1).add(1)).mul(t));
  },
  "sigmoid"
), ve = class ve extends l {
  constructor(e, t = {
    requires_grad: !0
  }, r = {}) {
    e instanceof l ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : e instanceof ve ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : super(e, t, r);
  }
};
c(ve, "Parameter");
let ee = ve;
const pt = class pt {
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
    t instanceof ee ? this.register_parameter(e, t) : this.register_module(e, t);
  }
  parameters() {
    let e = Object.values(this._parameters);
    for (const t of Object.values(this._modules))
      e = e.concat(t.parameters());
    return e;
  }
};
c(pt, "Module");
let te = pt;
const gt = class gt extends te {
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
};
c(gt, "Sequential");
let $e = gt;
const mt = class mt {
};
c(mt, "Loss");
let he = mt;
const wt = class wt extends he {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).pow(2).mean();
  }
};
c(wt, "MSELoss");
let Se = wt;
const xt = class xt extends he {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).abs().mean();
  }
};
c(xt, "L1Loss");
let ze = xt;
const bt = class bt extends he {
  weight;
  constructor(e = null) {
    super(), this.weight = e;
  }
  forward(e, t) {
    const r = t.mul(e.log()), n = t.neg().add(1).mul(e.neg().add(1).log()), i = r.add(n).neg().mean();
    return this.weight ? i.mul(this.weight) : i;
  }
};
c(bt, "BCELoss");
let Ce = bt;
function Ze(s) {
  return (...e) => new (H(s))().forward(...e);
}
c(Ze, "generate_function");
function Ct(s) {
  return (e) => (typeof e == "number" && (e = new l(e)), new (H(s))().forward(e));
}
c(Ct, "generate_unary_function");
const jt = Ct("relu"), Kt = Ct("sigmoid"), Lt = Ze("conv1d"), Gt = Ze("conv2d"), Vt = Ze("conv3d"), is = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  conv1d: Lt,
  conv2d: Gt,
  conv3d: Vt,
  relu: jt,
  sigmoid: Kt
}, Symbol.toStringTag, { value: "Module" })), qt = class qt extends te {
  weight;
  bias;
  constructor(e, t) {
    super();
    const r = Math.sqrt(1 / e);
    this.weight = new ee(
      xe([t, e]).mul(2 * r).sub(r)
    ), this.bias = new ee(
      xe([t]).mul(2 * r).sub(r)
    ), this.register("weight", this.weight), this.register("bias", this.bias);
  }
  forward(e) {
    return e.matmul(this.weight.transpose(0, 1)).add(this.bias);
  }
};
c(qt, "Linear");
let je = qt;
const yt = class yt extends te {
  constructor() {
    super();
  }
  forward(e) {
    return jt(e);
  }
};
c(yt, "ReLU");
let Ke = yt;
const vt = class vt extends te {
  constructor() {
    super();
  }
  forward(e) {
    return Kt(e);
  }
};
c(vt, "Sigmoid");
let Le = vt;
const At = class At extends te {
  weight;
  bias;
  in_channels;
  out_channels;
  kernel_size;
  stride;
  padding;
  dilation;
  groups;
  constructor(e, t, r, n, i, a, o, u, d) {
    if (super(), this.in_channels = e, this.out_channels = t, this.kernel_size = r, this.stride = n, this.padding = i, this.dilation = a, this.groups = o, e % o !== 0)
      throw new Error("in_channels must be divisible by groups");
    if (t % o !== 0)
      throw new Error("out_channels must be divisible by groups");
    const h = typeof r == "number" ? new Array(d).fill(r) : r, _ = h.reduce((p, R) => p * R, 1), f = Math.sqrt(o / (e * _));
    this.weight = new ee(
      xe([t, e / o, ...h]).mul(2 * f).sub(f)
    ), this.register("weight", this.weight), u ? (this.bias = new ee(
      xe([t]).mul(2 * f).sub(f)
    ), this.register("bias", this.bias)) : this.bias = null;
  }
};
c(At, "_ConvNd");
let le = At;
const kt = class kt extends le {
  constructor(e, t, r, n = 1, i = 0, a = 1, o = 1, u = !0) {
    super(e, t, r, n, i, a, o, u, 1);
  }
  forward(e) {
    return Lt(e, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
};
c(kt, "Conv1d");
let Ge = kt;
const Ot = class Ot extends le {
  constructor(e, t, r, n = 1, i = 0, a = 1, o = 1, u = !0) {
    super(e, t, r, n, i, a, o, u, 2);
  }
  forward(e) {
    return Gt(e, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
};
c(Ot, "Conv2d");
let Ve = Ot;
const Et = class Et extends le {
  constructor(e, t, r, n = 1, i = 0, a = 1, o = 1, u = !0) {
    super(e, t, r, n, i, a, o, u, 3);
  }
  forward(e) {
    return Vt(e, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
  }
};
c(Et, "Conv3d");
let He = Et;
const Mr = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  BCELoss: Ce,
  Conv1d: Ge,
  Conv2d: Ve,
  Conv3d: He,
  L1Loss: ze,
  Linear: je,
  MSELoss: Se,
  Module: te,
  Parameter: ee,
  ReLU: Ke,
  Sequential: $e,
  Sigmoid: Le,
  functional: is
}, Symbol.toStringTag, { value: "Module" })), Rt = class Rt {
  params;
  defaults;
  constructor(e, t) {
    this.params = e, this.defaults = t;
  }
  zero_grad() {
    for (const e of this.params)
      e.grad = null;
  }
};
c(Rt, "Optimizer");
let _e = Rt;
const Ft = class Ft extends _e {
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
};
c(Ft, "SGD");
let Je = Ft;
const Mt = class Mt extends _e {
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
        m: we(e),
        v: we(e),
        vmax: we(e)
      });
      const r = this.state.get(e);
      r.m = r.m.mul(this.beta1).add(t.mul(1 - this.beta1)), r.v = r.v.mul(this.beta2).add(t.mul(t).mul(1 - this.beta2));
      const n = 1 - Math.pow(this.beta1, this.step_count), i = 1 - Math.pow(this.beta2, this.step_count);
      let a;
      const o = r.m.div(n);
      this.amsgrad ? (r.vmax = r.vmax.maximum(r.v), a = r.vmax.div(i)) : a = r.v.div(i);
      const u = o.div(a.sqrt().add(this.eps)).mul(this.lr), d = e.sub(u);
      e.data = d.data;
    }
  }
};
c(Mt, "Adam");
let Qe = Mt;
const Br = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Adam: Qe,
  Optimizer: _e,
  SGD: Je
}, Symbol.toStringTag, { value: "Module" }));
export {
  ye as AccumulateGrad,
  br as Max,
  xr as Mean,
  qr as Min,
  wr as Sum,
  l as Tensor,
  I as TorchFunction,
  _s as __left_index__,
  fs as __right_index__,
  Es as abs,
  ps as add,
  ls as arange,
  Ws as cos,
  ws as div,
  Hs as eq,
  Y as eventBus,
  Z as events,
  ks as exp,
  Ds as expand,
  bs as fmod,
  Vs as ge,
  Ls as gt,
  Gs as le,
  hs as linspace,
  vs as log,
  Ks as lt,
  js as matmul,
  zs as max,
  qs as maximum,
  $s as mean,
  Ss as min,
  ys as minimum,
  ms as mul,
  Ms as nan_to_num,
  Js as ne,
  Rs as neg,
  Mr as nn,
  Pt as ones,
  ds as ones_like,
  Br as optim,
  xs as pow,
  xe as rand,
  cs as randint,
  us as randn,
  Fs as reciprocal,
  Bs as reshape,
  Zt as sign,
  Us as sin,
  As as sqrt,
  Os as square,
  Ts as squeeze,
  gs as sub,
  Ps as sum,
  Ns as tan,
  Cs as transpose,
  Is as unsqueeze,
  Jt as zeros,
  we as zeros_like
};
//# sourceMappingURL=torch.browser.es.js.map
