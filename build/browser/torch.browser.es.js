var Ze = Object.defineProperty;
var o = (r, e) => Ze(r, "name", { value: e, configurable: !0 });
function ze(r, e) {
  const t = Math.max(r.length, e.length), s = [...Array(t - r.length).fill(1), ...r], n = [...Array(t - e.length).fill(1), ...e], i = [];
  for (let a = 0; a < t; a++) {
    if (s[a] !== n[a] && s[a] !== 1 && n[a] !== 1)
      throw new Error(`Shape mismatch: ${r} and ${e}`);
    i.push(Math.max(s[a], n[a]));
  }
  return i;
}
o(ze, "_broadcast_shape");
function Ge(r, e, t) {
  const s = $(e, r), n = new Array(e.reduce((i, a) => i * a, 1)).fill(0);
  for (let i = 0; i < t.length; i++)
    n[j(s, r, i)] += t[i];
  return n;
}
o(Ge, "_unbroadcast");
function $(r, e) {
  return r.length >= e.length ? r : [...Array(e.length - r.length).fill(1), ...r];
}
o($, "_pad_shape");
function j(r, e, t) {
  let s = 0, n = 1, i = t;
  for (let a = r.length - 1; a >= 0; a--) {
    if (r[a] > 1) {
      const u = i % e[a];
      s = s + u * n;
    }
    n *= r[a], i = Math.floor(i / e[a]);
  }
  return s;
}
o(j, "_get_original_index");
function Q(r) {
  return Array.isArray(r[0]) ? r[0] : r;
}
o(Q, "get_shape_from_args");
function _t(...r) {
  const e = Q(r), t = new d(Array(e.reduce((s, n) => s * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
o(_t, "randn");
function je(...r) {
  const e = Q(r), t = new d(Array(e.reduce((s, n) => s * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
o(je, "rand");
function ft(r, e, t) {
  const s = new d(
    Array(t.reduce((n, i) => n * i, 1)).fill(Math.floor(Math.random() * (e - r) + r))
  );
  return s.shape = t, s;
}
o(ft, "randint");
function Le(...r) {
  const e = Q(r), t = new d(Array(e.reduce((s, n) => s * n, 1)).fill(1));
  return t.shape = e, t;
}
o(Le, "ones");
function et(...r) {
  const e = Q(r), t = new d(Array(e.reduce((s, n) => s * n, 1)).fill(0));
  return t.shape = e, t;
}
o(et, "zeros");
function pt(r) {
  return Le(r.shape);
}
o(pt, "ones_like");
function G(r) {
  return et(r.shape);
}
o(G, "zeros_like");
function gt(r, e, t) {
  const s = [], n = (e - r) / (t - 1);
  for (let i = 0; i < t - 1; i++)
    s.push(r + i * n);
  return s.push(e), new d(s);
}
o(gt, "linspace");
function mt(r, e = void 0, t = 1) {
  const s = [];
  for (let n = r; n < e; n += t)
    s.push(n);
  return new d(s);
}
o(mt, "arange");
let tt = 0;
const Ve = /* @__PURE__ */ o(() => tt++, "getNextId"), R = new EventTarget(), F = {
  TENSOR_BEFORE_BACKWARD: "tensor.beforeBackward",
  TENSOR_AFTER_BACKWARD: "tensor.afterBackward",
  OPERATION_BEFORE_FORWARD: "operation.beforeForward",
  OPERATION_AFTER_FORWARD: "operation.afterForward",
  OPERATION_BEFORE_BACKWARD: "operation.beforeBackward",
  OPERATION_AFTER_BACKWARD: "operation.afterBackward",
  OPERATION_BEFORE_ACCUMULATE_GRAD: "operation.beforeAccumulateGrad",
  OPERATION_AFTER_ACCUMULATE_GRAD: "operation.afterAccumulateGrad"
};
function rt(...r) {
  for (const e of r)
    if (e instanceof d && e.requires_grad)
      return !0;
  return !1;
}
o(rt, "resultRequiresGrad");
const me = class me {
  id = Ve();
  next_functions = [];
  saved_tensors = [];
  _retained_tensors = [];
  forward(...e) {
    const t = rt(...e);
    R.dispatchEvent(new CustomEvent(F.OPERATION_BEFORE_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e
      }
    }));
    const s = this._forward(...e);
    return R.dispatchEvent(new CustomEvent(F.OPERATION_AFTER_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e,
        result: s
      }
    })), s;
  }
  backward(e) {
    R.dispatchEvent(new CustomEvent(F.OPERATION_BEFORE_BACKWARD, { detail: { operation: this, dz: e } }));
    for (const t of this._retained_tensors)
      t.grad || (t.grad = new d(new Array(t.dataLength()).fill(0))), t.grad = t.grad.add(e);
    this._backward(e), R.dispatchEvent(new CustomEvent(F.OPERATION_AFTER_BACKWARD, { detail: { operation: this, dz: e } }));
  }
};
o(me, "TorchFunction");
let y = me;
const we = class we extends y {
  _forward(...e) {
    throw new Error("NullOp should not be called");
  }
  _backward(e) {
  }
};
o(we, "NullOp");
let Z = we;
const A = new Z(), xe = class xe extends y {
};
o(xe, "UnaryFunction");
let L = xe;
const be = class be extends y {
};
o(be, "BinaryFunction");
let V = be;
const ye = class ye extends L {
  variable;
  _forward(e) {
    return this.variable = e, e;
  }
  _backward(e) {
    if (this.variable.grad || (this.variable.grad = G(this.variable)), R.dispatchEvent(new CustomEvent(F.OPERATION_BEFORE_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } })), typeof e == "number")
      this.variable.grad = this.variable.grad.add(e);
    else {
      const t = Ge(e.shape, this.variable.shape, e.data);
      this.variable.grad = this.variable.grad.add(new d(t, {}, { shape: this.variable.shape }));
    }
    R.dispatchEvent(new CustomEvent(F.OPERATION_AFTER_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } }));
  }
};
o(ye, "AccumulateGrad");
let H = ye;
const He = /* @__PURE__ */ new Map(), Y = /* @__PURE__ */ new Map();
function O(r, e) {
  He.set(r, e);
}
o(O, "registerOperation");
function M(r) {
  const e = He.get(r);
  if (!e)
    throw new Error(`Operation '${r}' is not registered.`);
  return e;
}
o(M, "getOperation");
function Ke(r) {
  const e = Y.get(r);
  return e || (Y.set(r, new (M(r))()), Y.get(r));
}
o(Ke, "getOperationCache");
function st(r) {
  if (ArrayBuffer.isView(r))
    return [r.length];
  const e = [];
  for (; Array.isArray(r); )
    e.push(r.length), r = r[0];
  return e;
}
o(st, "_get_shape");
function Je(r) {
  return Array.isArray(r) ? r.flatMap((e) => Je(e)) : ArrayBuffer.isView(r) ? Array.from(r) : [r];
}
o(Je, "_flatten");
const I = class I {
  // Auto-generated ID
  id = Ve();
  // Optional user-defined name
  name = null;
  data;
  shape;
  grad_fn = null;
  grad = null;
  requires_grad;
  constructor(e, t = {}, s = {}) {
    if (this.data = Je(e), this.requires_grad = t.requires_grad ?? !1, t.name && (this.name = t.name), this.shape = s.shape ?? st(e), this.grad_fn = s.operation ?? null, this.requires_grad && !this.grad_fn) {
      const n = new H();
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
    const t = this.data, s = /* @__PURE__ */ o((n) => {
      const i = this.shape[n], a = new Array(i), u = n === this.shape.length - 1;
      for (let h = 0; h < i; h++)
        u ? a[h] = t[e++] : a[h] = s(n + 1);
      return a;
    }, "buildDimension");
    return s(0);
  }
  dataLength() {
    return this.data.length;
  }
  _executeUnaryOp(e) {
    return (this.requires_grad ? new (M(e))() : Ke(e)).forward(this);
  }
  _executeBinaryOp(e, t) {
    return typeof t == "number" && (t = new I(t)), (this.requires_grad || t.requires_grad ? new (M(e))() : Ke(e)).forward(this, t);
  }
  _executeOpRaw(e, ...t) {
    return new (M(e))().forward(this, ...t);
  }
  item() {
    if (this.dataLength() !== 1)
      throw new Error("Tensor.item() is only valid for scalars");
    return this.data[0];
  }
  detach() {
    return new I(this.data, { requires_grad: !1 }, { shape: this.shape });
  }
  detach_() {
    this.requires_grad = !1, this.grad = null, this.grad_fn = null;
  }
  zero_() {
    this.data = Array(this.dataLength()).fill(0);
  }
  is_retain_grad = !1;
  retain_grad() {
    this.grad_fn instanceof H || this.is_retain_grad || (this.is_retain_grad = !0, this.grad_fn._retained_tensors.push(this));
  }
  backward(e) {
    if (this.requires_grad) {
      if (e)
        e.toArray_();
      else {
        if (this.dataLength() !== 1)
          throw new Error("Gradient is required for non-scalar tensors");
        e = new I(1);
      }
      this.grad_fn && (R.dispatchEvent(new CustomEvent(F.TENSOR_BEFORE_BACKWARD, { detail: { tensor: this } })), this.grad_fn.backward(e), R.dispatchEvent(new CustomEvent(F.TENSOR_AFTER_BACKWARD, { detail: { tensor: this } })));
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
};
o(I, "Tensor");
let d = I;
function E(r) {
  return (...e) => new (M(r))().forward(...e);
}
o(E, "generate_function$1");
function q(r) {
  return (e) => (typeof e == "number" && (e = new d(e)), new (M(r))().forward(e));
}
o(q, "generate_unary_function$1");
function m(r) {
  return (e, t) => (typeof e == "number" && (e = new d(e)), typeof t == "number" && (t = new d(t)), new (M(r))().forward(e, t));
}
o(m, "generate_binary_function$1");
const wt = m("__left_index__"), xt = m("__right_index__"), bt = m("add"), yt = m("sub"), At = m("mul"), qt = m("div"), vt = m("pow"), Ot = m("fmod"), kt = m("maximum"), Et = m("minimum"), Rt = q("log"), Ft = q("sqrt"), Mt = q("exp"), Bt = q("square"), Tt = q("abs"), nt = q("sign"), Ct = q("neg"), Dt = q("reciprocal"), Ut = E("reshape"), It = E("squeeze"), Pt = E("unsqueeze"), St = E("expand"), Wt = q("sin"), $t = q("cos"), jt = q("tan"), Kt = E("sum"), Nt = E("mean"), zt = E("min"), Gt = E("max"), Lt = E("transpose"), Vt = m("matmul"), Ht = m("lt"), Jt = m("gt"), Qt = m("le"), Xt = m("ge"), Yt = m("eq"), Zt = m("ne");
function Ne(r) {
  const e = new Array(r.length).fill(1);
  for (let t = r.length - 2; t >= 0; t--)
    e[t] = e[t + 1] * r[t + 1];
  return e;
}
o(Ne, "_get_strides");
function at(r, e) {
  return e.map((t) => {
    const s = Math.floor(r / t);
    return r %= t, s;
  });
}
o(at, "_unravel_index");
function it(r, e) {
  return r.reduce((t, s, n) => t + s * e[n], 0);
}
o(it, "_ravel_index");
function ee(r, e, t = !1) {
  if (e === void 0) return t ? r.map(() => 1) : [];
  const n = (Array.isArray(e) ? e : [e]).map((i) => i < 0 ? i + r.length : i);
  return t ? r.map((i, a) => n.includes(a) ? 1 : i) : r.filter((i, a) => !n.includes(a));
}
o(ee, "_get_reduction_shape");
function w(r, e, t = null) {
  const s = /* @__PURE__ */ o((a, u, h, c, l, _) => {
    const f = Array(_);
    for (let p = 0; p < _; p++) {
      const k = j(u, l, p), x = j(c, l, p);
      f[p] = r(a, h, k, x);
    }
    return f;
  }, "kernel"), n = /* @__PURE__ */ o((a, u, h = null) => {
    const c = ze(a.shape, u.shape), l = $(a.shape, c), _ = $(u.shape, c), f = c.reduce((p, k) => p * k, 1);
    return new d(
      s(
        a.data,
        l,
        u.data,
        _,
        c,
        f
      ),
      { requires_grad: a.requires_grad || u.requires_grad },
      { operation: h, shape: c }
    );
  }, "forward_tensor"), i = {
    [t]: class extends V {
      _forward(a, u) {
        return (a.requires_grad || u.requires_grad) && (this.saved_tensors = [a, u]), this.next_functions.push(a.grad_fn ? a.grad_fn : A), this.next_functions.push(u.grad_fn ? u.grad_fn : A), n(a, u, a.requires_grad || u.requires_grad ? this : null);
      }
      _backward(a) {
        const [u, h] = this.saved_tensors, [c, l] = this.next_functions;
        e(u, h, c, l, a);
      }
    }
  }[t];
  return t && O(t, i), i;
}
o(w, "BinaryFunctionMixin");
function b(r, e, t = null) {
  const s = /* @__PURE__ */ o((a, u) => {
    const h = Array(u);
    for (let c = 0; c < u; c++)
      h[c] = r(a, c);
    return h;
  }, "kernel"), n = /* @__PURE__ */ o((a, u = null) => {
    const h = a.dataLength();
    return new d(
      s(a.data, h),
      { requires_grad: a.requires_grad },
      { operation: u, shape: a.shape }
    );
  }, "forward_tensor"), i = {
    [t]: class extends L {
      _forward(a) {
        return a.requires_grad && (this.saved_tensors = [a]), this.next_functions.push(a.grad_fn ? a.grad_fn : A), n(a, a.requires_grad ? this : null);
      }
      _backward(a) {
        const [u] = this.saved_tensors, [h] = this.next_functions;
        e(u, h, a);
      }
    }
  }[t];
  return t && O(t, i), i;
}
o(b, "UnaryFunctionMixin");
function X(r, e, t, s = null, n) {
  const i = {
    [s]: class extends y {
      dim;
      keepdim;
      _forward(a, u, h = !1) {
        this.dim = u, this.keepdim = h, a.requires_grad && (this.saved_tensors = [a]), this.next_functions.push(a.grad_fn ? a.grad_fn : A);
        const c = ee(a.shape, u, h), l = c.reduce((g, U) => g * U, 1), _ = new Array(l).fill(r), f = new Array(l).fill(0), p = Ne(a.shape), k = Ne(c), v = (u === void 0 ? [] : Array.isArray(u) ? u : [u]).map((g) => g < 0 ? g + a.shape.length : g), D = u === void 0;
        for (let g = 0; g < a.data.length; g++) {
          const U = at(g, p);
          let B;
          if (D)
            B = h ? U.map(() => 0) : [];
          else {
            B = [];
            for (let T = 0; T < a.shape.length; T++)
              v.includes(T) ? h && B.push(0) : B.push(U[T]);
          }
          const S = it(B, k);
          _[S] = e(_[S], a.data[g]), f[S]++;
        }
        if (n)
          for (let g = 0; g < l; g++)
            _[g] = n(_[g], f[g]);
        return new d(
          _,
          { requires_grad: a.requires_grad },
          { operation: a.requires_grad ? this : null, shape: c }
        );
      }
      _backward(a) {
        const [u] = this.saved_tensors, [h] = this.next_functions;
        let c = a;
        const l = ee(u.shape, this.dim, !0);
        a.shape.length !== l.length && (c = a.reshape(l));
        let _ = c.expand(u.shape);
        const f = t(u, _, this.dim, this.keepdim);
        h.backward(f);
      }
    }
  }[s];
  return s && O(s, i), i;
}
o(X, "ReductionFunctionMixin");
function W(r, e) {
  const t = Ge(r.shape, e, r.data);
  return new d(t, { requires_grad: r.requires_grad }, { shape: e });
}
o(W, "unbroadcast");
function ot(r, e) {
  return r.mul(Le(e));
}
o(ot, "broadcast");
const er = w(
  (r, e, t, s) => t,
  (r, e, t, s, n) => {
  },
  "__left_index__"
), tr = w(
  (r, e, t, s) => s,
  (r, e, t, s, n) => {
  },
  "__right_index__"
), rr = w(
  (r, e, t, s) => r[t] + e[s],
  (r, e, t, s, n) => {
    t.backward(n), s.backward(n);
  },
  "add"
), sr = w(
  (r, e, t, s) => r[t] - e[s],
  (r, e, t, s, n) => {
    t.backward(n), s.backward(n.mul(new d(-1)));
  },
  "sub"
), nr = w(
  (r, e, t, s) => r[t] * e[s],
  (r, e, t, s, n) => {
    t.backward(n.mul(e)), s.backward(n.mul(r));
  },
  "mul"
), ar = w(
  (r, e, t, s) => r[t] / e[s],
  (r, e, t, s, n) => {
    t.backward(n.div(e)), s.backward(n.mul(r).mul(new d(-1)).div(e).div(e));
  },
  "div"
), ir = w(
  (r, e, t, s) => Math.pow(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(e).mul(r.pow(e.sub(new d(1))))), s.backward(n.mul(r.pow(e)).mul(r.log()));
  },
  "pow"
), or = w(
  (r, e, t, s) => r[t] % e[s],
  (r, e, t, s, n) => {
    t.backward(n);
  },
  "fmod"
), ur = w(
  (r, e, t, s) => Math.max(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(r.ge(e))), s.backward(n.mul(e.gt(r)));
  },
  "maximum"
), cr = w(
  (r, e, t, s) => Math.min(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(r.le(e))), s.backward(n.mul(e.lt(r)));
  },
  "minimum"
);
function ut(r, e, t = null) {
  const s = new Array(r.dataLength());
  for (let n = 0; n < s.length; n++)
    s[n] = Math.pow(r.data[n], e);
  return new d(
    s,
    { requires_grad: r.requires_grad },
    { operation: t, shape: r.shape }
  );
}
o(ut, "_powint_tensor");
const Ae = class Ae extends y {
  n;
  _forward(e, t) {
    return e.requires_grad && (this.saved_tensors = [e], this.n = t), this.next_functions.push(e.grad_fn ? e.grad_fn : A), ut(e, t, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, s = this.n, [n] = this.next_functions;
    n.backward(e.mul(s).mul(t.pow(s - 1)));
  }
};
o(Ae, "PowInt");
let te = Ae;
O("powint", te);
const hr = b(
  (r, e) => Math.log(r[e]),
  (r, e, t) => {
    e.backward(t.mul(new d(1).div(r)));
  },
  "log"
), dr = b(
  (r, e) => Math.sqrt(r[e]),
  (r, e, t) => {
    e.backward(t.mul(new d(1).div(r.sqrt()).div(2)));
  },
  "sqrt"
), lr = b(
  (r, e) => Math.exp(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.exp())));
  },
  "exp"
), _r = b(
  (r, e) => r[e] * r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(r).mul(new d(2))));
  },
  "square"
), fr = b(
  (r, e) => Math.abs(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(nt(r))));
  },
  "abs"
), pr = b(
  (r, e) => Math.sign(r[e]),
  (r, e, t) => {
    e.backward(0);
  },
  "sign"
), gr = b(
  (r, e) => -r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(new d(-1))));
  },
  "neg"
), mr = b(
  (r, e) => 1 / r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.pow(-2))).neg());
  },
  "reciprocal"
), qe = class qe extends y {
  _forward(e, t) {
    const s = e.dataLength(), n = t.reduce((i, a) => i * a, 1);
    if (s !== n)
      throw new Error("Shape mismatch: " + e.shape + " and " + t);
    return e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(A), new d(
      e.data,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: t }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.reshape(t.shape));
  }
};
o(qe, "Reshape");
let re = qe;
O("reshape", re);
const ve = class ve extends y {
  _forward(e, t) {
    e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(A);
    let s = [...e.shape];
    return t !== void 0 ? (t < 0 && (t += e.shape.length), s[t] === 1 && s.splice(t, 1)) : s = s.filter((n) => n !== 1), new d(
      e.data,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: s }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.reshape(t.shape));
  }
};
o(ve, "Squeeze");
let se = ve;
O("squeeze", se);
const Oe = class Oe extends y {
  _forward(e, t) {
    e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(A), t < 0 && (t += e.shape.length + 1);
    const s = [...e.shape];
    return s.splice(t, 0, 1), new d(
      e.data,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: s }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.reshape(t.shape));
  }
};
o(Oe, "Unsqueeze");
let ne = Oe;
O("unsqueeze", ne);
const ke = class ke extends y {
  _forward(e, t) {
    e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(A);
    const s = t.length - e.shape.length, n = t.map((a, u) => {
      if (a === -1) {
        const h = u - s;
        return h >= 0 ? e.shape[h] : 1;
      }
      return a;
    }), i = ot(e, n).data;
    return new d(
      i,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: n }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(W(e, t.shape));
  }
};
o(ke, "Expand");
let ae = ke;
O("expand", ae);
const wr = b(
  (r, e) => Math.sin(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.cos())));
  },
  "sin"
), xr = b(
  (r, e) => Math.cos(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.sin().neg())));
  },
  "cos"
), br = b(
  (r, e) => Math.tan(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.cos().pow(-2))));
  },
  "tan"
), yr = X(
  0,
  (r, e) => r + e,
  (r, e) => e,
  "sum"
), Ar = X(
  0,
  (r, e) => r + e,
  (r, e, t) => {
    const s = ee(r.shape, t, !1), n = s.length > 0 ? s.reduce((a, u) => a * u, 1) : 1, i = r.dataLength() / n;
    return e.mul(new d([1 / i]));
  },
  "mean",
  (r, e) => r / e
), qr = X(
  -1 / 0,
  (r, e) => Math.max(r, e),
  (r, e, t) => {
    const n = r.max(t, !0).expand(r.shape), i = r.eq(n).detach();
    return e.mul(i);
  },
  "max"
), vr = X(
  1 / 0,
  (r, e) => Math.min(r, e),
  (r, e, t) => {
    const n = r.min(t, !0).expand(r.shape), i = r.eq(n).detach();
    return e.mul(i);
  },
  "min"
);
function ct(r, e, t, s = null) {
  if (r.shape.length + e < 0 || r.shape.length + t < 0)
    throw new Error(`Transpose: Dimension out of range (${e} and ${t})`);
  e = e < 0 ? r.shape.length + e : e, t = t < 0 ? r.shape.length + t : t;
  const n = [...r.shape];
  [n[e], n[t]] = [n[t], n[e]];
  const i = r.dataLength(), a = new Array(i), u = new Array(r.shape.length), h = new Array(n.length);
  for (let c = r.shape.length - 1, l = 1; c >= 0; c--)
    u[c] = l, l *= r.shape[c];
  for (let c = n.length - 1, l = 1; c >= 0; c--)
    h[c] = l, l *= n[c];
  for (let c = 0; c < i; c++) {
    let l = c, _ = 0;
    for (let f = 0; f < n.length; f++) {
      const p = h[f], k = Math.floor(l / p);
      l %= p;
      let x = f;
      f === e ? x = t : f === t && (x = e), _ += k * u[x];
    }
    a[c] = r.data[_];
  }
  return new d(
    a,
    { requires_grad: r.requires_grad },
    { operation: s, shape: n }
  );
}
o(ct, "_transpose_tensor");
const Ee = class Ee extends y {
  dim0;
  dim1;
  _forward(e, t, s) {
    return e.requires_grad && (this.saved_tensors = [e], this.dim0 = t, this.dim1 = s), this.next_functions.push(e.grad_fn ? e.grad_fn : A), ct(e, t, s, this);
  }
  _backward(e) {
    const [t] = this.saved_tensors, s = this.dim0, n = this.dim1, [i] = this.next_functions;
    i.backward(e.transpose(s, n));
  }
};
o(Ee, "Transpose");
let ie = Ee;
O("transpose", ie);
function ht(r, e, t = null) {
  if (r.shape.length == 1 && e.shape.length == 1)
    return [r.mul(e).sum(), []];
  const s = r.shape.length == 1, n = e.shape.length == 1, i = s ? [1, r.shape[0]] : r.shape, a = n ? [e.shape[0], 1] : e.shape;
  if (i[i.length - 1] != a[a.length - 2])
    throw new Error("Shape mismatch: " + r.shape + " and " + e.shape);
  const u = ze(i.slice(0, -2), a.slice(0, -2)).concat([
    i[i.length - 2],
    a[a.length - 1]
  ]), h = u.reduce((v, D) => v * D, 1), c = new Array(h).fill(0), l = $(i, u), _ = $(a, u), f = u[u.length - 2], p = u[u.length - 1], k = i[i.length - 1];
  for (let v = 0; v < h; v++) {
    const D = v % (f * p), g = Math.floor(D / p), U = D % p;
    let B = j(l, u, v - U), S = j(_, u, v - g * p), T = 0;
    for (let z = 0; z < k; z++)
      T += r.data[B + z] * e.data[S + z * p];
    c[v] = T;
  }
  let x = [...u];
  return s && (x = x.slice(0, -2).concat([u[u.length - 1]])), n && (x = x.slice(0, -1)), [new d(
    c,
    { requires_grad: r.requires_grad || e.requires_grad },
    { operation: t, shape: x }
  ), x];
}
o(ht, "_matmul_tensor");
const Re = class Re extends V {
  shape;
  _forward(e, t) {
    (e.requires_grad || t.requires_grad) && (this.saved_tensors = [e, t]), this.next_functions.push(e.grad_fn ? e.grad_fn : A), this.next_functions.push(t.grad_fn ? t.grad_fn : A);
    const s = ht(e, t, e.requires_grad || t.requires_grad ? this : null);
    return this.shape = s[1], s[0];
  }
  _backward(e) {
    const [t, s] = this.saved_tensors, [n, i] = this.next_functions;
    if (t.shape.length === 1 && s.shape.length === 1) {
      n.backward(e.mul(s)), i.backward(e.mul(t));
      return;
    }
    if (t.shape.length === 1) {
      const h = e.unsqueeze(-2), c = t.unsqueeze(-2);
      let l = h.matmul(s.transpose(-2, -1)), _ = c.transpose(-2, -1).matmul(h);
      l = l.squeeze(-2), _ = W(_, s.shape), n.backward(l), i.backward(_);
      return;
    }
    if (s.shape.length === 1) {
      const h = e.unsqueeze(-1), c = s.unsqueeze(-1);
      let l = h.matmul(c.transpose(-2, -1)), _ = t.transpose(-2, -1).matmul(h);
      l = W(l, t.shape), _ = _.squeeze(-1), n.backward(l), i.backward(_);
      return;
    }
    let a = e.matmul(s.transpose(-2, -1)), u = t.transpose(-2, -1).matmul(e);
    a = W(a, t.shape), u = W(u, s.shape), n.backward(a), i.backward(u);
  }
};
o(Re, "Matmul");
let oe = Re;
O("matmul", oe);
const Or = w(
  (r, e, t, s) => r[t] < e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "lt"
), kr = w(
  (r, e, t, s) => r[t] > e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "gt"
), Er = w(
  (r, e, t, s) => r[t] <= e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "le"
), Rr = w(
  (r, e, t, s) => r[t] >= e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "ge"
), Fr = w(
  (r, e, t, s) => r[t] == e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "eq"
), Mr = w(
  (r, e, t, s) => r[t] != e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "ne"
), Br = b(
  (r, e) => Math.max(r[e], 0),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.gt(0))));
  },
  "relu"
), Tr = b(
  (r, e) => 1 / (1 + Math.exp(-r[e])),
  (r, e, t) => {
    const s = r.sigmoid();
    e.backward(s.mul(s.mul(-1).add(1)).mul(t));
  },
  "sigmoid"
), J = class J extends d {
  constructor(e, t = {
    requires_grad: !0
  }, s = {}) {
    e instanceof d ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : e instanceof J ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : super(e, t, s);
  }
};
o(J, "Parameter");
let P = J;
const Fe = class Fe {
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
    t instanceof P ? this.register_parameter(e, t) : this.register_module(e, t);
  }
  parameters() {
    let e = Object.values(this._parameters);
    for (const t of Object.values(this._modules))
      e = e.concat(t.parameters());
    return e;
  }
};
o(Fe, "Module");
let C = Fe;
const Me = class Me extends C {
  weight;
  bias;
  constructor(e, t) {
    super();
    const s = Math.sqrt(1 / e);
    this.weight = new P(
      je([t, e]).mul(2 * s).sub(s)
    ), this.bias = new P(
      je([t]).mul(2 * s).sub(s)
    ), this.register("weight", this.weight), this.register("bias", this.bias);
  }
  forward(e) {
    return e.matmul(this.weight.transpose(0, 1)).add(this.bias);
  }
};
o(Me, "Linear");
let ue = Me;
const Be = class Be extends C {
  constructor() {
    super();
  }
  forward(e) {
    return Xe(e);
  }
};
o(Be, "ReLU");
let ce = Be;
const Te = class Te extends C {
  constructor() {
    super();
  }
  forward(e) {
    return Ye(e);
  }
};
o(Te, "Sigmoid");
let he = Te;
const Ce = class Ce extends C {
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
};
o(Ce, "Sequential");
let de = Ce;
const De = class De {
};
o(De, "Loss");
let K = De;
const Ue = class Ue extends K {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).pow(2).mean();
  }
};
o(Ue, "MSELoss");
let le = Ue;
const Ie = class Ie extends K {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).abs().mean();
  }
};
o(Ie, "L1Loss");
let _e = Ie;
const Pe = class Pe extends K {
  weight;
  constructor(e = null) {
    super(), this.weight = e;
  }
  forward(e, t) {
    const s = t.mul(e.log()), n = t.neg().add(1).mul(e.neg().add(1).log()), i = s.add(n).neg().mean();
    return this.weight ? i.mul(this.weight) : i;
  }
};
o(Pe, "BCELoss");
let fe = Pe;
function Qe(r) {
  return (e) => (typeof e == "number" && (e = new d(e)), new (M(r))().forward(e));
}
o(Qe, "generate_unary_function");
const Xe = Qe("relu"), Ye = Qe("sigmoid"), dt = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  relu: Xe,
  sigmoid: Ye
}, Symbol.toStringTag, { value: "Module" })), Cr = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  BCELoss: fe,
  L1Loss: _e,
  Linear: ue,
  MSELoss: le,
  Module: C,
  Parameter: P,
  ReLU: ce,
  Sequential: de,
  Sigmoid: he,
  functional: dt
}, Symbol.toStringTag, { value: "Module" })), Se = class Se {
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
o(Se, "Optimizer");
let N = Se;
const We = class We extends N {
  state = /* @__PURE__ */ new Map();
  lr;
  momentum;
  dampening;
  weight_decay;
  nesterov;
  maximize;
  constructor(e, t = 1e-3, s = 0, n = 0, i = 0, a = !1, u = !1) {
    super(e, {}), this.lr = t, this.momentum = s, this.dampening = n, this.weight_decay = i, this.nesterov = a, this.maximize = u;
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
};
o(We, "SGD");
let pe = We;
const $e = class $e extends N {
  state = /* @__PURE__ */ new Map();
  step_count = 0;
  lr;
  beta1;
  beta2;
  eps;
  weight_decay;
  amsgrad;
  maximize;
  constructor(e, t = 1e-3, s = [0.9, 0.999], n = 1e-8, i = 0, a = !1, u = !1) {
    super(e, {}), this.lr = t, this.beta1 = s[0], this.beta2 = s[1], this.eps = n, this.weight_decay = i, this.amsgrad = a, this.maximize = u;
  }
  step() {
    this.step_count += 1;
    for (const e of this.params) {
      let t = this.maximize ? e.grad.mul(-1) : e.grad;
      this.weight_decay !== 0 && (t = t.add(e.mul(this.weight_decay))), this.state.has(e) || this.state.set(e, {
        m: G(e),
        v: G(e),
        vmax: G(e)
      });
      const s = this.state.get(e);
      s.m = s.m.mul(this.beta1).add(t.mul(1 - this.beta1)), s.v = s.v.mul(this.beta2).add(t.mul(t).mul(1 - this.beta2));
      const n = 1 - Math.pow(this.beta1, this.step_count), i = 1 - Math.pow(this.beta2, this.step_count);
      let a;
      const u = s.m.div(n);
      this.amsgrad ? (s.vmax = s.vmax.maximum(s.v), a = s.vmax.div(i)) : a = s.v.div(i);
      const h = u.div(a.sqrt().add(this.eps)).mul(this.lr), c = e.sub(h);
      e.data = c.data;
    }
  }
};
o($e, "Adam");
let ge = $e;
const Dr = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Adam: ge,
  Optimizer: N,
  SGD: pe
}, Symbol.toStringTag, { value: "Module" }));
export {
  H as AccumulateGrad,
  qr as Max,
  Ar as Mean,
  vr as Min,
  yr as Sum,
  d as Tensor,
  y as TorchFunction,
  wt as __left_index__,
  xt as __right_index__,
  Tt as abs,
  bt as add,
  mt as arange,
  $t as cos,
  qt as div,
  Yt as eq,
  R as eventBus,
  F as events,
  Mt as exp,
  St as expand,
  Ot as fmod,
  Xt as ge,
  Jt as gt,
  Qt as le,
  gt as linspace,
  Rt as log,
  Ht as lt,
  Vt as matmul,
  Gt as max,
  kt as maximum,
  Nt as mean,
  zt as min,
  Et as minimum,
  At as mul,
  Zt as ne,
  Ct as neg,
  Cr as nn,
  Le as ones,
  pt as ones_like,
  Dr as optim,
  vt as pow,
  je as rand,
  ft as randint,
  _t as randn,
  Dt as reciprocal,
  Ut as reshape,
  nt as sign,
  Wt as sin,
  Ft as sqrt,
  Bt as square,
  It as squeeze,
  yt as sub,
  Kt as sum,
  jt as tan,
  Lt as transpose,
  Pt as unsqueeze,
  et as zeros,
  G as zeros_like
};
//# sourceMappingURL=torch.browser.es.js.map
