var et = Object.defineProperty;
var o = (r, e) => et(r, "name", { value: e, configurable: !0 });
function Ge(r, e) {
  const t = Math.max(r.length, e.length), s = [...Array(t - r.length).fill(1), ...r], n = [...Array(t - e.length).fill(1), ...e], a = [];
  for (let i = 0; i < t; i++) {
    if (s[i] !== n[i] && s[i] !== 1 && n[i] !== 1)
      throw new Error(`Shape mismatch: ${r} and ${e}`);
    a.push(Math.max(s[i], n[i]));
  }
  return a;
}
o(Ge, "_broadcast_shape");
function Le(r, e, t) {
  const s = S(e, r), n = new Array(e.reduce((a, i) => a * i, 1)).fill(0);
  for (let a = 0; a < t.length; a++)
    n[W(s, r, a)] += t[a];
  return n;
}
o(Le, "_unbroadcast");
function S(r, e) {
  return r.length >= e.length ? r : [...Array(e.length - r.length).fill(1), ...r];
}
o(S, "_pad_shape");
function W(r, e, t) {
  let s = 0, n = 1, a = t;
  for (let i = r.length - 1; i >= 0; i--) {
    if (r[i] > 1) {
      const h = a % e[i];
      s = s + h * n;
    }
    n *= r[i], a = Math.floor(a / e[i]);
  }
  return s;
}
o(W, "_get_original_index");
function J(r) {
  return Array.isArray(r[0]) ? r[0] : r;
}
o(J, "get_shape_from_args");
function ft(...r) {
  const e = J(r), t = new l(Array(e.reduce((s, n) => s * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
o(ft, "randn");
function je(...r) {
  const e = J(r), t = new l(Array(e.reduce((s, n) => s * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
o(je, "rand");
function pt(r, e, t) {
  const s = new l(
    Array(t.reduce((n, a) => n * a, 1)).fill(Math.floor(Math.random() * (e - r) + r))
  );
  return s.shape = t, s;
}
o(pt, "randint");
function Ve(...r) {
  const e = J(r), t = new l(Array(e.reduce((s, n) => s * n, 1)).fill(1));
  return t.shape = e, t;
}
o(Ve, "ones");
function tt(...r) {
  const e = J(r), t = new l(Array(e.reduce((s, n) => s * n, 1)).fill(0));
  return t.shape = e, t;
}
o(tt, "zeros");
function gt(r) {
  return Ve(r.shape);
}
o(gt, "ones_like");
function z(r) {
  return tt(r.shape);
}
o(z, "zeros_like");
function mt(r, e, t) {
  const s = [], n = (e - r) / (t - 1);
  for (let a = 0; a < t - 1; a++)
    s.push(r + a * n);
  return s.push(e), new l(s);
}
o(mt, "linspace");
function wt(r, e = void 0, t = 1) {
  const s = [];
  for (let n = r; n < e; n += t)
    s.push(n);
  return new l(s);
}
o(wt, "arange");
let rt = 0;
const He = /* @__PURE__ */ o(() => rt++, "getNextId"), M = new EventTarget(), B = {
  TENSOR_BEFORE_BACKWARD: "tensor.beforeBackward",
  TENSOR_AFTER_BACKWARD: "tensor.afterBackward",
  OPERATION_BEFORE_FORWARD: "operation.beforeForward",
  OPERATION_AFTER_FORWARD: "operation.afterForward",
  OPERATION_BEFORE_BACKWARD: "operation.beforeBackward",
  OPERATION_AFTER_BACKWARD: "operation.afterBackward",
  OPERATION_BEFORE_ACCUMULATE_GRAD: "operation.beforeAccumulateGrad",
  OPERATION_AFTER_ACCUMULATE_GRAD: "operation.afterAccumulateGrad"
};
function st(...r) {
  for (const e of r)
    if (e instanceof l && e.requires_grad)
      return !0;
  return !1;
}
o(st, "resultRequiresGrad");
const we = class we {
  id = He();
  next_functions = [];
  saved_tensors = [];
  _retained_tensors = [];
  forward(...e) {
    const t = st(...e);
    M.dispatchEvent(new CustomEvent(B.OPERATION_BEFORE_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e
      }
    }));
    const s = this._forward(...e);
    return M.dispatchEvent(new CustomEvent(B.OPERATION_AFTER_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e,
        result: s
      }
    })), s;
  }
  backward(e) {
    M.dispatchEvent(new CustomEvent(B.OPERATION_BEFORE_BACKWARD, { detail: { operation: this, dz: e } }));
    for (const t of this._retained_tensors)
      t.grad || (t.grad = new l(new Array(t.dataLength()).fill(0))), t.grad = t.grad.add(e);
    this._backward(e), M.dispatchEvent(new CustomEvent(B.OPERATION_AFTER_BACKWARD, { detail: { operation: this, dz: e } }));
  }
};
o(we, "TorchFunction");
let q = we;
const xe = class xe extends q {
  _forward(...e) {
    throw new Error("NullOp should not be called");
  }
  _backward(e) {
  }
};
o(xe, "NullOp");
let ee = xe;
const v = new ee(), be = class be extends q {
};
o(be, "UnaryFunction");
let G = be;
const ye = class ye extends q {
};
o(ye, "BinaryFunction");
let L = ye;
const Ae = class Ae extends G {
  variable;
  _forward(e) {
    return this.variable = e, e;
  }
  _backward(e) {
    if (this.variable.grad || (this.variable.grad = z(this.variable)), M.dispatchEvent(new CustomEvent(B.OPERATION_BEFORE_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } })), typeof e == "number")
      this.variable.grad = this.variable.grad.add(e);
    else {
      const t = Le(e.shape, this.variable.shape, e.data);
      this.variable.grad = this.variable.grad.add(new l(t, {}, { shape: this.variable.shape }));
    }
    M.dispatchEvent(new CustomEvent(B.OPERATION_AFTER_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } }));
  }
};
o(Ae, "AccumulateGrad");
let V = Ae;
const Je = /* @__PURE__ */ new Map(), Z = /* @__PURE__ */ new Map();
function k(r, e) {
  Je.set(r, e);
}
o(k, "registerOperation");
function T(r) {
  const e = Je.get(r);
  if (!e)
    throw new Error(`Operation '${r}' is not registered.`);
  return e;
}
o(T, "getOperation");
function Ke(r) {
  const e = Z.get(r);
  return e || (Z.set(r, new (T(r))()), Z.get(r));
}
o(Ke, "getOperationCache");
function nt(r) {
  if (ArrayBuffer.isView(r))
    return [r.length];
  const e = [];
  for (; Array.isArray(r); )
    e.push(r.length), r = r[0];
  return e;
}
o(nt, "_get_shape");
function Qe(r) {
  return Array.isArray(r) ? r.flatMap((e) => Qe(e)) : ArrayBuffer.isView(r) ? Array.from(r) : [r];
}
o(Qe, "_flatten");
const U = class U {
  // Auto-generated ID
  id = He();
  // Optional user-defined name
  name = null;
  data;
  shape;
  grad_fn = null;
  grad = null;
  requires_grad;
  constructor(e, t = {}, s = {}) {
    if (this.data = Qe(e), this.requires_grad = t.requires_grad ?? !1, t.name && (this.name = t.name), this.shape = s.shape ?? nt(e), this.grad_fn = s.operation ?? null, this.requires_grad && !this.grad_fn) {
      const n = new V();
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
      const a = this.shape[n], i = new Array(a), h = n === this.shape.length - 1;
      for (let u = 0; u < a; u++)
        h ? i[u] = t[e++] : i[u] = s(n + 1);
      return i;
    }, "buildDimension");
    return s(0);
  }
  dataLength() {
    return this.data.length;
  }
  _executeUnaryOp(e) {
    return (this.requires_grad ? new (T(e))() : Ke(e)).forward(this);
  }
  _executeBinaryOp(e, t) {
    return typeof t == "number" && (t = new U(t)), (this.requires_grad || t.requires_grad ? new (T(e))() : Ke(e)).forward(this, t);
  }
  _executeOpRaw(e, ...t) {
    return new (T(e))().forward(this, ...t);
  }
  item() {
    if (this.dataLength() !== 1)
      throw new Error("Tensor.item() is only valid for scalars");
    return this.data[0];
  }
  detach() {
    return new U(this.data, { requires_grad: !1 }, { shape: this.shape });
  }
  detach_() {
    this.requires_grad = !1, this.grad = null, this.grad_fn = null;
  }
  zero_() {
    this.data = Array(this.dataLength()).fill(0);
  }
  is_retain_grad = !1;
  retain_grad() {
    this.grad_fn instanceof V || this.is_retain_grad || (this.is_retain_grad = !0, this.grad_fn._retained_tensors.push(this));
  }
  backward(e) {
    if (this.requires_grad) {
      if (e)
        e.toArray_();
      else {
        if (this.dataLength() !== 1)
          throw new Error("Gradient is required for non-scalar tensors");
        e = new U(1);
      }
      this.grad_fn && (M.dispatchEvent(new CustomEvent(B.TENSOR_BEFORE_BACKWARD, { detail: { tensor: this } })), this.grad_fn.backward(e), M.dispatchEvent(new CustomEvent(B.TENSOR_AFTER_BACKWARD, { detail: { tensor: this } })));
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
o(U, "Tensor");
let l = U;
function E(r) {
  return (...e) => new (T(r))().forward(...e);
}
o(E, "generate_function$1");
function O(r) {
  return (e) => (typeof e == "number" && (e = new l(e)), new (T(r))().forward(e));
}
o(O, "generate_unary_function$1");
function m(r) {
  return (e, t) => (typeof e == "number" && (e = new l(e)), typeof t == "number" && (t = new l(t)), new (T(r))().forward(e, t));
}
o(m, "generate_binary_function$1");
const xt = m("__left_index__"), bt = m("__right_index__"), yt = m("add"), At = m("sub"), qt = m("mul"), vt = m("div"), Ot = m("pow"), kt = m("fmod"), Et = m("maximum"), Rt = m("minimum"), Ft = O("log"), Mt = O("sqrt"), Bt = O("exp"), Tt = O("square"), Ct = O("abs"), at = O("sign"), Dt = O("neg"), Ut = O("reciprocal"), It = E("reshape"), Nt = E("squeeze"), Pt = E("unsqueeze"), St = E("expand"), Wt = O("sin"), $t = O("cos"), jt = O("tan"), Kt = E("sum"), zt = E("mean"), Gt = E("min"), Lt = E("max"), Vt = E("transpose"), Ht = m("matmul"), Jt = m("lt"), Qt = m("gt"), Xt = m("le"), Yt = m("ge"), Zt = m("eq"), er = m("ne");
function ze(r) {
  const e = new Array(r.length).fill(1);
  for (let t = r.length - 2; t >= 0; t--)
    e[t] = e[t + 1] * r[t + 1];
  return e;
}
o(ze, "_get_strides");
function it(r, e) {
  return e.map((t) => {
    const s = Math.floor(r / t);
    return r %= t, s;
  });
}
o(it, "_unravel_index");
function ot(r, e) {
  return r.reduce((t, s, n) => t + s * e[n], 0);
}
o(ot, "_ravel_index");
function te(r, e, t = !1) {
  if (e === void 0) return t ? r.map(() => 1) : [];
  const n = (Array.isArray(e) ? e : [e]).map((a) => a < 0 ? a + r.length : a);
  return t ? r.map((a, i) => n.includes(i) ? 1 : a) : r.filter((a, i) => !n.includes(i));
}
o(te, "_get_reduction_shape");
function b(r, e, t = null) {
  var i;
  const s = /* @__PURE__ */ o((h, u, c, d, _, f) => {
    const p = Array(f);
    for (let w = 0; w < f; w++) {
      const x = W(u, _, w), A = W(d, _, w);
      p[w] = r(h, c, x, A);
    }
    return p;
  }, "kernel"), n = /* @__PURE__ */ o((h, u, c = null) => {
    const d = Ge(h.shape, u.shape), _ = S(h.shape, d), f = S(u.shape, d), p = d.reduce((w, x) => w * x, 1);
    return new l(
      s(h.data, _, u.data, f, d, p),
      { requires_grad: h.requires_grad || u.requires_grad },
      { operation: c, shape: d }
    );
  }, "forward_tensor"), a = (i = class extends L {
    _forward(u, c) {
      return (u.requires_grad || c.requires_grad) && (this.saved_tensors = [u, c]), this.next_functions.push(u.grad_fn ? u.grad_fn : v), this.next_functions.push(c.grad_fn ? c.grad_fn : v), n(u, c, u.requires_grad || c.requires_grad ? this : null);
    }
    _backward(u) {
      const [c, d] = this.saved_tensors, [_, f] = this.next_functions;
      e(c, d, _, f, u);
    }
  }, o(i, "result"), i);
  return t && k(t, a), a;
}
o(b, "BinaryFunctionMixin");
function y(r, e, t = null) {
  var i;
  const s = /* @__PURE__ */ o((h, u) => {
    const c = Array(u);
    for (let d = 0; d < u; d++)
      c[d] = r(h, d);
    return c;
  }, "kernel"), n = /* @__PURE__ */ o((h, u = null) => {
    const c = h.dataLength();
    return new l(
      s(h.data, c),
      { requires_grad: h.requires_grad },
      { operation: u, shape: h.shape }
    );
  }, "forward_tensor"), a = (i = class extends G {
    _forward(u) {
      return u.requires_grad && (this.saved_tensors = [u]), this.next_functions.push(u.grad_fn ? u.grad_fn : v), n(u, u.requires_grad ? this : null);
    }
    _backward(u) {
      const [c] = this.saved_tensors, [d] = this.next_functions;
      e(c, d, u);
    }
  }, o(i, "result"), i);
  return t && k(t, a), a;
}
o(y, "UnaryFunctionMixin");
function Q(r, e, t, s = null, n) {
  var i;
  const a = (i = class extends q {
    dim;
    keepdim;
    _forward(u, c, d = !1) {
      this.dim = c, this.keepdim = d, u.requires_grad && (this.saved_tensors = [u]), this.next_functions.push(u.grad_fn ? u.grad_fn : v);
      const _ = te(u.shape, c, d), f = _.reduce((g, D) => g * D, 1), p = new Array(f).fill(r), w = new Array(f).fill(0), x = ze(u.shape), A = ze(_), X = (c === void 0 ? [] : Array.isArray(c) ? c : [c]).map((g) => g < 0 ? g + u.shape.length : g), Y = c === void 0;
      for (let g = 0; g < u.data.length; g++) {
        const D = it(g, x);
        let R;
        if (Y)
          R = d ? D.map(() => 0) : [];
        else {
          R = [];
          for (let K = 0; K < u.shape.length; K++)
            X.includes(K) ? d && R.push(0) : R.push(D[K]);
        }
        const F = ot(R, A);
        p[F] = e(p[F], u.data[g]), w[F]++;
      }
      if (n)
        for (let g = 0; g < f; g++)
          p[g] = n(p[g], w[g]);
      return new l(
        p,
        { requires_grad: u.requires_grad },
        { operation: u.requires_grad ? this : null, shape: _ }
      );
    }
    _backward(u) {
      const [c] = this.saved_tensors, [d] = this.next_functions;
      let _ = u;
      const f = te(c.shape, this.dim, !0);
      u.shape.length !== f.length && (_ = u.reshape(f));
      let p = _.expand(c.shape);
      const w = t(c, p, this.dim, this.keepdim);
      d.backward(w);
    }
  }, o(i, "result"), i);
  return s && k(s, a), a;
}
o(Q, "ReductionFunctionMixin");
function P(r, e) {
  const t = Le(r.shape, e, r.data);
  return new l(t, { requires_grad: r.requires_grad }, { shape: e });
}
o(P, "unbroadcast");
function ut(r, e) {
  return r.mul(Ve(e));
}
o(ut, "broadcast");
const tr = b(
  (r, e, t, s) => t,
  (r, e, t, s, n) => {
  },
  "__left_index__"
), rr = b(
  (r, e, t, s) => s,
  (r, e, t, s, n) => {
  },
  "__right_index__"
), sr = b(
  (r, e, t, s) => r[t] + e[s],
  (r, e, t, s, n) => {
    t.backward(n), s.backward(n);
  },
  "add"
), nr = b(
  (r, e, t, s) => r[t] - e[s],
  (r, e, t, s, n) => {
    t.backward(n), s.backward(n.mul(new l(-1)));
  },
  "sub"
), ar = b(
  (r, e, t, s) => r[t] * e[s],
  (r, e, t, s, n) => {
    t.backward(n.mul(e)), s.backward(n.mul(r));
  },
  "mul"
), ir = b(
  (r, e, t, s) => r[t] / e[s],
  (r, e, t, s, n) => {
    t.backward(n.div(e)), s.backward(n.mul(r).mul(new l(-1)).div(e).div(e));
  },
  "div"
), or = b(
  (r, e, t, s) => Math.pow(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(e).mul(r.pow(e.sub(new l(1))))), s.backward(n.mul(r.pow(e)).mul(r.log()));
  },
  "pow"
), ur = b(
  (r, e, t, s) => r[t] % e[s],
  (r, e, t, s, n) => {
    t.backward(n);
  },
  "fmod"
), cr = b(
  (r, e, t, s) => Math.max(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(r.ge(e))), s.backward(n.mul(e.gt(r)));
  },
  "maximum"
), hr = b(
  (r, e, t, s) => Math.min(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(r.le(e))), s.backward(n.mul(e.lt(r)));
  },
  "minimum"
);
function ct(r, e, t = null) {
  const s = new Array(r.dataLength());
  for (let n = 0; n < s.length; n++)
    s[n] = Math.pow(r.data[n], e);
  return new l(
    s,
    { requires_grad: r.requires_grad },
    { operation: t, shape: r.shape }
  );
}
o(ct, "_powint_tensor");
const qe = class qe extends q {
  n;
  _forward(e, t) {
    return e.requires_grad && (this.saved_tensors = [e], this.n = t), this.next_functions.push(e.grad_fn ? e.grad_fn : v), ct(e, t, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, s = this.n, [n] = this.next_functions;
    n.backward(e.mul(s).mul(t.pow(s - 1)));
  }
};
o(qe, "PowInt");
let re = qe;
k("powint", re);
const dr = y(
  (r, e) => Math.log(r[e]),
  (r, e, t) => {
    e.backward(t.mul(new l(1).div(r)));
  },
  "log"
), lr = y(
  (r, e) => Math.sqrt(r[e]),
  (r, e, t) => {
    e.backward(t.mul(new l(1).div(r.sqrt()).div(2)));
  },
  "sqrt"
), _r = y(
  (r, e) => Math.exp(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.exp())));
  },
  "exp"
), fr = y(
  (r, e) => r[e] * r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(r).mul(new l(2))));
  },
  "square"
), pr = y(
  (r, e) => Math.abs(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(at(r))));
  },
  "abs"
), gr = y(
  (r, e) => Math.sign(r[e]),
  (r, e, t) => {
    e.backward(0);
  },
  "sign"
), mr = y(
  (r, e) => -r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(new l(-1))));
  },
  "neg"
), wr = y(
  (r, e) => 1 / r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.pow(-2))).neg());
  },
  "reciprocal"
), ve = class ve extends q {
  _forward(e, t) {
    const s = e.dataLength(), n = t.reduce((a, i) => a * i, 1);
    if (s !== n)
      throw new Error("Shape mismatch: " + e.shape + " and " + t);
    return e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(v), new l(
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
o(ve, "Reshape");
let se = ve;
k("reshape", se);
const Oe = class Oe extends q {
  _forward(e, t) {
    e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(v);
    let s = [...e.shape];
    return t !== void 0 ? (t < 0 && (t += e.shape.length), s[t] === 1 && s.splice(t, 1)) : s = s.filter((n) => n !== 1), new l(
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
o(Oe, "Squeeze");
let ne = Oe;
k("squeeze", ne);
const ke = class ke extends q {
  _forward(e, t) {
    e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(v), t < 0 && (t += e.shape.length + 1);
    const s = [...e.shape];
    return s.splice(t, 0, 1), new l(
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
o(ke, "Unsqueeze");
let ae = ke;
k("unsqueeze", ae);
const Ee = class Ee extends q {
  _forward(e, t) {
    e.requires_grad && (this.saved_tensors = [e]), e.grad_fn ? this.next_functions.push(e.grad_fn) : this.next_functions.push(v);
    const s = t.length - e.shape.length, n = t.map((i, h) => {
      if (i === -1) {
        const u = h - s;
        return u >= 0 ? e.shape[u] : 1;
      }
      return i;
    }), a = ut(e, n).data;
    return new l(
      a,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: n }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(P(e, t.shape));
  }
};
o(Ee, "Expand");
let ie = Ee;
k("expand", ie);
const xr = y(
  (r, e) => Math.sin(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.cos())));
  },
  "sin"
), br = y(
  (r, e) => Math.cos(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.sin().neg())));
  },
  "cos"
), yr = y(
  (r, e) => Math.tan(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.cos().pow(-2))));
  },
  "tan"
), Ar = Q(
  0,
  (r, e) => r + e,
  (r, e) => e,
  "sum"
), qr = Q(
  0,
  (r, e) => r + e,
  (r, e, t) => {
    const s = te(r.shape, t, !1), n = s.length > 0 ? s.reduce((i, h) => i * h, 1) : 1, a = r.dataLength() / n;
    return e.mul(new l([1 / a]));
  },
  "mean",
  (r, e) => r / e
), vr = Q(
  -1 / 0,
  (r, e) => Math.max(r, e),
  (r, e, t) => {
    const n = r.max(t, !0).expand(r.shape), a = r.eq(n).detach();
    return e.mul(a);
  },
  "max"
), Or = Q(
  1 / 0,
  (r, e) => Math.min(r, e),
  (r, e, t) => {
    const n = r.min(t, !0).expand(r.shape), a = r.eq(n).detach();
    return e.mul(a);
  },
  "min"
);
function ht(r, e, t, s = null) {
  if (r.shape.length + e < 0 || r.shape.length + t < 0)
    throw new Error(`Transpose: Dimension out of range (${e} and ${t})`);
  e = e < 0 ? r.shape.length + e : e, t = t < 0 ? r.shape.length + t : t;
  const n = [...r.shape];
  [n[e], n[t]] = [n[t], n[e]];
  const a = r.dataLength(), i = new Array(a), h = new Array(r.shape.length), u = new Array(n.length);
  for (let c = r.shape.length - 1, d = 1; c >= 0; c--)
    h[c] = d, d *= r.shape[c];
  for (let c = n.length - 1, d = 1; c >= 0; c--)
    u[c] = d, d *= n[c];
  for (let c = 0; c < a; c++) {
    let d = c, _ = 0;
    for (let f = 0; f < n.length; f++) {
      const p = u[f], w = Math.floor(d / p);
      d %= p;
      let x = f;
      f === e ? x = t : f === t && (x = e), _ += w * h[x];
    }
    i[c] = r.data[_];
  }
  return new l(
    i,
    { requires_grad: r.requires_grad },
    { operation: s, shape: n }
  );
}
o(ht, "_transpose_tensor");
const Re = class Re extends q {
  dim0;
  dim1;
  _forward(e, t, s) {
    return e.requires_grad && (this.saved_tensors = [e], this.dim0 = t, this.dim1 = s), this.next_functions.push(e.grad_fn ? e.grad_fn : v), ht(e, t, s, this);
  }
  _backward(e) {
    const [t] = this.saved_tensors, s = this.dim0, n = this.dim1, [a] = this.next_functions;
    a.backward(e.transpose(s, n));
  }
};
o(Re, "Transpose");
let oe = Re;
k("transpose", oe);
function dt(r, e, t = null) {
  if (r.shape.length == 1 && e.shape.length == 1)
    return [r.mul(e).sum(), []];
  const s = r.shape.length == 1, n = e.shape.length == 1, a = s ? [1, r.shape[0]] : r.shape, i = n ? [e.shape[0], 1] : e.shape;
  if (a[a.length - 1] != i[i.length - 2])
    throw new Error("Shape mismatch: " + r.shape + " and " + e.shape);
  const h = Ge(a.slice(0, -2), i.slice(0, -2)).concat([
    a[a.length - 2],
    i[i.length - 1]
  ]), u = h.reduce((A, N) => A * N, 1), c = new Array(u).fill(0), d = S(a, h), _ = S(i, h), f = h[h.length - 2], p = h[h.length - 1], w = a[a.length - 1];
  for (let A = 0; A < u; A++) {
    const N = A % (f * p), X = Math.floor(N / p), Y = N % p;
    let g = W(d, h, A - Y), D = W(_, h, A - X * p), R = 0;
    for (let F = 0; F < w; F++)
      R += r.data[g + F] * e.data[D + F * p];
    c[A] = R;
  }
  let x = [...h];
  return s && (x = x.slice(0, -2).concat([h[h.length - 1]])), n && (x = x.slice(0, -1)), [new l(
    c,
    { requires_grad: r.requires_grad || e.requires_grad },
    { operation: t, shape: x }
  ), x];
}
o(dt, "_matmul_tensor");
const Fe = class Fe extends L {
  shape;
  _forward(e, t) {
    (e.requires_grad || t.requires_grad) && (this.saved_tensors = [e, t]), this.next_functions.push(e.grad_fn ? e.grad_fn : v), this.next_functions.push(t.grad_fn ? t.grad_fn : v);
    const s = dt(e, t, e.requires_grad || t.requires_grad ? this : null);
    return this.shape = s[1], s[0];
  }
  _backward(e) {
    const [t, s] = this.saved_tensors, [n, a] = this.next_functions;
    if (t.shape.length === 1 && s.shape.length === 1) {
      n.backward(e.mul(s)), a.backward(e.mul(t));
      return;
    }
    if (t.shape.length === 1) {
      const u = e.unsqueeze(-2), c = t.unsqueeze(-2);
      let d = u.matmul(s.transpose(-2, -1)), _ = c.transpose(-2, -1).matmul(u);
      d = d.squeeze(-2), _ = P(_, s.shape), n.backward(d), a.backward(_);
      return;
    }
    if (s.shape.length === 1) {
      const u = e.unsqueeze(-1), c = s.unsqueeze(-1);
      let d = u.matmul(c.transpose(-2, -1)), _ = t.transpose(-2, -1).matmul(u);
      d = P(d, t.shape), _ = _.squeeze(-1), n.backward(d), a.backward(_);
      return;
    }
    let i = e.matmul(s.transpose(-2, -1)), h = t.transpose(-2, -1).matmul(e);
    i = P(i, t.shape), h = P(h, s.shape), n.backward(i), a.backward(h);
  }
};
o(Fe, "Matmul");
let ue = Fe;
k("matmul", ue);
const kr = b(
  (r, e, t, s) => r[t] < e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "lt"
), Er = b(
  (r, e, t, s) => r[t] > e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "gt"
), Rr = b(
  (r, e, t, s) => r[t] <= e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "le"
), Fr = b(
  (r, e, t, s) => r[t] >= e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "ge"
), Mr = b(
  (r, e, t, s) => r[t] == e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "eq"
), Br = b(
  (r, e, t, s) => r[t] != e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "ne"
), Tr = y(
  (r, e) => Math.max(r[e], 0),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.gt(0))));
  },
  "relu"
), Cr = y(
  (r, e) => 1 / (1 + Math.exp(-r[e])),
  (r, e, t) => {
    const s = r.sigmoid();
    e.backward(s.mul(s.mul(-1).add(1)).mul(t));
  },
  "sigmoid"
), H = class H extends l {
  constructor(e, t = {
    requires_grad: !0
  }, s = {}) {
    e instanceof l ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : e instanceof H ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : super(e, t, s);
  }
};
o(H, "Parameter");
let I = H;
const Me = class Me {
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
    t instanceof I ? this.register_parameter(e, t) : this.register_module(e, t);
  }
  parameters() {
    let e = Object.values(this._parameters);
    for (const t of Object.values(this._modules))
      e = e.concat(t.parameters());
    return e;
  }
};
o(Me, "Module");
let C = Me;
const Be = class Be extends C {
  weight;
  bias;
  constructor(e, t) {
    super();
    const s = Math.sqrt(1 / e);
    this.weight = new I(
      je([t, e]).mul(2 * s).sub(s)
    ), this.bias = new I(
      je([t]).mul(2 * s).sub(s)
    ), this.register("weight", this.weight), this.register("bias", this.bias);
  }
  forward(e) {
    return e.matmul(this.weight.transpose(0, 1)).add(this.bias);
  }
};
o(Be, "Linear");
let ce = Be;
const Te = class Te extends C {
  constructor() {
    super();
  }
  forward(e) {
    return Ye(e);
  }
};
o(Te, "ReLU");
let he = Te;
const Ce = class Ce extends C {
  constructor() {
    super();
  }
  forward(e) {
    return Ze(e);
  }
};
o(Ce, "Sigmoid");
let de = Ce;
const De = class De extends C {
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
o(De, "Sequential");
let le = De;
const Ue = class Ue {
};
o(Ue, "Loss");
let $ = Ue;
const Ie = class Ie extends $ {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).pow(2).mean();
  }
};
o(Ie, "MSELoss");
let _e = Ie;
const Ne = class Ne extends $ {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).abs().mean();
  }
};
o(Ne, "L1Loss");
let fe = Ne;
const Pe = class Pe extends $ {
  weight;
  constructor(e = null) {
    super(), this.weight = e;
  }
  forward(e, t) {
    const s = t.mul(e.log()), n = t.neg().add(1).mul(e.neg().add(1).log()), a = s.add(n).neg().mean();
    return this.weight ? a.mul(this.weight) : a;
  }
};
o(Pe, "BCELoss");
let pe = Pe;
function Xe(r) {
  return (e) => (typeof e == "number" && (e = new l(e)), new (T(r))().forward(e));
}
o(Xe, "generate_unary_function");
const Ye = Xe("relu"), Ze = Xe("sigmoid"), lt = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  relu: Ye,
  sigmoid: Ze
}, Symbol.toStringTag, { value: "Module" })), Dr = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  BCELoss: pe,
  L1Loss: fe,
  Linear: ce,
  MSELoss: _e,
  Module: C,
  Parameter: I,
  ReLU: he,
  Sequential: le,
  Sigmoid: de,
  functional: lt
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
let j = Se;
const We = class We extends j {
  state = /* @__PURE__ */ new Map();
  lr;
  momentum;
  dampening;
  weight_decay;
  nesterov;
  maximize;
  constructor(e, t = 1e-3, s = 0, n = 0, a = 0, i = !1, h = !1) {
    super(e, {}), this.lr = t, this.momentum = s, this.dampening = n, this.weight_decay = a, this.nesterov = i, this.maximize = h;
  }
  step() {
    for (const e of this.params) {
      let t = this.maximize ? e.grad.mul(-1) : e.grad;
      if (this.weight_decay !== 0 && (t = t.add(e.mul(this.weight_decay))), this.momentum !== 0) {
        if (this.state.has(e)) {
          let a = this.state.get(e).velocity;
          a = a.mul(this.momentum), a = a.add(t.mul(1 - this.dampening)), this.state.set(e, { velocity: a });
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
let ge = We;
const $e = class $e extends j {
  state = /* @__PURE__ */ new Map();
  step_count = 0;
  lr;
  beta1;
  beta2;
  eps;
  weight_decay;
  amsgrad;
  maximize;
  constructor(e, t = 1e-3, s = [0.9, 0.999], n = 1e-8, a = 0, i = !1, h = !1) {
    super(e, {}), this.lr = t, this.beta1 = s[0], this.beta2 = s[1], this.eps = n, this.weight_decay = a, this.amsgrad = i, this.maximize = h;
  }
  step() {
    this.step_count += 1;
    for (const e of this.params) {
      let t = this.maximize ? e.grad.mul(-1) : e.grad;
      this.weight_decay !== 0 && (t = t.add(e.mul(this.weight_decay))), this.state.has(e) || this.state.set(e, {
        m: z(e),
        v: z(e),
        vmax: z(e)
      });
      const s = this.state.get(e);
      s.m = s.m.mul(this.beta1).add(t.mul(1 - this.beta1)), s.v = s.v.mul(this.beta2).add(t.mul(t).mul(1 - this.beta2));
      const n = 1 - Math.pow(this.beta1, this.step_count), a = 1 - Math.pow(this.beta2, this.step_count);
      let i;
      const h = s.m.div(n);
      this.amsgrad ? (s.vmax = s.vmax.maximum(s.v), i = s.vmax.div(a)) : i = s.v.div(a);
      const u = h.div(i.sqrt().add(this.eps)).mul(this.lr), c = e.sub(u);
      e.data = c.data;
    }
  }
};
o($e, "Adam");
let me = $e;
const Ur = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Adam: me,
  Optimizer: j,
  SGD: ge
}, Symbol.toStringTag, { value: "Module" }));
export {
  V as AccumulateGrad,
  vr as Max,
  qr as Mean,
  Or as Min,
  Ar as Sum,
  l as Tensor,
  q as TorchFunction,
  xt as __left_index__,
  bt as __right_index__,
  Ct as abs,
  yt as add,
  wt as arange,
  $t as cos,
  vt as div,
  Zt as eq,
  M as eventBus,
  B as events,
  Bt as exp,
  St as expand,
  kt as fmod,
  Yt as ge,
  Qt as gt,
  Xt as le,
  mt as linspace,
  Ft as log,
  Jt as lt,
  Ht as matmul,
  Lt as max,
  Et as maximum,
  zt as mean,
  Gt as min,
  Rt as minimum,
  qt as mul,
  er as ne,
  Dt as neg,
  Dr as nn,
  Ve as ones,
  gt as ones_like,
  Ur as optim,
  Ot as pow,
  je as rand,
  pt as randint,
  ft as randn,
  Ut as reciprocal,
  It as reshape,
  at as sign,
  Wt as sin,
  Mt as sqrt,
  Tt as square,
  Nt as squeeze,
  At as sub,
  Kt as sum,
  jt as tan,
  Vt as transpose,
  Pt as unsqueeze,
  tt as zeros,
  z as zeros_like
};
//# sourceMappingURL=torch.browser.es.js.map
