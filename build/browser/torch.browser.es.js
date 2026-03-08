var ze = Object.defineProperty;
var a = (r, e) => ze(r, "name", { value: e, configurable: !0 });
function K(r) {
  return Array.isArray(r[0]) ? r[0] : r;
}
a(K, "get_shape_from_args");
function ot(...r) {
  const e = K(r), t = new h(Array(e.reduce((s, n) => s * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
a(ot, "randn");
function Ue(...r) {
  const e = K(r), t = new h(Array(e.reduce((s, n) => s * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
a(Ue, "rand");
function ut(r, e, t) {
  const s = new h(
    Array(t.reduce((n, i) => n * i, 1)).fill(Math.floor(Math.random() * (e - r) + r))
  );
  return s.shape = t, s;
}
a(ut, "randint");
function He(...r) {
  const e = K(r), t = new h(Array(e.reduce((s, n) => s * n, 1)).fill(1));
  return t.shape = e, t;
}
a(He, "ones");
function Je(...r) {
  const e = K(r), t = new h(Array(e.reduce((s, n) => s * n, 1)).fill(0));
  return t.shape = e, t;
}
a(Je, "zeros");
function ct(r) {
  return He(r.shape);
}
a(ct, "ones_like");
function B(r) {
  return Je(r.shape);
}
a(B, "zeros_like");
function dt(r, e, t) {
  const s = [], n = (e - r) / (t - 1);
  for (let i = 0; i < t - 1; i++)
    s.push(r + i * n);
  return s.push(e), new h(s);
}
a(dt, "linspace");
function ht(r, e = void 0, t = 1) {
  const s = [];
  for (let n = r; n < e; n += t)
    s.push(n);
  return new h(s);
}
a(ht, "arange");
let Qe = 0;
const Ne = /* @__PURE__ */ a(() => Qe++, "getNextId"), k = new EventTarget(), E = {
  TENSOR_BEFORE_BACKWARD: "tensor.beforeBackward",
  TENSOR_AFTER_BACKWARD: "tensor.afterBackward",
  OPERATION_BEFORE_FORWARD: "operation.beforeForward",
  OPERATION_AFTER_FORWARD: "operation.afterForward",
  OPERATION_BEFORE_BACKWARD: "operation.beforeBackward",
  OPERATION_AFTER_BACKWARD: "operation.afterBackward",
  OPERATION_BEFORE_ACCUMULATE_GRAD: "operation.beforeAccumulateGrad",
  OPERATION_AFTER_ACCUMULATE_GRAD: "operation.afterAccumulateGrad"
};
function Xe(...r) {
  for (const e of r)
    if (e instanceof h && e.requires_grad)
      return !0;
  return !1;
}
a(Xe, "resultRequiresGrad");
const ce = class ce {
  id = Ne();
  next_functions = [];
  saved_tensors = [];
  _retained_tensors = [];
  forward(...e) {
    const t = Xe(...e);
    k.dispatchEvent(new CustomEvent(E.OPERATION_BEFORE_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e
      }
    }));
    const s = this._forward(...e);
    return k.dispatchEvent(new CustomEvent(E.OPERATION_AFTER_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e,
        result: s
      }
    })), s;
  }
  backward(e) {
    k.dispatchEvent(new CustomEvent(E.OPERATION_BEFORE_BACKWARD, { detail: { operation: this, dz: e } }));
    for (const t of this._retained_tensors)
      t.grad || (t.grad = new h(new Array(t.dataLength()).fill(0))), t.grad = t.grad.add(e);
    this._backward(e), k.dispatchEvent(new CustomEvent(E.OPERATION_AFTER_BACKWARD, { detail: { operation: this, dz: e } }));
  }
};
a(ce, "TorchFunction");
let q = ce;
const de = class de extends q {
  _forward(...e) {
    throw new Error("NullOp should not be called");
  }
  _backward(e) {
  }
};
a(de, "NullOp");
let L = de;
const A = new L(), he = class he extends q {
};
a(he, "UnaryFunction");
let T = he;
const le = class le extends q {
};
a(le, "BinaryFunction");
let S = le;
const _e = class _e extends T {
  variable;
  _forward(e) {
    return this.variable = e, e;
  }
  _backward(e) {
    this.variable.grad || (this.variable.grad = B(this.variable)), k.dispatchEvent(new CustomEvent(E.OPERATION_BEFORE_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } })), this.variable.grad = this.variable.grad.add(e), k.dispatchEvent(new CustomEvent(E.OPERATION_AFTER_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } }));
  }
};
a(_e, "AccumulateGrad");
let C = _e;
const Pe = /* @__PURE__ */ new Map(), G = /* @__PURE__ */ new Map();
function O(r, e) {
  Pe.set(r, e);
}
a(O, "registerOperation");
function R(r) {
  const e = Pe.get(r);
  if (!e)
    throw new Error(`Operation '${r}' is not registered.`);
  return e;
}
a(R, "getOperation");
function De(r) {
  const e = G.get(r);
  return e || (G.set(r, new (R(r))()), G.get(r));
}
a(De, "getOperationCache");
function Ye(r) {
  if (ArrayBuffer.isView(r))
    return [r.length];
  const e = [];
  for (; Array.isArray(r); )
    e.push(r.length), r = r[0];
  return e;
}
a(Ye, "_get_shape");
function Ie(r) {
  return Array.isArray(r) ? r.flatMap((e) => Ie(e)) : ArrayBuffer.isView(r) ? Array.from(r) : [r];
}
a(Ie, "_flatten");
const M = class M {
  // Auto-generated ID
  id = Ne();
  // Optional user-defined name
  name = null;
  data;
  _shape;
  grad_fn = null;
  grad = null;
  requires_grad;
  constructor(e, t = {}, s = {}) {
    if (this.data = Ie(e), this.requires_grad = t.requires_grad ?? !1, t.name && (this.name = t.name), this._shape = s.shape ?? Ye(e), this.grad_fn = s.operation ?? null, this.requires_grad && !this.grad_fn) {
      const n = new C();
      n.variable = this, this.grad_fn = n;
    }
  }
  // TODO: Somehow having a shape of [] will have a weird error:
  // TypeError: Cannot read properties of undefined (reading 'length')
  // when running kernel (something to do with constants?)
  // so a little hack to return [1] when the shape is []
  get shape() {
    return this._shape.length === 0 ? [1] : this._shape;
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
    const t = this.data, s = /* @__PURE__ */ a((n) => {
      const i = this.shape[n], o = new Array(i), u = n === this.shape.length - 1;
      for (let c = 0; c < i; c++)
        u ? o[c] = t[e++] : o[c] = s(n + 1);
      return o;
    }, "buildDimension");
    return s(0);
  }
  dataLength() {
    return this.data.length;
  }
  set shape(e) {
    this._shape = e;
  }
  _executeUnaryOp(e) {
    return (this.requires_grad ? new (R(e))() : De(e)).forward(this);
  }
  _executeBinaryOp(e, t) {
    return typeof t == "number" && (t = new M(t)), (this.requires_grad || t.requires_grad ? new (R(e))() : De(e)).forward(this, t);
  }
  _executeOpRaw(e, ...t) {
    return new (R(e))().forward(this, ...t);
  }
  item() {
    if (this.dataLength() !== 1)
      throw new Error("Tensor.item() is only valid for scalars");
    return this.toArray()[0];
  }
  detach() {
    return new M(this.data, { requires_grad: !1 }, { shape: this.shape });
  }
  detach_() {
    this.requires_grad = !1, this.grad = null, this.grad_fn = null;
  }
  zero_() {
    this.data = Array(this.dataLength()).fill(0);
  }
  is_retain_grad = !1;
  retain_grad() {
    this.grad_fn instanceof C || this.is_retain_grad || (this.is_retain_grad = !0, this.grad_fn._retained_tensors.push(this));
  }
  backward(e) {
    if (this.requires_grad) {
      if (e)
        e.toArray_();
      else {
        if (this.dataLength() !== 1)
          throw new Error("Gradient is required for non-scalar tensors");
        e = new M(1);
      }
      this.grad_fn && (k.dispatchEvent(new CustomEvent(E.TENSOR_BEFORE_BACKWARD, { detail: { tensor: this } })), this.grad_fn.backward(e), k.dispatchEvent(new CustomEvent(E.TENSOR_AFTER_BACKWARD, { detail: { tensor: this } })));
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
  unsqueeze(e) {
    return this._executeOpRaw("unsqueeze", e);
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
  sum() {
    return this._executeUnaryOp("sum");
  }
  mean() {
    return this._executeUnaryOp("mean");
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
a(M, "Tensor");
let h = M;
function Se(r, e) {
  const t = Math.max(r.length, e.length), s = [...Array(t - r.length).fill(1), ...r], n = [...Array(t - e.length).fill(1), ...e], i = [];
  for (let o = 0; o < t; o++) {
    if (s[o] !== n[o] && s[o] !== 1 && n[o] !== 1)
      throw new Error(`Shape mismatch: ${r} and ${e}`);
    i.push(Math.max(s[o], n[o]));
  }
  return i;
}
a(Se, "_broadcast_shape");
function W(r, e) {
  return r.length >= e.length ? r : [...Array(e.length - r.length).fill(1), ...r];
}
a(W, "_pad_shape");
function $(r, e, t) {
  let s = 0, n = 1, i = t;
  for (let o = r.length - 1; o >= 0; o--) {
    if (r[o] > 1) {
      const u = i % e[o];
      s = s + u * n;
    }
    n *= r[o], i = Math.floor(i / e[o]);
  }
  return s;
}
a($, "_get_original_index");
function ue(r) {
  return (...e) => new (R(r))().forward(...e);
}
a(ue, "generate_function$1");
function w(r) {
  return (e) => (typeof e == "number" && (e = new h(e)), new (R(r))().forward(e));
}
a(w, "generate_unary_function$1");
function _(r) {
  return (e, t) => (typeof e == "number" && (e = new h(e)), typeof t == "number" && (t = new h(t)), new (R(r))().forward(e, t));
}
a(_, "generate_binary_function$1");
const lt = _("__left_index__"), _t = _("__right_index__"), ft = _("add"), pt = _("sub"), gt = _("mul"), mt = _("div"), wt = _("pow"), xt = _("fmod"), bt = _("maximum"), yt = _("minimum"), At = w("log"), vt = w("sqrt"), qt = w("exp"), Ot = w("square"), kt = w("abs"), Ze = w("sign"), Et = w("neg"), Rt = w("reciprocal"), Ft = ue("reshape"), Mt = ue("unsqueeze"), Bt = w("sin"), Tt = w("cos"), Ct = w("tan"), Ut = w("sum"), Dt = w("mean"), Nt = ue("transpose"), Pt = _("matmul"), It = _("lt"), St = _("gt"), Wt = _("le"), $t = _("ge"), jt = _("eq"), Kt = _("ne");
function f(r, e, t = null) {
  var o;
  const s = /* @__PURE__ */ a((u, c, d, l, b, p) => {
    const g = Array(p);
    for (let y = 0; y < p; y++) {
      const m = $(c, b, y), v = $(l, b, y);
      g[y] = r(u, d, m, v);
    }
    return g;
  }, "kernel"), n = /* @__PURE__ */ a((u, c, d = null) => {
    const l = Se(u.shape, c.shape), b = W(u.shape, l), p = W(c.shape, l), g = l.reduce((y, m) => y * m, 1);
    return new h(
      s(u.data, b, c.data, p, l, g),
      { requires_grad: u.requires_grad || c.requires_grad },
      { operation: d, shape: l }
    );
  }, "forward_tensor"), i = (o = class extends S {
    _forward(c, d) {
      return (c.requires_grad || d.requires_grad) && (this.saved_tensors = [c, d]), this.next_functions.push(c.grad_fn ? c.grad_fn : A), this.next_functions.push(d.grad_fn ? d.grad_fn : A), n(c, d, c.requires_grad || d.requires_grad ? this : null);
    }
    _backward(c) {
      const [d, l] = this.saved_tensors, [b, p] = this.next_functions;
      e(d, l, b, p, c);
    }
  }, a(o, "result"), o);
  return t && O(t, i), i;
}
a(f, "BinaryFunctionMixin");
function x(r, e, t = null) {
  var o;
  const s = /* @__PURE__ */ a((u, c) => {
    const d = Array(c);
    for (let l = 0; l < c; l++)
      d[l] = r(u, l);
    return d;
  }, "kernel"), n = /* @__PURE__ */ a((u, c = null) => {
    const d = u.dataLength();
    return new h(
      s(u.data, d),
      { requires_grad: u.requires_grad },
      { operation: c, shape: u.shape }
    );
  }, "forward_tensor"), i = (o = class extends T {
    _forward(c) {
      return c.requires_grad && (this.saved_tensors = [c]), this.next_functions.push(c.grad_fn ? c.grad_fn : A), n(c, c.requires_grad ? this : null);
    }
    _backward(c) {
      const [d] = this.saved_tensors, [l] = this.next_functions;
      e(d, l, c);
    }
  }, a(o, "result"), o);
  return t && O(t, i), i;
}
a(x, "UnaryFunctionMixin");
const Gt = f(
  (r, e, t, s) => t,
  (r, e, t, s, n) => {
  },
  "__left_index__"
), Lt = f(
  (r, e, t, s) => s,
  (r, e, t, s, n) => {
  },
  "__right_index__"
), Vt = f(
  (r, e, t, s) => r[t] + e[s],
  (r, e, t, s, n) => {
    t.backward(n), s.backward(n);
  },
  "add"
), zt = f(
  (r, e, t, s) => r[t] - e[s],
  (r, e, t, s, n) => {
    t.backward(n), s.backward(n.mul(new h(-1)));
  },
  "sub"
), Ht = f(
  (r, e, t, s) => r[t] * e[s],
  (r, e, t, s, n) => {
    t.backward(n.mul(e)), s.backward(n.mul(r));
  },
  "mul"
), Jt = f(
  (r, e, t, s) => r[t] / e[s],
  (r, e, t, s, n) => {
    t.backward(n.div(e)), s.backward(n.mul(r).mul(new h(-1)).div(e).div(e));
  },
  "div"
), Qt = f(
  (r, e, t, s) => Math.pow(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(e).mul(r.pow(e.sub(new h(1))))), s.backward(n.mul(r.pow(e)).mul(r.log()));
  },
  "pow"
), Xt = f(
  (r, e, t, s) => r[t] % e[s],
  (r, e, t, s, n) => {
    t.backward(n);
  },
  "fmod"
), Yt = f(
  (r, e, t, s) => Math.max(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(r.ge(e))), s.backward(n.mul(e.gt(r)));
  },
  "maximum"
), Zt = f(
  (r, e, t, s) => Math.min(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(r.le(e))), s.backward(n.mul(e.lt(r)));
  },
  "minimum"
);
function et(r, e, t = null) {
  const s = new Array(r.dataLength());
  for (let n = 0; n < s.length; n++)
    s[n] = Math.pow(r.data[n], e);
  return new h(
    s,
    { requires_grad: r.requires_grad },
    { operation: t, shape: r.shape }
  );
}
a(et, "_powint_tensor");
const fe = class fe extends q {
  n;
  _forward(e, t) {
    return e.requires_grad && (this.saved_tensors = [e], this.n = t), this.next_functions.push(e.grad_fn ? e.grad_fn : A), et(e, t, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, s = this.n, [n] = this.next_functions;
    n.backward(e.mul(s).mul(t.pow(s - 1)));
  }
};
a(fe, "PowInt");
let V = fe;
O("powint", V);
const er = x(
  (r, e) => Math.log(r[e]),
  (r, e, t) => {
    e.backward(t.mul(new h(1).div(r)));
  },
  "log"
), tr = x(
  (r, e) => Math.sqrt(r[e]),
  (r, e, t) => {
    e.backward(t.mul(new h(1).div(r.sqrt()).div(2)));
  },
  "sqrt"
), rr = x(
  (r, e) => Math.exp(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.exp())));
  },
  "exp"
), sr = x(
  (r, e) => r[e] * r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(r).mul(new h(2))));
  },
  "square"
), nr = x(
  (r, e) => Math.abs(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(Ze(r))));
  },
  "abs"
), ar = x(
  (r, e) => Math.sign(r[e]),
  (r, e, t) => {
    e.backward(0);
  },
  "sign"
), ir = x(
  (r, e) => -r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(new h(-1))));
  },
  "neg"
), or = x(
  (r, e) => 1 / r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.pow(-2))).neg());
  },
  "reciprocal"
), pe = class pe extends q {
  _forward(e, t) {
    const s = e.dataLength(), n = t.reduce((i, o) => i * o, 1);
    if (s !== n)
      throw new Error("Shape mismatch: " + e.shape + " and " + t);
    if (e.requires_grad && (this.saved_tensors = [e]), e.grad_fn)
      this.next_functions.push(e.grad_fn);
    else if (e.requires_grad) {
      const i = new C();
      i.variable = e, this.next_functions.push(i);
    } else
      this.next_functions.push(A);
    return new h(
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
a(pe, "Reshape");
let z = pe;
O("reshape", z);
const ge = class ge extends q {
  _forward(e, t) {
    if (e.requires_grad && (this.saved_tensors = [e]), e.grad_fn)
      this.next_functions.push(e.grad_fn);
    else if (e.requires_grad) {
      const n = new C();
      n.variable = e, this.next_functions.push(n);
    } else
      this.next_functions.push(A);
    t < 0 && (t += e.shape.length + 1);
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
};
a(ge, "Unsqueeze");
let H = ge;
O("unsqueeze", H);
const ur = x(
  (r, e) => Math.sin(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.cos())));
  },
  "sin"
), cr = x(
  (r, e) => Math.cos(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.sin().neg())));
  },
  "cos"
), dr = x(
  (r, e) => Math.tan(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.cos().pow(-2))));
  },
  "tan"
);
function tt(r, e = null) {
  return new h(
    r.toFlatArray().reduce((t, s) => t + s, 0),
    { requires_grad: r.requires_grad },
    { operation: e }
  );
}
a(tt, "_sum_tensor");
const me = class me extends T {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : A), tt(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(B(t).add(e.item()));
  }
};
a(me, "Sum");
let J = me;
O("sum", J);
function rt(r, e = null) {
  return new h(
    r.toFlatArray().reduce((t, s) => t + s, 0) / r.dataLength(),
    { requires_grad: r.requires_grad },
    { operation: e }
  );
}
a(rt, "_mean_tensor");
const we = class we extends T {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : A), rt(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(B(t).add(e.item() / t.dataLength()));
  }
};
a(we, "Mean");
let Q = we;
O("mean", Q);
function st(r, e, t, s = null) {
  if (r.shape.length + e < 0 || r.shape.length + t < 0)
    throw new Error(`Transpose: Dimension out of range (${e} and ${t})`);
  e = e < 0 ? r.shape.length + e : e, t = t < 0 ? r.shape.length + t : t;
  const n = [...r.shape];
  [n[e], n[t]] = [n[t], n[e]];
  const i = r.dataLength(), o = new Array(i), u = new Array(r.shape.length), c = new Array(n.length);
  for (let d = r.shape.length - 1, l = 1; d >= 0; d--)
    u[d] = l, l *= r.shape[d];
  for (let d = n.length - 1, l = 1; d >= 0; d--)
    c[d] = l, l *= n[d];
  for (let d = 0; d < i; d++) {
    let l = d, b = 0;
    for (let p = 0; p < n.length; p++) {
      const g = c[p], y = Math.floor(l / g);
      l %= g;
      let m = p;
      p === e ? m = t : p === t && (m = e), b += y * u[m];
    }
    o[d] = r.data[b];
  }
  return new h(
    o,
    { requires_grad: r.requires_grad },
    { operation: s, shape: n }
  );
}
a(st, "_transpose_tensor");
const xe = class xe extends q {
  dim0;
  dim1;
  _forward(e, t, s) {
    return e.requires_grad && (this.saved_tensors = [e], this.dim0 = t, this.dim1 = s), this.next_functions.push(e.grad_fn ? e.grad_fn : A), st(e, t, s, this);
  }
  _backward(e) {
    const [t] = this.saved_tensors, s = this.dim0, n = this.dim1, [i] = this.next_functions;
    i.backward(e.transpose(s, n));
  }
};
a(xe, "Transpose");
let X = xe;
O("transpose", X);
function nt(r, e, t = null) {
  if (r.shape.length == 1 && e.shape.length == 1)
    return r.mul(e).sum();
  const s = r.shape.length == 1, n = e.shape.length == 1, i = s ? [1, r.shape[0]] : r.shape, o = n ? [e.shape[0], 1] : e.shape;
  if (i[i.length - 1] != o[o.length - 2])
    throw new Error("Shape mismatch: " + r.shape + " and " + e.shape);
  const u = Se(i.slice(0, -2), o.slice(0, -2)).concat([
    i[i.length - 2],
    o[o.length - 1]
  ]), c = u.reduce((v, P) => v * P, 1), d = new Array(c).fill(0), l = W(i, u), b = W(o, u), p = u[u.length - 2], g = u[u.length - 1], y = i[i.length - 1];
  for (let v = 0; v < c; v++) {
    const P = v % (p * g), Ke = Math.floor(P / g), Ge = P % g;
    let Le = $(l, u, v - Ge), Ve = $(b, u, v - Ke * g), Ce = 0;
    for (let I = 0; I < y; I++)
      Ce += r.data[Le + I] * e.data[Ve + I * g];
    d[v] = Ce;
  }
  let m = [...u];
  return s && (m = m.slice(0, -2).concat([u[u.length - 1]])), n && (m = m.slice(0, -1)), new h(
    d,
    { requires_grad: r.requires_grad || e.requires_grad },
    { operation: t, shape: m }
  );
}
a(nt, "_matmul_tensor");
const be = class be extends S {
  _forward(e, t) {
    return (e.requires_grad || t.requires_grad) && (this.saved_tensors = [e, t]), this.next_functions.push(e.grad_fn ? e.grad_fn : A), this.next_functions.push(t.grad_fn ? t.grad_fn : A), nt(e, t, e.requires_grad || t.requires_grad ? this : null);
  }
  _backward(e) {
    const [t, s] = this.saved_tensors, [n, i] = this.next_functions;
    if (t.shape.length == 1 && s.shape.length == 1) {
      n.backward(e), i.backward(e);
      return;
    }
    if (t.shape.length == 1) {
      const o = e.unsqueeze(0), u = t.unsqueeze(0);
      n.backward(o.matmul(s.transpose(-2, -1))), i.backward(u.transpose(0, 1).matmul(o));
      return;
    }
    if (s.shape.length == 1) {
      const o = e.unsqueeze(0), u = s.unsqueeze(1);
      n.backward(o.matmul(u.transpose(0, 1))), i.backward(t.transpose(-2, -1).matmul(o));
      return;
    }
    n.backward(e.matmul(s.transpose(-2, -1))), i.backward(t.transpose(-2, -1).matmul(e));
  }
};
a(be, "Matmul");
let Y = be;
O("matmul", Y);
const hr = f(
  (r, e, t, s) => r[t] < e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "lt"
), lr = f(
  (r, e, t, s) => r[t] > e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "gt"
), _r = f(
  (r, e, t, s) => r[t] <= e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "le"
), fr = f(
  (r, e, t, s) => r[t] >= e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "ge"
), pr = f(
  (r, e, t, s) => r[t] == e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "eq"
), gr = f(
  (r, e, t, s) => r[t] != e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "ne"
), mr = x(
  (r, e) => Math.max(r[e], 0),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.gt(0))));
  },
  "relu"
), wr = x(
  (r, e) => 1 / (1 + Math.exp(-r[e])),
  (r, e, t) => {
    const s = r.sigmoid();
    e.backward(s.mul(s.mul(-1).add(1)).mul(t));
  },
  "sigmoid"
), j = class j extends h {
  constructor(e, t = {
    requires_grad: !0
  }, s = {}) {
    e instanceof h ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : e instanceof j ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : super(e, t, s);
  }
};
a(j, "Parameter");
let U = j;
const ye = class ye {
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
};
a(ye, "Module");
let F = ye;
const Ae = class Ae extends F {
  weight;
  bias;
  constructor(e, t) {
    super();
    const s = Math.sqrt(1 / e);
    this.weight = new U(
      Ue([t, e]).mul(2 * s).sub(s)
    ), this.bias = new U(
      Ue([t]).mul(2 * s).sub(s)
    ), this.register("weight", this.weight), this.register("bias", this.bias);
  }
  forward(e) {
    return e.matmul(this.weight.transpose(0, 1)).add(this.bias);
  }
};
a(Ae, "Linear");
let Z = Ae;
const ve = class ve extends F {
  constructor() {
    super();
  }
  forward(e) {
    return $e(e);
  }
};
a(ve, "ReLU");
let ee = ve;
const qe = class qe extends F {
  constructor() {
    super();
  }
  forward(e) {
    return je(e);
  }
};
a(qe, "Sigmoid");
let te = qe;
const Oe = class Oe extends F {
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
a(Oe, "Sequential");
let re = Oe;
const ke = class ke {
};
a(ke, "Loss");
let D = ke;
const Ee = class Ee extends D {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).pow(2).mean();
  }
};
a(Ee, "MSELoss");
let se = Ee;
const Re = class Re extends D {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).abs().mean();
  }
};
a(Re, "L1Loss");
let ne = Re;
const Fe = class Fe extends D {
  weight;
  constructor(e = null) {
    super(), this.weight = e;
  }
  forward(e, t) {
    const s = t.mul(e.log()), n = t.neg().add(1).mul(e.neg().add(1).log()), i = s.add(n).neg().mean();
    return this.weight ? i.mul(this.weight) : i;
  }
};
a(Fe, "BCELoss");
let ae = Fe;
function We(r) {
  return (e) => (typeof e == "number" && (e = new h(e)), new (R(r))().forward(e));
}
a(We, "generate_unary_function");
const $e = We("relu"), je = We("sigmoid"), at = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  relu: $e,
  sigmoid: je
}, Symbol.toStringTag, { value: "Module" })), xr = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  BCELoss: ae,
  L1Loss: ne,
  Linear: Z,
  MSELoss: se,
  Module: F,
  Parameter: U,
  ReLU: ee,
  Sequential: re,
  Sigmoid: te,
  functional: at
}, Symbol.toStringTag, { value: "Module" })), Me = class Me {
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
a(Me, "Optimizer");
let N = Me;
const Be = class Be extends N {
  state = /* @__PURE__ */ new Map();
  lr;
  momentum;
  dampening;
  weight_decay;
  nesterov;
  maximize;
  constructor(e, t = 1e-3, s = 0, n = 0, i = 0, o = !1, u = !1) {
    super(e, {}), this.lr = t, this.momentum = s, this.dampening = n, this.weight_decay = i, this.nesterov = o, this.maximize = u;
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
a(Be, "SGD");
let ie = Be;
const Te = class Te extends N {
  state = /* @__PURE__ */ new Map();
  step_count = 0;
  lr;
  beta1;
  beta2;
  eps;
  weight_decay;
  amsgrad;
  maximize;
  constructor(e, t = 1e-3, s = [0.9, 0.999], n = 1e-8, i = 0, o = !1, u = !1) {
    super(e, {}), this.lr = t, this.beta1 = s[0], this.beta2 = s[1], this.eps = n, this.weight_decay = i, this.amsgrad = o, this.maximize = u;
  }
  step() {
    this.step_count += 1;
    for (const e of this.params) {
      let t = this.maximize ? e.grad.mul(-1) : e.grad;
      this.weight_decay !== 0 && (t = t.add(e.mul(this.weight_decay))), this.state.has(e) || this.state.set(e, {
        m: B(e),
        v: B(e),
        vmax: B(e)
      });
      const s = this.state.get(e);
      s.m = s.m.mul(this.beta1).add(t.mul(1 - this.beta1)), s.v = s.v.mul(this.beta2).add(t.mul(t).mul(1 - this.beta2));
      const n = 1 - Math.pow(this.beta1, this.step_count), i = 1 - Math.pow(this.beta2, this.step_count);
      let o;
      const u = s.m.div(n);
      this.amsgrad ? (s.vmax = s.vmax.maximum(s.v), o = s.vmax.div(i)) : o = s.v.div(i);
      const c = u.div(o.sqrt().add(this.eps)).mul(this.lr), d = e.sub(c);
      e.data = d.data;
    }
  }
};
a(Te, "Adam");
let oe = Te;
const br = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Adam: oe,
  Optimizer: N,
  SGD: ie
}, Symbol.toStringTag, { value: "Module" }));
export {
  C as AccumulateGrad,
  Y as Matmul,
  Q as Mean,
  V as PowInt,
  z as Reshape,
  J as Sum,
  h as Tensor,
  q as TorchFunction,
  X as Transpose,
  H as Unsqueeze,
  lt as __left_index__,
  _t as __right_index__,
  kt as abs,
  ft as add,
  ht as arange,
  Tt as cos,
  mt as div,
  jt as eq,
  k as eventBus,
  E as events,
  qt as exp,
  xt as fmod,
  $t as ge,
  St as gt,
  Wt as le,
  dt as linspace,
  At as log,
  It as lt,
  Pt as matmul,
  bt as maximum,
  Dt as mean,
  yt as minimum,
  gt as mul,
  Kt as ne,
  Et as neg,
  xr as nn,
  He as ones,
  ct as ones_like,
  br as optim,
  wt as pow,
  Ue as rand,
  ut as randint,
  ot as randn,
  Rt as reciprocal,
  Ft as reshape,
  Ze as sign,
  Bt as sin,
  vt as sqrt,
  Ot as square,
  pt as sub,
  Ut as sum,
  Ct as tan,
  Nt as transpose,
  Mt as unsqueeze,
  Je as zeros,
  B as zeros_like
};
//# sourceMappingURL=torch.browser.es.js.map
