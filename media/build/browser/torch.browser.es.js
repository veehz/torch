var Je = Object.defineProperty;
var a = (r, e) => Je(r, "name", { value: e, configurable: !0 });
function K(r) {
  return Array.isArray(r[0]) ? r[0] : r;
}
a(K, "get_shape_from_args");
function ot(...r) {
  const e = K(r), t = new h(Array(e.reduce((n, s) => n * s, 1)).fill(Math.random()));
  return t.shape = e, t;
}
a(ot, "randn");
function Ue(...r) {
  const e = K(r), t = new h(Array(e.reduce((n, s) => n * s, 1)).fill(Math.random()));
  return t.shape = e, t;
}
a(Ue, "rand");
function ut(r, e, t) {
  const n = new h(
    Array(t.reduce((s, i) => s * i, 1)).fill(Math.floor(Math.random() * (e - r) + r))
  );
  return n.shape = t, n;
}
a(ut, "randint");
function Qe(...r) {
  const e = K(r), t = new h(Array(e.reduce((n, s) => n * s, 1)).fill(1));
  return t.shape = e, t;
}
a(Qe, "ones");
function Xe(...r) {
  const e = K(r), t = new h(Array(e.reduce((n, s) => n * s, 1)).fill(0));
  return t.shape = e, t;
}
a(Xe, "zeros");
function ct(r) {
  return Qe(r.shape);
}
a(ct, "ones_like");
function B(r) {
  return Xe(r.shape);
}
a(B, "zeros_like");
function dt(r, e, t) {
  const n = [], s = (e - r) / (t - 1);
  for (let i = 0; i < t - 1; i++)
    n.push(r + i * s);
  return n.push(e), new h(n);
}
a(dt, "linspace");
function ht(r, e = void 0, t = 1) {
  const n = [];
  for (let s = r; s < e; s += t)
    n.push(s);
  return new h(n);
}
a(ht, "arange");
let Ye = 0;
const Ne = /* @__PURE__ */ a(() => Ye++, "getNextId"), k = new EventTarget(), E = {
  TENSOR_BEFORE_BACKWARD: "tensor.beforeBackward",
  TENSOR_AFTER_BACKWARD: "tensor.afterBackward",
  OPERATION_BEFORE_FORWARD: "operation.beforeForward",
  OPERATION_AFTER_FORWARD: "operation.afterForward",
  OPERATION_BEFORE_BACKWARD: "operation.beforeBackward",
  OPERATION_AFTER_BACKWARD: "operation.afterBackward",
  OPERATION_BEFORE_ACCUMULATE_GRAD: "operation.beforeAccumulateGrad",
  OPERATION_AFTER_ACCUMULATE_GRAD: "operation.afterAccumulateGrad"
};
function Ze(...r) {
  for (const e of r)
    if (e instanceof h && e.requires_grad)
      return !0;
  return !1;
}
a(Ze, "resultRequiresGrad");
const ce = class ce {
  id = Ne();
  next_functions = [];
  saved_tensors = [];
  _retained_tensors = [];
  forward(...e) {
    const t = Ze(...e);
    k.dispatchEvent(new CustomEvent(E.OPERATION_BEFORE_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e
      }
    }));
    const n = this._forward(...e);
    return k.dispatchEvent(new CustomEvent(E.OPERATION_AFTER_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e,
        result: n
      }
    })), n;
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
const y = new L(), he = class he extends q {
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
function ze(r) {
  if (ArrayBuffer.isView(r))
    return [r.length];
  const e = [];
  for (; Array.isArray(r); )
    e.push(r.length), r = r[0];
  return e;
}
a(ze, "_get_shape");
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
  constructor(e, t = {}, n = {}) {
    if (this.data = Ie(e), this.requires_grad = t.requires_grad ?? !1, t.name && (this.name = t.name), this._shape = n.shape ?? ze(e), this.grad_fn = n.operation ?? null, this.requires_grad && !this.grad_fn) {
      const s = new C();
      s.variable = this, this.grad_fn = s;
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
  toArray() {
    return this.data;
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
};
a(M, "Tensor");
let h = M;
function Se(r, e) {
  const t = Math.max(r.length, e.length), n = [...Array(t - r.length).fill(1), ...r], s = [...Array(t - e.length).fill(1), ...e], i = [];
  for (let o = 0; o < t; o++) {
    if (n[o] !== s[o] && n[o] !== 1 && s[o] !== 1)
      throw new Error(`Shape mismatch: ${r} and ${e}`);
    i.push(Math.max(n[o], s[o]));
  }
  return i;
}
a(Se, "_broadcast_shape");
function W(r, e) {
  return r.length >= e.length ? r : [...Array(e.length - r.length).fill(1), ...r];
}
a(W, "_pad_shape");
function $(r, e, t) {
  let n = 0, s = 1, i = t;
  for (let o = r.length - 1; o >= 0; o--) {
    if (r[o] > 1) {
      const u = i % e[o];
      n = n + u * s;
    }
    s *= r[o], i = Math.floor(i / e[o]);
  }
  return n;
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
const lt = _("__left_index__"), _t = _("__right_index__"), ft = _("add"), pt = _("sub"), gt = _("mul"), mt = _("div"), wt = _("pow"), xt = _("fmod"), bt = _("maximum"), At = _("minimum"), yt = w("log"), vt = w("sqrt"), qt = w("exp"), Ot = w("square"), kt = w("abs"), We = w("sign"), Et = w("neg"), Rt = w("reciprocal"), Ft = ue("reshape"), Mt = ue("unsqueeze"), Bt = w("sin"), Tt = w("cos"), Ct = w("tan"), Ut = w("sum"), Dt = w("mean"), Nt = ue("transpose"), Pt = _("matmul"), It = _("lt"), St = _("gt"), Wt = _("le"), $t = _("ge"), jt = _("eq"), Kt = _("ne");
function f(r, e, t = null) {
  var o;
  const n = /* @__PURE__ */ a((u, d, c, l, b, p) => {
    const g = Array(p);
    for (let A = 0; A < p; A++) {
      const m = $(d, b, A), v = $(l, b, A);
      g[A] = r(u, c, m, v);
    }
    return g;
  }, "kernel"), s = /* @__PURE__ */ a((u, d, c = null) => {
    const l = Se(u.shape, d.shape), b = W(u.shape, l), p = W(d.shape, l), g = l.reduce((A, m) => A * m, 1);
    return new h(
      n(u.data, b, d.data, p, l, g),
      { requires_grad: u.requires_grad || d.requires_grad },
      { operation: c, shape: l }
    );
  }, "forward_tensor"), i = (o = class extends S {
    _forward(d, c) {
      return (d.requires_grad || c.requires_grad) && (this.saved_tensors = [d, c]), this.next_functions.push(d.grad_fn ? d.grad_fn : y), this.next_functions.push(c.grad_fn ? c.grad_fn : y), s(d, c, d.requires_grad || c.requires_grad ? this : null);
    }
    _backward(d) {
      const [c, l] = this.saved_tensors, [b, p] = this.next_functions;
      e(c, l, b, p, d);
    }
  }, a(o, "result"), o);
  return t && O(t, i), i;
}
a(f, "BinaryFunctionMixin");
function x(r, e, t = null) {
  var o;
  const n = /* @__PURE__ */ a((u, d) => {
    const c = Array(d);
    for (let l = 0; l < d; l++)
      c[l] = r(u, l);
    return c;
  }, "kernel"), s = /* @__PURE__ */ a((u, d = null) => {
    const c = u.dataLength();
    return new h(
      n(u.data, c),
      { requires_grad: u.requires_grad },
      { operation: d, shape: u.shape }
    );
  }, "forward_tensor"), i = (o = class extends T {
    _forward(d) {
      return d.requires_grad && (this.saved_tensors = [d]), this.next_functions.push(d.grad_fn ? d.grad_fn : y), s(d, d.requires_grad ? this : null);
    }
    _backward(d) {
      const [c] = this.saved_tensors, [l] = this.next_functions;
      e(c, l, d);
    }
  }, a(o, "result"), o);
  return t && O(t, i), i;
}
a(x, "UnaryFunctionMixin");
const Gt = f(
  (r, e, t, n) => t,
  (r, e, t, n, s) => {
  },
  "__left_index__"
), Lt = f(
  (r, e, t, n) => n,
  (r, e, t, n, s) => {
  },
  "__right_index__"
), Vt = f(
  (r, e, t, n) => r[t] + e[n],
  (r, e, t, n, s) => {
    t.backward(s), n.backward(s);
  },
  "add"
), Ht = f(
  (r, e, t, n) => r[t] - e[n],
  (r, e, t, n, s) => {
    t.backward(s), n.backward(s.mul(new h(-1)));
  },
  "sub"
), Jt = f(
  (r, e, t, n) => r[t] * e[n],
  (r, e, t, n, s) => {
    t.backward(s.mul(e)), n.backward(s.mul(r));
  },
  "mul"
), Qt = f(
  (r, e, t, n) => r[t] / e[n],
  (r, e, t, n, s) => {
    t.backward(s.div(e)), n.backward(s.mul(r).mul(new h(-1)).div(e).div(e));
  },
  "div"
), Xt = f(
  (r, e, t, n) => Math.pow(r[t], e[n]),
  (r, e, t, n, s) => {
    t.backward(s.mul(e).mul(r.pow(e.sub(new h(1))))), n.backward(s.mul(r.pow(e)).mul(r.log()));
  },
  "pow"
), Yt = f(
  (r, e, t, n) => r[t] % e[n],
  (r, e, t, n, s) => {
    t.backward(s);
  },
  "fmod"
), Zt = f(
  (r, e, t, n) => Math.max(r[t], e[n]),
  (r, e, t, n, s) => {
    t.backward(s.mul(r.ge(e))), n.backward(s.mul(e.gt(r)));
  },
  "maximum"
), zt = f(
  (r, e, t, n) => Math.min(r[t], e[n]),
  (r, e, t, n, s) => {
    t.backward(s.mul(r.le(e))), n.backward(s.mul(e.lt(r)));
  },
  "minimum"
);
function et(r, e, t = null) {
  const n = new Array(r.dataLength());
  for (let s = 0; s < n.length; s++)
    n[s] = Math.pow(r.data[s], e);
  return new h(
    n,
    { requires_grad: r.requires_grad },
    { operation: t, shape: r.shape }
  );
}
a(et, "_powint_tensor");
const fe = class fe extends q {
  n;
  _forward(e, t) {
    return e.requires_grad && (this.saved_tensors = [e], this.n = t), this.next_functions.push(e.grad_fn ? e.grad_fn : y), et(e, t, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, n = this.n, [s] = this.next_functions;
    s.backward(e.mul(n).mul(t.pow(n - 1)));
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
), nr = x(
  (r, e) => r[e] * r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(r).mul(new h(2))));
  },
  "square"
), sr = x(
  (r, e) => Math.abs(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(We(r))));
  },
  "abs"
), ar = x(
  (r, e) => Math.sign(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(We(r))));
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
    e.backward(t.mul(t.mul(r.pow(-2))));
  },
  "reciprocal"
), pe = class pe extends q {
  _forward(e, t) {
    const n = e.dataLength(), s = t.reduce((i, o) => i * o, 1);
    if (n !== s)
      throw new Error("Shape mismatch: " + e.shape + " and " + t);
    if (e.requires_grad && (this.saved_tensors = [e]), e.grad_fn)
      this.next_functions.push(e.grad_fn);
    else if (e.requires_grad) {
      const i = new C();
      i.variable = e, this.next_functions.push(i);
    } else
      this.next_functions.push(y);
    return new h(
      e.data,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: t }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [n] = this.next_functions;
    n.backward(e.reshape(t.shape));
  }
};
a(pe, "Reshape");
let H = pe;
O("reshape", H);
const ge = class ge extends q {
  _forward(e, t) {
    if (e.requires_grad && (this.saved_tensors = [e]), e.grad_fn)
      this.next_functions.push(e.grad_fn);
    else if (e.requires_grad) {
      const s = new C();
      s.variable = e, this.next_functions.push(s);
    } else
      this.next_functions.push(y);
    t < 0 && (t += e.shape.length + 1);
    const n = [...e.shape];
    return n.splice(t, 0, 1), new h(
      e.data,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: n }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [n] = this.next_functions;
    n.backward(e.reshape(t.shape));
  }
};
a(ge, "Unsqueeze");
let J = ge;
O("unsqueeze", J);
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
    r.toArray().reduce((t, n) => t + n, 0),
    { requires_grad: r.requires_grad },
    { operation: e }
  );
}
a(tt, "_sum_tensor");
const me = class me extends T {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : y), tt(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, [n] = this.next_functions;
    n.backward(B(t).add(e.item()));
  }
};
a(me, "Sum");
let Q = me;
O("sum", Q);
function rt(r, e = null) {
  return new h(
    r.toArray().reduce((t, n) => t + n, 0) / r.dataLength(),
    { requires_grad: r.requires_grad },
    { operation: e }
  );
}
a(rt, "_mean_tensor");
const we = class we extends T {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : y), rt(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, [n] = this.next_functions;
    n.backward(B(t).add(e.item() / t.dataLength()));
  }
};
a(we, "Mean");
let X = we;
O("mean", X);
function nt(r, e, t, n = null) {
  if (r.shape.length + e < 0 || r.shape.length + t < 0)
    throw new Error(`Transpose: Dimension out of range (${e} and ${t})`);
  e = e < 0 ? r.shape.length + e : e, t = t < 0 ? r.shape.length + t : t;
  const s = [...r.shape];
  [s[e], s[t]] = [s[t], s[e]];
  const i = r.dataLength(), o = new Array(i), u = new Array(r.shape.length), d = new Array(s.length);
  for (let c = r.shape.length - 1, l = 1; c >= 0; c--)
    u[c] = l, l *= r.shape[c];
  for (let c = s.length - 1, l = 1; c >= 0; c--)
    d[c] = l, l *= s[c];
  for (let c = 0; c < i; c++) {
    let l = c, b = 0;
    for (let p = 0; p < s.length; p++) {
      const g = d[p], A = Math.floor(l / g);
      l %= g;
      let m = p;
      p === e ? m = t : p === t && (m = e), b += A * u[m];
    }
    o[c] = r.data[b];
  }
  return new h(
    o,
    { requires_grad: r.requires_grad },
    { operation: n, shape: s }
  );
}
a(nt, "_transpose_tensor");
const xe = class xe extends q {
  dim0;
  dim1;
  _forward(e, t, n) {
    return e.requires_grad && (this.saved_tensors = [e], this.dim0 = t, this.dim1 = n), this.next_functions.push(e.grad_fn ? e.grad_fn : y), nt(e, t, n, this);
  }
  _backward(e) {
    const [t] = this.saved_tensors, n = this.dim0, s = this.dim1, [i] = this.next_functions;
    i.backward(e.transpose(n, s));
  }
};
a(xe, "Transpose");
let Y = xe;
O("transpose", Y);
function st(r, e, t = null) {
  if (r.shape.length == 1 && e.shape.length == 1)
    return r.mul(e).sum();
  const n = r.shape.length == 1, s = e.shape.length == 1, i = n ? [1, r.shape[0]] : r.shape, o = s ? [e.shape[0], 1] : e.shape;
  if (i[i.length - 1] != o[o.length - 2])
    throw new Error("Shape mismatch: " + r.shape + " and " + e.shape);
  const u = Se(i.slice(0, -2), o.slice(0, -2)).concat([
    i[i.length - 2],
    o[o.length - 1]
  ]), d = u.reduce((v, P) => v * P, 1), c = new Array(d).fill(0), l = W(i, u), b = W(o, u), p = u[u.length - 2], g = u[u.length - 1], A = i[i.length - 1];
  for (let v = 0; v < d; v++) {
    const P = v % (p * g), Ge = Math.floor(P / g), Le = P % g;
    let Ve = $(l, u, v - Le), He = $(b, u, v - Ge * g), Ce = 0;
    for (let I = 0; I < A; I++)
      Ce += r.data[Ve + I] * e.data[He + I * g];
    c[v] = Ce;
  }
  let m = [...u];
  return n && (m = m.slice(0, -2).concat([u[u.length - 1]])), s && (m = m.slice(0, -1)), new h(
    c,
    { requires_grad: r.requires_grad || e.requires_grad },
    { operation: t, shape: m }
  );
}
a(st, "_matmul_tensor");
const be = class be extends S {
  _forward(e, t) {
    return (e.requires_grad || t.requires_grad) && (this.saved_tensors = [e, t]), this.next_functions.push(e.grad_fn ? e.grad_fn : y), this.next_functions.push(t.grad_fn ? t.grad_fn : y), st(e, t, e.requires_grad || t.requires_grad ? this : null);
  }
  _backward(e) {
    const [t, n] = this.saved_tensors, [s, i] = this.next_functions;
    if (t.shape.length == 1 && n.shape.length == 1) {
      s.backward(e), i.backward(e);
      return;
    }
    if (t.shape.length == 1) {
      const o = e.unsqueeze(0), u = t.unsqueeze(0);
      s.backward(o.matmul(n.transpose(-2, -1))), i.backward(u.transpose(0, 1).matmul(o));
      return;
    }
    if (n.shape.length == 1) {
      const o = e.unsqueeze(0), u = n.unsqueeze(1);
      s.backward(o.matmul(u.transpose(0, 1))), i.backward(t.transpose(-2, -1).matmul(o));
      return;
    }
    s.backward(e.matmul(n.transpose(-2, -1))), i.backward(t.transpose(-2, -1).matmul(e));
  }
};
a(be, "Matmul");
let Z = be;
O("matmul", Z);
const hr = f(
  (r, e, t, n) => r[t] < e[n] ? 1 : 0,
  (r, e, t, n) => {
  },
  "lt"
), lr = f(
  (r, e, t, n) => r[t] > e[n] ? 1 : 0,
  (r, e, t, n) => {
  },
  "gt"
), _r = f(
  (r, e, t, n) => r[t] <= e[n] ? 1 : 0,
  (r, e, t, n) => {
  },
  "le"
), fr = f(
  (r, e, t, n) => r[t] >= e[n] ? 1 : 0,
  (r, e, t, n) => {
  },
  "ge"
), pr = f(
  (r, e, t, n) => r[t] == e[n] ? 1 : 0,
  (r, e, t, n) => {
  },
  "eq"
), gr = f(
  (r, e, t, n) => r[t] != e[n] ? 1 : 0,
  (r, e, t, n) => {
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
    e.backward(t.mul(t.mul(r.exp().add(1).pow(-2).reciprocal().mul(r.exp()).mul(-1))));
  },
  "sigmoid"
), j = class j extends h {
  constructor(e, t = {
    requires_grad: !0
  }, n = {}) {
    e instanceof h ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : e instanceof j ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : super(e, t, n);
  }
};
a(j, "Parameter");
let U = j;
const Ae = class Ae {
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
a(Ae, "Module");
let F = Ae;
const ye = class ye extends F {
  weight;
  bias;
  constructor(e, t) {
    super();
    const n = Math.sqrt(1 / e);
    this.weight = new U(
      Ue([t, e]).mul(2 * n).sub(n)
    ), this.bias = new U(
      Ue([t]).mul(2 * n).sub(n)
    ), this.register("weight", this.weight), this.register("bias", this.bias);
  }
  forward(e) {
    return e.matmul(this.weight.transpose(0, 1)).add(this.bias);
  }
};
a(ye, "Linear");
let z = ye;
const ve = class ve extends F {
  constructor() {
    super();
  }
  forward(e) {
    return je(e);
  }
};
a(ve, "ReLU");
let ee = ve;
const qe = class qe extends F {
  constructor() {
    super();
  }
  forward(e) {
    return Ke(e);
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
    for (let n = e; n < this._modulesArr.length; n++)
      this.register(n.toString(), this._modulesArr[n]);
    return this;
  }
  forward(e) {
    let t = e;
    for (const n of this._modulesArr)
      t = n.forward(t);
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
let ne = Ee;
const Re = class Re extends D {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).abs().mean();
  }
};
a(Re, "L1Loss");
let se = Re;
const Fe = class Fe extends D {
  weight;
  constructor(e = null) {
    super(), this.weight = e;
  }
  forward(e, t) {
    const n = t.mul(e.log()), s = t.neg().add(1).mul(e.neg().add(1).log()), i = n.add(s).neg().mean();
    return this.weight ? i.mul(this.weight) : i;
  }
};
a(Fe, "BCELoss");
let ae = Fe;
function $e(r) {
  return (e) => (typeof e == "number" && (e = new h(e)), new (R(r))().forward(e));
}
a($e, "generate_unary_function");
const je = $e("relu"), Ke = $e("sigmoid"), at = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  relu: je,
  sigmoid: Ke
}, Symbol.toStringTag, { value: "Module" })), xr = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  BCELoss: ae,
  L1Loss: se,
  Linear: z,
  MSELoss: ne,
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
  constructor(e, t = 1e-3, n = 0, s = 0, i = 0, o = !1, u = !1) {
    super(e, {}), this.lr = t, this.momentum = n, this.dampening = s, this.weight_decay = i, this.nesterov = o, this.maximize = u;
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
        let s = this.state.get(e).velocity;
        this.nesterov ? t = t.add(s.mul(this.momentum)) : t = s, this.state.set(e, { velocity: s });
      }
      const n = e.sub(t.mul(this.lr));
      e.data = n.data;
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
  constructor(e, t = 1e-3, n = [0.9, 0.999], s = 1e-8, i = 0, o = !1, u = !1) {
    super(e, {}), this.lr = t, this.beta1 = n[0], this.beta2 = n[1], this.eps = s, this.weight_decay = i, this.amsgrad = o, this.maximize = u;
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
      const n = this.state.get(e);
      n.m = n.m.mul(this.beta1).add(t.mul(1 - this.beta1)), n.v = n.v.mul(this.beta2).add(t.mul(t).mul(1 - this.beta2));
      const s = 1 - Math.pow(this.beta1, this.step_count), i = 1 - Math.pow(this.beta2, this.step_count);
      let o;
      const u = n.m.div(s);
      this.amsgrad ? (n.vmax = n.vmax.maximum(n.v), o = n.vmax.div(i)) : o = n.v.div(i);
      const d = u.div(o.sqrt().add(this.eps)).mul(this.lr), c = e.sub(d);
      e.data = c.data;
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
  Z as Matmul,
  X as Mean,
  V as PowInt,
  H as Reshape,
  Q as Sum,
  h as Tensor,
  q as TorchFunction,
  Y as Transpose,
  J as Unsqueeze,
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
  yt as log,
  It as lt,
  Pt as matmul,
  bt as maximum,
  Dt as mean,
  At as minimum,
  gt as mul,
  Kt as ne,
  Et as neg,
  xr as nn,
  Qe as ones,
  ct as ones_like,
  br as optim,
  wt as pow,
  Ue as rand,
  ut as randint,
  ot as randn,
  Rt as reciprocal,
  Ft as reshape,
  We as sign,
  Bt as sin,
  vt as sqrt,
  Ot as square,
  pt as sub,
  Ut as sum,
  Ct as tan,
  Nt as transpose,
  Mt as unsqueeze,
  Xe as zeros,
  B as zeros_like
};
//# sourceMappingURL=torch.browser.es.js.map
