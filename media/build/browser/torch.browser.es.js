var Xe = Object.defineProperty;
var i = (r, e) => Xe(r, "name", { value: e, configurable: !0 });
function We(r, e) {
  const t = Math.max(r.length, e.length), s = [...Array(t - r.length).fill(1), ...r], n = [...Array(t - e.length).fill(1), ...e], a = [];
  for (let o = 0; o < t; o++) {
    if (s[o] !== n[o] && s[o] !== 1 && n[o] !== 1)
      throw new Error(`Shape mismatch: ${r} and ${e}`);
    a.push(Math.max(s[o], n[o]));
  }
  return a;
}
i(We, "_broadcast_shape");
function $e(r, e, t) {
  const s = D(e, r), n = new Array(e.reduce((a, o) => a * o, 1)).fill(0);
  for (let a = 0; a < t.length; a++)
    n[N(s, r, a)] += t[a];
  return n;
}
i($e, "_unbroadcast");
function D(r, e) {
  return r.length >= e.length ? r : [...Array(e.length - r.length).fill(1), ...r];
}
i(D, "_pad_shape");
function N(r, e, t) {
  let s = 0, n = 1, a = t;
  for (let o = r.length - 1; o >= 0; o--) {
    if (r[o] > 1) {
      const c = a % e[o];
      s = s + c * n;
    }
    n *= r[o], a = Math.floor(a / e[o]);
  }
  return s;
}
i(N, "_get_original_index");
function G(r) {
  return Array.isArray(r[0]) ? r[0] : r;
}
i(G, "get_shape_from_args");
function ht(...r) {
  const e = G(r), t = new h(Array(e.reduce((s, n) => s * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
i(ht, "randn");
function Pe(...r) {
  const e = G(r), t = new h(Array(e.reduce((s, n) => s * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
i(Pe, "rand");
function lt(r, e, t) {
  const s = new h(
    Array(t.reduce((n, a) => n * a, 1)).fill(Math.floor(Math.random() * (e - r) + r))
  );
  return s.shape = t, s;
}
i(lt, "randint");
function Ye(...r) {
  const e = G(r), t = new h(Array(e.reduce((s, n) => s * n, 1)).fill(1));
  return t.shape = e, t;
}
i(Ye, "ones");
function Ze(...r) {
  const e = G(r), t = new h(Array(e.reduce((s, n) => s * n, 1)).fill(0));
  return t.shape = e, t;
}
i(Ze, "zeros");
function _t(r) {
  return Ye(r.shape);
}
i(_t, "ones_like");
function T(r) {
  return Ze(r.shape);
}
i(T, "zeros_like");
function ft(r, e, t) {
  const s = [], n = (e - r) / (t - 1);
  for (let a = 0; a < t - 1; a++)
    s.push(r + a * n);
  return s.push(e), new h(s);
}
i(ft, "linspace");
function pt(r, e = void 0, t = 1) {
  const s = [];
  for (let n = r; n < e; n += t)
    s.push(n);
  return new h(s);
}
i(pt, "arange");
let et = 0;
const Se = /* @__PURE__ */ i(() => et++, "getNextId"), k = new EventTarget(), E = {
  TENSOR_BEFORE_BACKWARD: "tensor.beforeBackward",
  TENSOR_AFTER_BACKWARD: "tensor.afterBackward",
  OPERATION_BEFORE_FORWARD: "operation.beforeForward",
  OPERATION_AFTER_FORWARD: "operation.afterForward",
  OPERATION_BEFORE_BACKWARD: "operation.beforeBackward",
  OPERATION_AFTER_BACKWARD: "operation.afterBackward",
  OPERATION_BEFORE_ACCUMULATE_GRAD: "operation.beforeAccumulateGrad",
  OPERATION_AFTER_ACCUMULATE_GRAD: "operation.afterAccumulateGrad"
};
function tt(...r) {
  for (const e of r)
    if (e instanceof h && e.requires_grad)
      return !0;
  return !1;
}
i(tt, "resultRequiresGrad");
const he = class he {
  id = Se();
  next_functions = [];
  saved_tensors = [];
  _retained_tensors = [];
  forward(...e) {
    const t = tt(...e);
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
i(he, "TorchFunction");
let A = he;
const le = class le extends A {
  _forward(...e) {
    throw new Error("NullOp should not be called");
  }
  _backward(e) {
  }
};
i(le, "NullOp");
let z = le;
const q = new z(), _e = class _e extends A {
};
i(_e, "UnaryFunction");
let C = _e;
const fe = class fe extends A {
};
i(fe, "BinaryFunction");
let j = fe;
const pe = class pe extends C {
  variable;
  _forward(e) {
    return this.variable = e, e;
  }
  _backward(e) {
    if (this.variable.grad || (this.variable.grad = T(this.variable)), k.dispatchEvent(new CustomEvent(E.OPERATION_BEFORE_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } })), typeof e == "number")
      this.variable.grad = this.variable.grad.add(e);
    else {
      const t = $e(e.shape, this.variable.shape, e.data);
      this.variable.grad = this.variable.grad.add(new h(t, {}, { shape: this.variable.shape }));
    }
    k.dispatchEvent(new CustomEvent(E.OPERATION_AFTER_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } }));
  }
};
i(pe, "AccumulateGrad");
let F = pe;
const je = /* @__PURE__ */ new Map(), L = /* @__PURE__ */ new Map();
function O(r, e) {
  je.set(r, e);
}
i(O, "registerOperation");
function R(r) {
  const e = je.get(r);
  if (!e)
    throw new Error(`Operation '${r}' is not registered.`);
  return e;
}
i(R, "getOperation");
function Ie(r) {
  const e = L.get(r);
  return e || (L.set(r, new (R(r))()), L.get(r));
}
i(Ie, "getOperationCache");
function rt(r) {
  if (ArrayBuffer.isView(r))
    return [r.length];
  const e = [];
  for (; Array.isArray(r); )
    e.push(r.length), r = r[0];
  return e;
}
i(rt, "_get_shape");
function Ke(r) {
  return Array.isArray(r) ? r.flatMap((e) => Ke(e)) : ArrayBuffer.isView(r) ? Array.from(r) : [r];
}
i(Ke, "_flatten");
const B = class B {
  // Auto-generated ID
  id = Se();
  // Optional user-defined name
  name = null;
  data;
  _shape;
  grad_fn = null;
  grad = null;
  requires_grad;
  constructor(e, t = {}, s = {}) {
    if (this.data = Ke(e), this.requires_grad = t.requires_grad ?? !1, t.name && (this.name = t.name), this._shape = s.shape ?? rt(e), this.grad_fn = s.operation ?? null, this.requires_grad && !this.grad_fn) {
      const n = new F();
      n.variable = this, this.grad_fn = n;
    }
  }
  get shape() {
    return this._shape;
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
    const t = this.data, s = /* @__PURE__ */ i((n) => {
      const a = this.shape[n], o = new Array(a), c = n === this.shape.length - 1;
      for (let u = 0; u < a; u++)
        c ? o[u] = t[e++] : o[u] = s(n + 1);
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
    return (this.requires_grad ? new (R(e))() : Ie(e)).forward(this);
  }
  _executeBinaryOp(e, t) {
    return typeof t == "number" && (t = new B(t)), (this.requires_grad || t.requires_grad ? new (R(e))() : Ie(e)).forward(this, t);
  }
  _executeOpRaw(e, ...t) {
    return new (R(e))().forward(this, ...t);
  }
  item() {
    if (this.dataLength() !== 1)
      throw new Error("Tensor.item() is only valid for scalars");
    return this.data[0];
  }
  detach() {
    return new B(this.data, { requires_grad: !1 }, { shape: this.shape });
  }
  detach_() {
    this.requires_grad = !1, this.grad = null, this.grad_fn = null;
  }
  zero_() {
    this.data = Array(this.dataLength()).fill(0);
  }
  is_retain_grad = !1;
  retain_grad() {
    this.grad_fn instanceof F || this.is_retain_grad || (this.is_retain_grad = !0, this.grad_fn._retained_tensors.push(this));
  }
  backward(e) {
    if (this.requires_grad) {
      if (e)
        e.toArray_();
      else {
        if (this.dataLength() !== 1)
          throw new Error("Gradient is required for non-scalar tensors");
        e = new B(1);
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
  squeeze(e) {
    return this._executeOpRaw("squeeze", e);
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
i(B, "Tensor");
let h = B;
function de(r) {
  return (...e) => new (R(r))().forward(...e);
}
i(de, "generate_function$1");
function x(r) {
  return (e) => (typeof e == "number" && (e = new h(e)), new (R(r))().forward(e));
}
i(x, "generate_unary_function$1");
function f(r) {
  return (e, t) => (typeof e == "number" && (e = new h(e)), typeof t == "number" && (t = new h(t)), new (R(r))().forward(e, t));
}
i(f, "generate_binary_function$1");
const gt = f("__left_index__"), mt = f("__right_index__"), wt = f("add"), xt = f("sub"), bt = f("mul"), qt = f("div"), vt = f("pow"), yt = f("fmod"), At = f("maximum"), Ot = f("minimum"), kt = x("log"), Et = x("sqrt"), Rt = x("exp"), Ft = x("square"), Mt = x("abs"), st = x("sign"), Bt = x("neg"), Tt = x("reciprocal"), Ct = de("reshape"), Ut = de("unsqueeze"), Dt = x("sin"), Nt = x("cos"), Pt = x("tan"), It = x("sum"), Wt = x("mean"), $t = de("transpose"), St = f("matmul"), jt = f("lt"), Kt = f("gt"), Gt = f("le"), Lt = f("ge"), zt = f("eq"), Vt = f("ne");
function p(r, e, t = null) {
  var o;
  const s = /* @__PURE__ */ i((c, u, d, l, _, g) => {
    const w = Array(g);
    for (let v = 0; v < g; v++) {
      const m = N(u, _, v), y = N(l, _, v);
      w[v] = r(c, d, m, y);
    }
    return w;
  }, "kernel"), n = /* @__PURE__ */ i((c, u, d = null) => {
    const l = We(c.shape, u.shape), _ = D(c.shape, l), g = D(u.shape, l), w = l.reduce((v, m) => v * m, 1);
    return new h(
      s(c.data, _, u.data, g, l, w),
      { requires_grad: c.requires_grad || u.requires_grad },
      { operation: d, shape: l }
    );
  }, "forward_tensor"), a = (o = class extends j {
    _forward(u, d) {
      return (u.requires_grad || d.requires_grad) && (this.saved_tensors = [u, d]), this.next_functions.push(u.grad_fn ? u.grad_fn : q), this.next_functions.push(d.grad_fn ? d.grad_fn : q), n(u, d, u.requires_grad || d.requires_grad ? this : null);
    }
    _backward(u) {
      const [d, l] = this.saved_tensors, [_, g] = this.next_functions;
      e(d, l, _, g, u);
    }
  }, i(o, "result"), o);
  return t && O(t, a), a;
}
i(p, "BinaryFunctionMixin");
function b(r, e, t = null) {
  var o;
  const s = /* @__PURE__ */ i((c, u) => {
    const d = Array(u);
    for (let l = 0; l < u; l++)
      d[l] = r(c, l);
    return d;
  }, "kernel"), n = /* @__PURE__ */ i((c, u = null) => {
    const d = c.dataLength();
    return new h(
      s(c.data, d),
      { requires_grad: c.requires_grad },
      { operation: u, shape: c.shape }
    );
  }, "forward_tensor"), a = (o = class extends C {
    _forward(u) {
      return u.requires_grad && (this.saved_tensors = [u]), this.next_functions.push(u.grad_fn ? u.grad_fn : q), n(u, u.requires_grad ? this : null);
    }
    _backward(u) {
      const [d] = this.saved_tensors, [l] = this.next_functions;
      e(d, l, u);
    }
  }, i(o, "result"), o);
  return t && O(t, a), a;
}
i(b, "UnaryFunctionMixin");
function S(r, e) {
  const t = $e(r.shape, e, r.data);
  return new h(t, { requires_grad: r.requires_grad }, { shape: e });
}
i(S, "unbroadcast");
const Ht = p(
  (r, e, t, s) => t,
  (r, e, t, s, n) => {
  },
  "__left_index__"
), Jt = p(
  (r, e, t, s) => s,
  (r, e, t, s, n) => {
  },
  "__right_index__"
), Qt = p(
  (r, e, t, s) => r[t] + e[s],
  (r, e, t, s, n) => {
    t.backward(n), s.backward(n);
  },
  "add"
), Xt = p(
  (r, e, t, s) => r[t] - e[s],
  (r, e, t, s, n) => {
    t.backward(n), s.backward(n.mul(new h(-1)));
  },
  "sub"
), Yt = p(
  (r, e, t, s) => r[t] * e[s],
  (r, e, t, s, n) => {
    t.backward(n.mul(e)), s.backward(n.mul(r));
  },
  "mul"
), Zt = p(
  (r, e, t, s) => r[t] / e[s],
  (r, e, t, s, n) => {
    t.backward(n.div(e)), s.backward(n.mul(r).mul(new h(-1)).div(e).div(e));
  },
  "div"
), er = p(
  (r, e, t, s) => Math.pow(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(e).mul(r.pow(e.sub(new h(1))))), s.backward(n.mul(r.pow(e)).mul(r.log()));
  },
  "pow"
), tr = p(
  (r, e, t, s) => r[t] % e[s],
  (r, e, t, s, n) => {
    t.backward(n);
  },
  "fmod"
), rr = p(
  (r, e, t, s) => Math.max(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(r.ge(e))), s.backward(n.mul(e.gt(r)));
  },
  "maximum"
), sr = p(
  (r, e, t, s) => Math.min(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(r.le(e))), s.backward(n.mul(e.lt(r)));
  },
  "minimum"
);
function nt(r, e, t = null) {
  const s = new Array(r.dataLength());
  for (let n = 0; n < s.length; n++)
    s[n] = Math.pow(r.data[n], e);
  return new h(
    s,
    { requires_grad: r.requires_grad },
    { operation: t, shape: r.shape }
  );
}
i(nt, "_powint_tensor");
const ge = class ge extends A {
  n;
  _forward(e, t) {
    return e.requires_grad && (this.saved_tensors = [e], this.n = t), this.next_functions.push(e.grad_fn ? e.grad_fn : q), nt(e, t, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, s = this.n, [n] = this.next_functions;
    n.backward(e.mul(s).mul(t.pow(s - 1)));
  }
};
i(ge, "PowInt");
let V = ge;
O("powint", V);
const nr = b(
  (r, e) => Math.log(r[e]),
  (r, e, t) => {
    e.backward(t.mul(new h(1).div(r)));
  },
  "log"
), ar = b(
  (r, e) => Math.sqrt(r[e]),
  (r, e, t) => {
    e.backward(t.mul(new h(1).div(r.sqrt()).div(2)));
  },
  "sqrt"
), ir = b(
  (r, e) => Math.exp(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.exp())));
  },
  "exp"
), or = b(
  (r, e) => r[e] * r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(r).mul(new h(2))));
  },
  "square"
), ur = b(
  (r, e) => Math.abs(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(st(r))));
  },
  "abs"
), cr = b(
  (r, e) => Math.sign(r[e]),
  (r, e, t) => {
    e.backward(0);
  },
  "sign"
), dr = b(
  (r, e) => -r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(new h(-1))));
  },
  "neg"
), hr = b(
  (r, e) => 1 / r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.pow(-2))).neg());
  },
  "reciprocal"
), me = class me extends A {
  _forward(e, t) {
    const s = e.dataLength(), n = t.reduce((a, o) => a * o, 1);
    if (s !== n)
      throw new Error("Shape mismatch: " + e.shape + " and " + t);
    if (e.requires_grad && (this.saved_tensors = [e]), e.grad_fn)
      this.next_functions.push(e.grad_fn);
    else if (e.requires_grad) {
      const a = new F();
      a.variable = e, this.next_functions.push(a);
    } else
      this.next_functions.push(q);
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
i(me, "Reshape");
let H = me;
O("reshape", H);
const we = class we extends A {
  _forward(e, t) {
    if (e.requires_grad && (this.saved_tensors = [e]), e.grad_fn)
      this.next_functions.push(e.grad_fn);
    else if (e.requires_grad) {
      const n = new F();
      n.variable = e, this.next_functions.push(n);
    } else
      this.next_functions.push(q);
    let s = [...e.shape];
    return t !== void 0 ? (t < 0 && (t += e.shape.length), s[t] === 1 && s.splice(t, 1)) : s = s.filter((n) => n !== 1), new h(
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
i(we, "Squeeze");
let J = we;
O("squeeze", J);
const xe = class xe extends A {
  _forward(e, t) {
    if (e.requires_grad && (this.saved_tensors = [e]), e.grad_fn)
      this.next_functions.push(e.grad_fn);
    else if (e.requires_grad) {
      const n = new F();
      n.variable = e, this.next_functions.push(n);
    } else
      this.next_functions.push(q);
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
i(xe, "Unsqueeze");
let Q = xe;
O("unsqueeze", Q);
const lr = b(
  (r, e) => Math.sin(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.cos())));
  },
  "sin"
), _r = b(
  (r, e) => Math.cos(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.sin().neg())));
  },
  "cos"
), fr = b(
  (r, e) => Math.tan(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.cos().pow(-2))));
  },
  "tan"
);
function at(r, e = null) {
  return new h(
    r.toFlatArray().reduce((t, s) => t + s, 0),
    { requires_grad: r.requires_grad },
    { operation: e }
  );
}
i(at, "_sum_tensor");
const be = class be extends C {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : q), at(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(T(t).add(e.item()));
  }
};
i(be, "Sum");
let X = be;
O("sum", X);
function it(r, e = null) {
  return new h(
    r.toFlatArray().reduce((t, s) => t + s, 0) / r.dataLength(),
    { requires_grad: r.requires_grad },
    { operation: e }
  );
}
i(it, "_mean_tensor");
const qe = class qe extends C {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : q), it(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(T(t).add(e.item() / t.dataLength()));
  }
};
i(qe, "Mean");
let Y = qe;
O("mean", Y);
function ot(r, e, t, s = null) {
  if (r.shape.length + e < 0 || r.shape.length + t < 0)
    throw new Error(`Transpose: Dimension out of range (${e} and ${t})`);
  e = e < 0 ? r.shape.length + e : e, t = t < 0 ? r.shape.length + t : t;
  const n = [...r.shape];
  [n[e], n[t]] = [n[t], n[e]];
  const a = r.dataLength(), o = new Array(a), c = new Array(r.shape.length), u = new Array(n.length);
  for (let d = r.shape.length - 1, l = 1; d >= 0; d--)
    c[d] = l, l *= r.shape[d];
  for (let d = n.length - 1, l = 1; d >= 0; d--)
    u[d] = l, l *= n[d];
  for (let d = 0; d < a; d++) {
    let l = d, _ = 0;
    for (let g = 0; g < n.length; g++) {
      const w = u[g], v = Math.floor(l / w);
      l %= w;
      let m = g;
      g === e ? m = t : g === t && (m = e), _ += v * c[m];
    }
    o[d] = r.data[_];
  }
  return new h(
    o,
    { requires_grad: r.requires_grad },
    { operation: s, shape: n }
  );
}
i(ot, "_transpose_tensor");
const ve = class ve extends A {
  dim0;
  dim1;
  _forward(e, t, s) {
    return e.requires_grad && (this.saved_tensors = [e], this.dim0 = t, this.dim1 = s), this.next_functions.push(e.grad_fn ? e.grad_fn : q), ot(e, t, s, this);
  }
  _backward(e) {
    const [t] = this.saved_tensors, s = this.dim0, n = this.dim1, [a] = this.next_functions;
    a.backward(e.transpose(s, n));
  }
};
i(ve, "Transpose");
let Z = ve;
O("transpose", Z);
function ut(r, e, t = null) {
  if (r.shape.length == 1 && e.shape.length == 1)
    return [r.mul(e).sum(), []];
  const s = r.shape.length == 1, n = e.shape.length == 1, a = s ? [1, r.shape[0]] : r.shape, o = n ? [e.shape[0], 1] : e.shape;
  if (a[a.length - 1] != o[o.length - 2])
    throw new Error("Shape mismatch: " + r.shape + " and " + e.shape);
  const c = We(a.slice(0, -2), o.slice(0, -2)).concat([
    a[a.length - 2],
    o[o.length - 1]
  ]), u = c.reduce((y, W) => y * W, 1), d = new Array(u).fill(0), l = D(a, c), _ = D(o, c), g = c[c.length - 2], w = c[c.length - 1], v = a[a.length - 1];
  for (let y = 0; y < u; y++) {
    const W = y % (g * w), Ve = Math.floor(W / w), He = W % w;
    let Je = N(l, c, y - He), Qe = N(_, c, y - Ve * w), Ne = 0;
    for (let $ = 0; $ < v; $++)
      Ne += r.data[Je + $] * e.data[Qe + $ * w];
    d[y] = Ne;
  }
  let m = [...c];
  return s && (m = m.slice(0, -2).concat([c[c.length - 1]])), n && (m = m.slice(0, -1)), [new h(
    d,
    { requires_grad: r.requires_grad || e.requires_grad },
    { operation: t, shape: m }
  ), m];
}
i(ut, "_matmul_tensor");
const ye = class ye extends j {
  shape;
  _forward(e, t) {
    (e.requires_grad || t.requires_grad) && (this.saved_tensors = [e, t]), this.next_functions.push(e.grad_fn ? e.grad_fn : q), this.next_functions.push(t.grad_fn ? t.grad_fn : q);
    const s = ut(e, t, e.requires_grad || t.requires_grad ? this : null);
    return this.shape = s[1], s[0];
  }
  _backward(e) {
    const [t, s] = this.saved_tensors, [n, a] = this.next_functions;
    if (t.shape.length === 1 && s.shape.length === 1) {
      n.backward(e.mul(s)), a.backward(e.mul(t));
      return;
    }
    if (t.shape.length === 1) {
      const u = e.unsqueeze(-2), d = t.unsqueeze(-2);
      let l = u.matmul(s.transpose(-2, -1)), _ = d.transpose(-2, -1).matmul(u);
      l = l.squeeze(-2), _ = S(_, s.shape), n.backward(l), a.backward(_);
      return;
    }
    if (s.shape.length === 1) {
      const u = e.unsqueeze(-1), d = s.unsqueeze(-1);
      let l = u.matmul(d.transpose(-2, -1)), _ = t.transpose(-2, -1).matmul(u);
      l = S(l, t.shape), _ = _.squeeze(-1), n.backward(l), a.backward(_);
      return;
    }
    let o = e.matmul(s.transpose(-2, -1)), c = t.transpose(-2, -1).matmul(e);
    o = S(o, t.shape), c = S(c, s.shape), n.backward(o), a.backward(c);
  }
};
i(ye, "Matmul");
let ee = ye;
O("matmul", ee);
const pr = p(
  (r, e, t, s) => r[t] < e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "lt"
), gr = p(
  (r, e, t, s) => r[t] > e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "gt"
), mr = p(
  (r, e, t, s) => r[t] <= e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "le"
), wr = p(
  (r, e, t, s) => r[t] >= e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "ge"
), xr = p(
  (r, e, t, s) => r[t] == e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "eq"
), br = p(
  (r, e, t, s) => r[t] != e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "ne"
), qr = b(
  (r, e) => Math.max(r[e], 0),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.gt(0))));
  },
  "relu"
), vr = b(
  (r, e) => 1 / (1 + Math.exp(-r[e])),
  (r, e, t) => {
    const s = r.sigmoid();
    e.backward(s.mul(s.mul(-1).add(1)).mul(t));
  },
  "sigmoid"
), K = class K extends h {
  constructor(e, t = {
    requires_grad: !0
  }, s = {}) {
    e instanceof h ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : e instanceof K ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : super(e, t, s);
  }
};
i(K, "Parameter");
let U = K;
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
i(Ae, "Module");
let M = Ae;
const Oe = class Oe extends M {
  weight;
  bias;
  constructor(e, t) {
    super();
    const s = Math.sqrt(1 / e);
    this.weight = new U(
      Pe([t, e]).mul(2 * s).sub(s)
    ), this.bias = new U(
      Pe([t]).mul(2 * s).sub(s)
    ), this.register("weight", this.weight), this.register("bias", this.bias);
  }
  forward(e) {
    return e.matmul(this.weight.transpose(0, 1)).add(this.bias);
  }
};
i(Oe, "Linear");
let te = Oe;
const ke = class ke extends M {
  constructor() {
    super();
  }
  forward(e) {
    return Le(e);
  }
};
i(ke, "ReLU");
let re = ke;
const Ee = class Ee extends M {
  constructor() {
    super();
  }
  forward(e) {
    return ze(e);
  }
};
i(Ee, "Sigmoid");
let se = Ee;
const Re = class Re extends M {
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
i(Re, "Sequential");
let ne = Re;
const Fe = class Fe {
};
i(Fe, "Loss");
let P = Fe;
const Me = class Me extends P {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).pow(2).mean();
  }
};
i(Me, "MSELoss");
let ae = Me;
const Be = class Be extends P {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).abs().mean();
  }
};
i(Be, "L1Loss");
let ie = Be;
const Te = class Te extends P {
  weight;
  constructor(e = null) {
    super(), this.weight = e;
  }
  forward(e, t) {
    const s = t.mul(e.log()), n = t.neg().add(1).mul(e.neg().add(1).log()), a = s.add(n).neg().mean();
    return this.weight ? a.mul(this.weight) : a;
  }
};
i(Te, "BCELoss");
let oe = Te;
function Ge(r) {
  return (e) => (typeof e == "number" && (e = new h(e)), new (R(r))().forward(e));
}
i(Ge, "generate_unary_function");
const Le = Ge("relu"), ze = Ge("sigmoid"), ct = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  relu: Le,
  sigmoid: ze
}, Symbol.toStringTag, { value: "Module" })), yr = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  BCELoss: oe,
  L1Loss: ie,
  Linear: te,
  MSELoss: ae,
  Module: M,
  Parameter: U,
  ReLU: re,
  Sequential: ne,
  Sigmoid: se,
  functional: ct
}, Symbol.toStringTag, { value: "Module" })), Ce = class Ce {
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
i(Ce, "Optimizer");
let I = Ce;
const Ue = class Ue extends I {
  state = /* @__PURE__ */ new Map();
  lr;
  momentum;
  dampening;
  weight_decay;
  nesterov;
  maximize;
  constructor(e, t = 1e-3, s = 0, n = 0, a = 0, o = !1, c = !1) {
    super(e, {}), this.lr = t, this.momentum = s, this.dampening = n, this.weight_decay = a, this.nesterov = o, this.maximize = c;
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
i(Ue, "SGD");
let ue = Ue;
const De = class De extends I {
  state = /* @__PURE__ */ new Map();
  step_count = 0;
  lr;
  beta1;
  beta2;
  eps;
  weight_decay;
  amsgrad;
  maximize;
  constructor(e, t = 1e-3, s = [0.9, 0.999], n = 1e-8, a = 0, o = !1, c = !1) {
    super(e, {}), this.lr = t, this.beta1 = s[0], this.beta2 = s[1], this.eps = n, this.weight_decay = a, this.amsgrad = o, this.maximize = c;
  }
  step() {
    this.step_count += 1;
    for (const e of this.params) {
      let t = this.maximize ? e.grad.mul(-1) : e.grad;
      this.weight_decay !== 0 && (t = t.add(e.mul(this.weight_decay))), this.state.has(e) || this.state.set(e, {
        m: T(e),
        v: T(e),
        vmax: T(e)
      });
      const s = this.state.get(e);
      s.m = s.m.mul(this.beta1).add(t.mul(1 - this.beta1)), s.v = s.v.mul(this.beta2).add(t.mul(t).mul(1 - this.beta2));
      const n = 1 - Math.pow(this.beta1, this.step_count), a = 1 - Math.pow(this.beta2, this.step_count);
      let o;
      const c = s.m.div(n);
      this.amsgrad ? (s.vmax = s.vmax.maximum(s.v), o = s.vmax.div(a)) : o = s.v.div(a);
      const u = c.div(o.sqrt().add(this.eps)).mul(this.lr), d = e.sub(u);
      e.data = d.data;
    }
  }
};
i(De, "Adam");
let ce = De;
const Ar = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Adam: ce,
  Optimizer: I,
  SGD: ue
}, Symbol.toStringTag, { value: "Module" }));
export {
  F as AccumulateGrad,
  ee as Matmul,
  Y as Mean,
  V as PowInt,
  H as Reshape,
  J as Squeeze,
  X as Sum,
  h as Tensor,
  A as TorchFunction,
  Z as Transpose,
  Q as Unsqueeze,
  gt as __left_index__,
  mt as __right_index__,
  Mt as abs,
  wt as add,
  pt as arange,
  Nt as cos,
  qt as div,
  zt as eq,
  k as eventBus,
  E as events,
  Rt as exp,
  yt as fmod,
  Lt as ge,
  Kt as gt,
  Gt as le,
  ft as linspace,
  kt as log,
  jt as lt,
  St as matmul,
  At as maximum,
  Wt as mean,
  Ot as minimum,
  bt as mul,
  Vt as ne,
  Bt as neg,
  yr as nn,
  Ye as ones,
  _t as ones_like,
  Ar as optim,
  vt as pow,
  Pe as rand,
  lt as randint,
  ht as randn,
  Tt as reciprocal,
  Ct as reshape,
  st as sign,
  Dt as sin,
  Et as sqrt,
  Ft as square,
  xt as sub,
  It as sum,
  Pt as tan,
  $t as transpose,
  Ut as unsqueeze,
  Ze as zeros,
  T as zeros_like
};
//# sourceMappingURL=torch.browser.es.js.map
