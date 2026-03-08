function S(r) {
  return Array.isArray(r[0]) ? r[0] : r;
}
function Te(...r) {
  const e = S(r), t = new u(Array(e.reduce((s, n) => s * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
function K(...r) {
  const e = S(r), t = new u(Array(e.reduce((s, n) => s * n, 1)).fill(Math.random()));
  return t.shape = e, t;
}
function Ce(r, e, t) {
  const s = new u(
    Array(t.reduce((n, i) => n * i, 1)).fill(Math.floor(Math.random() * (e - r) + r))
  );
  return s.shape = t, s;
}
function se(...r) {
  const e = S(r), t = new u(Array(e.reduce((s, n) => s * n, 1)).fill(1));
  return t.shape = e, t;
}
function ne(...r) {
  const e = S(r), t = new u(Array(e.reduce((s, n) => s * n, 1)).fill(0));
  return t.shape = e, t;
}
function Ue(r) {
  return se(r.shape);
}
function R(r) {
  return ne(r.shape);
}
function Le(r, e, t) {
  const s = [], n = (e - r) / (t - 1);
  for (let i = 0; i < t - 1; i++)
    s.push(r + i * n);
  return s.push(e), new u(s);
}
function Se(r, e = void 0, t = 1) {
  const s = [];
  for (let n = r; n < e; n += t)
    s.push(n);
  return new u(s);
}
let ae = 0;
const z = () => ae++, O = new EventTarget(), q = {
  TENSOR_BEFORE_BACKWARD: "tensor.beforeBackward",
  TENSOR_AFTER_BACKWARD: "tensor.afterBackward",
  OPERATION_BEFORE_FORWARD: "operation.beforeForward",
  OPERATION_AFTER_FORWARD: "operation.afterForward",
  OPERATION_BEFORE_BACKWARD: "operation.beforeBackward",
  OPERATION_AFTER_BACKWARD: "operation.afterBackward",
  OPERATION_BEFORE_ACCUMULATE_GRAD: "operation.beforeAccumulateGrad",
  OPERATION_AFTER_ACCUMULATE_GRAD: "operation.afterAccumulateGrad"
};
function ie(...r) {
  for (const e of r)
    if (e instanceof u && e.requires_grad)
      return !0;
  return !1;
}
class k {
  id = z();
  next_functions = [];
  saved_tensors = [];
  _retained_tensors = [];
  forward(...e) {
    const t = ie(...e);
    O.dispatchEvent(new CustomEvent(q.OPERATION_BEFORE_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e
      }
    }));
    const s = this._forward(...e);
    return O.dispatchEvent(new CustomEvent(q.OPERATION_AFTER_FORWARD, {
      detail: {
        operation: this,
        requires_grad: t,
        args: e,
        result: s
      }
    })), s;
  }
  backward(e) {
    O.dispatchEvent(new CustomEvent(q.OPERATION_BEFORE_BACKWARD, { detail: { operation: this, dz: e } }));
    for (const t of this._retained_tensors)
      t.grad || (t.grad = new u(new Array(t.dataLength()).fill(0))), t.grad = t.grad.add(e);
    this._backward(e), O.dispatchEvent(new CustomEvent(q.OPERATION_AFTER_BACKWARD, { detail: { operation: this, dz: e } }));
  }
}
class oe extends k {
  _forward(...e) {
    throw new Error("NullOp should not be called");
  }
  _backward(e) {
  }
}
const x = new oe();
class D extends k {
}
class $ extends k {
}
class C extends D {
  variable;
  _forward(e) {
    return this.variable = e, e;
  }
  _backward(e) {
    this.variable.grad || (this.variable.grad = R(this.variable)), O.dispatchEvent(new CustomEvent(q.OPERATION_BEFORE_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } })), this.variable.grad = this.variable.grad.add(e), O.dispatchEvent(new CustomEvent(q.OPERATION_AFTER_ACCUMULATE_GRAD, { detail: { operation: this, dz: e } }));
  }
}
const V = /* @__PURE__ */ new Map(), P = /* @__PURE__ */ new Map();
function A(r, e) {
  V.set(r, e);
}
function v(r) {
  const e = V.get(r);
  if (!e)
    throw new Error(`Operation '${r}' is not registered.`);
  return e;
}
function j(r) {
  const e = P.get(r);
  return e || (P.set(r, new (v(r))()), P.get(r));
}
function ue(r) {
  if (ArrayBuffer.isView(r))
    return [r.length];
  const e = [];
  for (; Array.isArray(r); )
    e.push(r.length), r = r[0];
  return e;
}
function H(r) {
  return Array.isArray(r) ? r.flatMap((e) => H(e)) : ArrayBuffer.isView(r) ? Array.from(r) : [r];
}
class u {
  // Auto-generated ID
  id = z();
  // Optional user-defined name
  name = null;
  data;
  _shape;
  grad_fn = null;
  grad = null;
  requires_grad;
  constructor(e, t = {}, s = {}) {
    if (this.data = H(e), this.requires_grad = t.requires_grad ?? !1, t.name && (this.name = t.name), this._shape = s.shape ?? ue(e), this.grad_fn = s.operation ?? null, this.requires_grad && !this.grad_fn) {
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
    const t = this.data, s = (n) => {
      const i = this.shape[n], a = new Array(i), o = n === this.shape.length - 1;
      for (let h = 0; h < i; h++)
        o ? a[h] = t[e++] : a[h] = s(n + 1);
      return a;
    };
    return s(0);
  }
  dataLength() {
    return this.data.length;
  }
  set shape(e) {
    this._shape = e;
  }
  _executeUnaryOp(e) {
    return (this.requires_grad ? new (v(e))() : j(e)).forward(this);
  }
  _executeBinaryOp(e, t) {
    return typeof t == "number" && (t = new u(t)), (this.requires_grad || t.requires_grad ? new (v(e))() : j(e)).forward(this, t);
  }
  _executeOpRaw(e, ...t) {
    return new (v(e))().forward(this, ...t);
  }
  item() {
    if (this.dataLength() !== 1)
      throw new Error("Tensor.item() is only valid for scalars");
    return this.toArray()[0];
  }
  detach() {
    return new u(this.data, { requires_grad: !1 }, { shape: this.shape });
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
        e = new u(1);
      }
      this.grad_fn && (O.dispatchEvent(new CustomEvent(q.TENSOR_BEFORE_BACKWARD, { detail: { tensor: this } })), this.grad_fn.backward(e), O.dispatchEvent(new CustomEvent(q.TENSOR_AFTER_BACKWARD, { detail: { tensor: this } })));
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
}
function J(r, e) {
  const t = Math.max(r.length, e.length), s = [...Array(t - r.length).fill(1), ...r], n = [...Array(t - e.length).fill(1), ...e], i = [];
  for (let a = 0; a < t; a++) {
    if (s[a] !== n[a] && s[a] !== 1 && n[a] !== 1)
      throw new Error(`Shape mismatch: ${r} and ${e}`);
    i.push(Math.max(s[a], n[a]));
  }
  return i;
}
function U(r, e) {
  return r.length >= e.length ? r : [...Array(e.length - r.length).fill(1), ...r];
}
function L(r, e, t) {
  let s = 0, n = 1, i = t;
  for (let a = r.length - 1; a >= 0; a--) {
    if (r[a] > 1) {
      const o = i % e[a];
      s = s + o * n;
    }
    n *= r[a], i = Math.floor(i / e[a]);
  }
  return s;
}
function I(r) {
  return (...e) => new (v(r))().forward(...e);
}
function g(r) {
  return (e) => (typeof e == "number" && (e = new u(e)), new (v(r))().forward(e));
}
function l(r) {
  return (e, t) => (typeof e == "number" && (e = new u(e)), typeof t == "number" && (t = new u(t)), new (v(r))().forward(e, t));
}
const De = l("__left_index__"), Pe = l("__right_index__"), Ie = l("add"), Ne = l("sub"), We = l("mul"), Ge = l("div"), Ke = l("pow"), je = l("fmod"), ze = l("maximum"), $e = l("minimum"), Ve = g("log"), He = g("sqrt"), Je = g("exp"), Qe = g("square"), Xe = g("abs"), ce = g("sign"), Ye = g("neg"), Ze = g("reciprocal"), et = I("reshape"), tt = I("unsqueeze"), rt = g("sin"), st = g("cos"), nt = g("tan"), at = g("sum"), it = g("mean"), ot = I("transpose"), ut = l("matmul"), ct = l("lt"), ht = l("gt"), dt = l("le"), lt = l("ge"), _t = l("eq"), pt = l("ne");
function p(r, e, t = null) {
  const s = (a, o, h, c, d, b) => {
    const f = Array(b);
    for (let _ = 0; _ < b; _++) {
      const E = L(o, d, _), w = L(c, d, _);
      f[_] = r(a, h, E, w);
    }
    return f;
  }, n = (a, o, h = null) => {
    const c = J(a.shape, o.shape), d = U(a.shape, c), b = U(o.shape, c), f = c.reduce((_, E) => _ * E, 1);
    return new u(
      s(a.data, d, o.data, b, c, f),
      { requires_grad: a.requires_grad || o.requires_grad },
      { operation: h, shape: c }
    );
  }, i = class extends $ {
    _forward(a, o) {
      return (a.requires_grad || o.requires_grad) && (this.saved_tensors = [a, o]), this.next_functions.push(a.grad_fn ? a.grad_fn : x), this.next_functions.push(o.grad_fn ? o.grad_fn : x), n(a, o, a.requires_grad || o.requires_grad ? this : null);
    }
    _backward(a) {
      const [o, h] = this.saved_tensors, [c, d] = this.next_functions;
      e(o, h, c, d, a);
    }
  };
  return t && A(t, i), i;
}
function m(r, e, t = null) {
  const s = (a, o) => {
    const h = Array(o);
    for (let c = 0; c < o; c++)
      h[c] = r(a, c);
    return h;
  }, n = (a, o = null) => {
    const h = a.dataLength();
    return new u(
      s(a.data, h),
      { requires_grad: a.requires_grad },
      { operation: o, shape: a.shape }
    );
  }, i = class extends D {
    _forward(a) {
      return a.requires_grad && (this.saved_tensors = [a]), this.next_functions.push(a.grad_fn ? a.grad_fn : x), n(a, a.requires_grad ? this : null);
    }
    _backward(a) {
      const [o] = this.saved_tensors, [h] = this.next_functions;
      e(o, h, a);
    }
  };
  return t && A(t, i), i;
}
p(
  (r, e, t, s) => t,
  (r, e, t, s, n) => {
  },
  "__left_index__"
);
p(
  (r, e, t, s) => s,
  (r, e, t, s, n) => {
  },
  "__right_index__"
);
p(
  (r, e, t, s) => r[t] + e[s],
  (r, e, t, s, n) => {
    t.backward(n), s.backward(n);
  },
  "add"
);
p(
  (r, e, t, s) => r[t] - e[s],
  (r, e, t, s, n) => {
    t.backward(n), s.backward(n.mul(new u(-1)));
  },
  "sub"
);
p(
  (r, e, t, s) => r[t] * e[s],
  (r, e, t, s, n) => {
    t.backward(n.mul(e)), s.backward(n.mul(r));
  },
  "mul"
);
p(
  (r, e, t, s) => r[t] / e[s],
  (r, e, t, s, n) => {
    t.backward(n.div(e)), s.backward(n.mul(r).mul(new u(-1)).div(e).div(e));
  },
  "div"
);
p(
  (r, e, t, s) => Math.pow(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(e).mul(r.pow(e.sub(new u(1))))), s.backward(n.mul(r.pow(e)).mul(r.log()));
  },
  "pow"
);
p(
  (r, e, t, s) => r[t] % e[s],
  (r, e, t, s, n) => {
    t.backward(n);
  },
  "fmod"
);
p(
  (r, e, t, s) => Math.max(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(r.ge(e))), s.backward(n.mul(e.gt(r)));
  },
  "maximum"
);
p(
  (r, e, t, s) => Math.min(r[t], e[s]),
  (r, e, t, s, n) => {
    t.backward(n.mul(r.le(e))), s.backward(n.mul(e.lt(r)));
  },
  "minimum"
);
function he(r, e, t = null) {
  const s = new Array(r.dataLength());
  for (let n = 0; n < s.length; n++)
    s[n] = Math.pow(r.data[n], e);
  return new u(
    s,
    { requires_grad: r.requires_grad },
    { operation: t, shape: r.shape }
  );
}
class de extends k {
  n;
  _forward(e, t) {
    return e.requires_grad && (this.saved_tensors = [e], this.n = t), this.next_functions.push(e.grad_fn ? e.grad_fn : x), he(e, t, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, s = this.n, [n] = this.next_functions;
    n.backward(e.mul(s).mul(t.pow(s - 1)));
  }
}
A("powint", de);
m(
  (r, e) => Math.log(r[e]),
  (r, e, t) => {
    e.backward(t.mul(new u(1).div(r)));
  },
  "log"
);
m(
  (r, e) => Math.sqrt(r[e]),
  (r, e, t) => {
    e.backward(t.mul(new u(1).div(r.sqrt()).div(2)));
  },
  "sqrt"
);
m(
  (r, e) => Math.exp(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.exp())));
  },
  "exp"
);
m(
  (r, e) => r[e] * r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(r).mul(new u(2))));
  },
  "square"
);
m(
  (r, e) => Math.abs(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(ce(r))));
  },
  "abs"
);
m(
  (r, e) => Math.sign(r[e]),
  (r, e, t) => {
    e.backward(0);
  },
  "sign"
);
m(
  (r, e) => -r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(new u(-1))));
  },
  "neg"
);
m(
  (r, e) => 1 / r[e],
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.pow(-2))).neg());
  },
  "reciprocal"
);
class le extends k {
  _forward(e, t) {
    const s = e.dataLength(), n = t.reduce((i, a) => i * a, 1);
    if (s !== n)
      throw new Error("Shape mismatch: " + e.shape + " and " + t);
    if (e.requires_grad && (this.saved_tensors = [e]), e.grad_fn)
      this.next_functions.push(e.grad_fn);
    else if (e.requires_grad) {
      const i = new C();
      i.variable = e, this.next_functions.push(i);
    } else
      this.next_functions.push(x);
    return new u(
      e.data,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: t }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.reshape(t.shape));
  }
}
A("reshape", le);
class _e extends k {
  _forward(e, t) {
    if (e.requires_grad && (this.saved_tensors = [e]), e.grad_fn)
      this.next_functions.push(e.grad_fn);
    else if (e.requires_grad) {
      const n = new C();
      n.variable = e, this.next_functions.push(n);
    } else
      this.next_functions.push(x);
    t < 0 && (t += e.shape.length + 1);
    const s = [...e.shape];
    return s.splice(t, 0, 1), new u(
      e.data,
      { requires_grad: e.requires_grad },
      { operation: e.requires_grad ? this : null, shape: s }
    );
  }
  _backward(e) {
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(e.reshape(t.shape));
  }
}
A("unsqueeze", _e);
m(
  (r, e) => Math.sin(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.cos())));
  },
  "sin"
);
m(
  (r, e) => Math.cos(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.sin().neg())));
  },
  "cos"
);
m(
  (r, e) => Math.tan(r[e]),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.cos().pow(-2))));
  },
  "tan"
);
function pe(r, e = null) {
  return new u(
    r.toFlatArray().reduce((t, s) => t + s, 0),
    { requires_grad: r.requires_grad },
    { operation: e }
  );
}
class fe extends D {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : x), pe(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(R(t).add(e.item()));
  }
}
A("sum", fe);
function ge(r, e = null) {
  return new u(
    r.toFlatArray().reduce((t, s) => t + s, 0) / r.dataLength(),
    { requires_grad: r.requires_grad },
    { operation: e }
  );
}
class me extends D {
  _forward(e) {
    return e.requires_grad && (this.saved_tensors = [e]), this.next_functions.push(e.grad_fn ? e.grad_fn : x), ge(e, e.requires_grad ? this : null);
  }
  _backward(e) {
    const [t] = this.saved_tensors, [s] = this.next_functions;
    s.backward(R(t).add(e.item() / t.dataLength()));
  }
}
A("mean", me);
function we(r, e, t, s = null) {
  if (r.shape.length + e < 0 || r.shape.length + t < 0)
    throw new Error(`Transpose: Dimension out of range (${e} and ${t})`);
  e = e < 0 ? r.shape.length + e : e, t = t < 0 ? r.shape.length + t : t;
  const n = [...r.shape];
  [n[e], n[t]] = [n[t], n[e]];
  const i = r.dataLength(), a = new Array(i), o = new Array(r.shape.length), h = new Array(n.length);
  for (let c = r.shape.length - 1, d = 1; c >= 0; c--)
    o[c] = d, d *= r.shape[c];
  for (let c = n.length - 1, d = 1; c >= 0; c--)
    h[c] = d, d *= n[c];
  for (let c = 0; c < i; c++) {
    let d = c, b = 0;
    for (let f = 0; f < n.length; f++) {
      const _ = h[f], E = Math.floor(d / _);
      d %= _;
      let w = f;
      f === e ? w = t : f === t && (w = e), b += E * o[w];
    }
    a[c] = r.data[b];
  }
  return new u(
    a,
    { requires_grad: r.requires_grad },
    { operation: s, shape: n }
  );
}
class xe extends k {
  dim0;
  dim1;
  _forward(e, t, s) {
    return e.requires_grad && (this.saved_tensors = [e], this.dim0 = t, this.dim1 = s), this.next_functions.push(e.grad_fn ? e.grad_fn : x), we(e, t, s, this);
  }
  _backward(e) {
    const [t] = this.saved_tensors, s = this.dim0, n = this.dim1, [i] = this.next_functions;
    i.backward(e.transpose(s, n));
  }
}
A("transpose", xe);
function be(r, e, t = null) {
  if (r.shape.length == 1 && e.shape.length == 1)
    return r.mul(e).sum();
  const s = r.shape.length == 1, n = e.shape.length == 1, i = s ? [1, r.shape[0]] : r.shape, a = n ? [e.shape[0], 1] : e.shape;
  if (i[i.length - 1] != a[a.length - 2])
    throw new Error("Shape mismatch: " + r.shape + " and " + e.shape);
  const o = J(i.slice(0, -2), a.slice(0, -2)).concat([
    i[i.length - 2],
    a[a.length - 1]
  ]), h = o.reduce((y, B) => y * B, 1), c = new Array(h).fill(0), d = U(i, o), b = U(a, o), f = o[o.length - 2], _ = o[o.length - 1], E = i[i.length - 1];
  for (let y = 0; y < h; y++) {
    const B = y % (f * _), Z = Math.floor(B / _), ee = B % _;
    let te = L(d, o, y - ee), re = L(b, o, y - Z * _), G = 0;
    for (let T = 0; T < E; T++)
      G += r.data[te + T] * e.data[re + T * _];
    c[y] = G;
  }
  let w = [...o];
  return s && (w = w.slice(0, -2).concat([o[o.length - 1]])), n && (w = w.slice(0, -1)), new u(
    c,
    { requires_grad: r.requires_grad || e.requires_grad },
    { operation: t, shape: w }
  );
}
class Ae extends $ {
  _forward(e, t) {
    return (e.requires_grad || t.requires_grad) && (this.saved_tensors = [e, t]), this.next_functions.push(e.grad_fn ? e.grad_fn : x), this.next_functions.push(t.grad_fn ? t.grad_fn : x), be(e, t, e.requires_grad || t.requires_grad ? this : null);
  }
  _backward(e) {
    const [t, s] = this.saved_tensors, [n, i] = this.next_functions;
    if (t.shape.length == 1 && s.shape.length == 1) {
      n.backward(e), i.backward(e);
      return;
    }
    if (t.shape.length == 1) {
      const a = e.unsqueeze(0), o = t.unsqueeze(0);
      n.backward(a.matmul(s.transpose(-2, -1))), i.backward(o.transpose(0, 1).matmul(a));
      return;
    }
    if (s.shape.length == 1) {
      const a = e.unsqueeze(0), o = s.unsqueeze(1);
      n.backward(a.matmul(o.transpose(0, 1))), i.backward(t.transpose(-2, -1).matmul(a));
      return;
    }
    n.backward(e.matmul(s.transpose(-2, -1))), i.backward(t.transpose(-2, -1).matmul(e));
  }
}
A("matmul", Ae);
p(
  (r, e, t, s) => r[t] < e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "lt"
);
p(
  (r, e, t, s) => r[t] > e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "gt"
);
p(
  (r, e, t, s) => r[t] <= e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "le"
);
p(
  (r, e, t, s) => r[t] >= e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "ge"
);
p(
  (r, e, t, s) => r[t] == e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "eq"
);
p(
  (r, e, t, s) => r[t] != e[s] ? 1 : 0,
  (r, e, t, s) => {
  },
  "ne"
);
m(
  (r, e) => Math.max(r[e], 0),
  (r, e, t) => {
    e.backward(t.mul(t.mul(r.gt(0))));
  },
  "relu"
);
m(
  (r, e) => 1 / (1 + Math.exp(-r[e])),
  (r, e, t) => {
    const s = r.sigmoid();
    e.backward(s.mul(s.mul(-1).add(1)).mul(t));
  },
  "sigmoid"
);
class F extends u {
  constructor(e, t = {
    requires_grad: !0
  }, s = {}) {
    e instanceof u ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : e instanceof F ? super(e.data, { requires_grad: !0 }, { shape: e.shape }) : super(e, t, s);
  }
}
class M {
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
    t instanceof F ? this.register_parameter(e, t) : this.register_module(e, t);
  }
  parameters() {
    let e = Object.values(this._parameters);
    for (const t of Object.values(this._modules))
      e = e.concat(t.parameters());
    return e;
  }
}
class ye extends M {
  weight;
  bias;
  constructor(e, t) {
    super();
    const s = Math.sqrt(1 / e);
    this.weight = new F(
      K([t, e]).mul(2 * s).sub(s)
    ), this.bias = new F(
      K([t]).mul(2 * s).sub(s)
    ), this.register("weight", this.weight), this.register("bias", this.bias);
  }
  forward(e) {
    return e.matmul(this.weight.transpose(0, 1)).add(this.bias);
  }
}
class Oe extends M {
  constructor() {
    super();
  }
  forward(e) {
    return X(e);
  }
}
class qe extends M {
  constructor() {
    super();
  }
  forward(e) {
    return Y(e);
  }
}
class ve extends M {
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
}
class N {
}
class Ee extends N {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).pow(2).mean();
  }
}
class ke extends N {
  constructor() {
    super();
  }
  forward(e, t) {
    return e.sub(t).abs().mean();
  }
}
class Re extends N {
  weight;
  constructor(e = null) {
    super(), this.weight = e;
  }
  forward(e, t) {
    const s = t.mul(e.log()), n = t.neg().add(1).mul(e.neg().add(1).log()), i = s.add(n).neg().mean();
    return this.weight ? i.mul(this.weight) : i;
  }
}
function Q(r) {
  return (e) => (typeof e == "number" && (e = new u(e)), new (v(r))().forward(e));
}
const X = Q("relu"), Y = Q("sigmoid"), Fe = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  relu: X,
  sigmoid: Y
}, Symbol.toStringTag, { value: "Module" })), ft = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  BCELoss: Re,
  L1Loss: ke,
  Linear: ye,
  MSELoss: Ee,
  Module: M,
  Parameter: F,
  ReLU: Oe,
  Sequential: ve,
  Sigmoid: qe,
  functional: Fe
}, Symbol.toStringTag, { value: "Module" }));
class W {
  params;
  defaults;
  constructor(e, t) {
    this.params = e, this.defaults = t;
  }
  zero_grad() {
    for (const e of this.params)
      e.grad = null;
  }
}
class Me extends W {
  state = /* @__PURE__ */ new Map();
  lr;
  momentum;
  dampening;
  weight_decay;
  nesterov;
  maximize;
  constructor(e, t = 1e-3, s = 0, n = 0, i = 0, a = !1, o = !1) {
    super(e, {}), this.lr = t, this.momentum = s, this.dampening = n, this.weight_decay = i, this.nesterov = a, this.maximize = o;
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
}
class Be extends W {
  state = /* @__PURE__ */ new Map();
  step_count = 0;
  lr;
  beta1;
  beta2;
  eps;
  weight_decay;
  amsgrad;
  maximize;
  constructor(e, t = 1e-3, s = [0.9, 0.999], n = 1e-8, i = 0, a = !1, o = !1) {
    super(e, {}), this.lr = t, this.beta1 = s[0], this.beta2 = s[1], this.eps = n, this.weight_decay = i, this.amsgrad = a, this.maximize = o;
  }
  step() {
    this.step_count += 1;
    for (const e of this.params) {
      let t = this.maximize ? e.grad.mul(-1) : e.grad;
      this.weight_decay !== 0 && (t = t.add(e.mul(this.weight_decay))), this.state.has(e) || this.state.set(e, {
        m: R(e),
        v: R(e),
        vmax: R(e)
      });
      const s = this.state.get(e);
      s.m = s.m.mul(this.beta1).add(t.mul(1 - this.beta1)), s.v = s.v.mul(this.beta2).add(t.mul(t).mul(1 - this.beta2));
      const n = 1 - Math.pow(this.beta1, this.step_count), i = 1 - Math.pow(this.beta2, this.step_count);
      let a;
      const o = s.m.div(n);
      this.amsgrad ? (s.vmax = s.vmax.maximum(s.v), a = s.vmax.div(i)) : a = s.v.div(i);
      const h = o.div(a.sqrt().add(this.eps)).mul(this.lr), c = e.sub(h);
      e.data = c.data;
    }
  }
}
const gt = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Adam: Be,
  Optimizer: W,
  SGD: Me
}, Symbol.toStringTag, { value: "Module" }));
export {
  C as AccumulateGrad,
  Ae as Matmul,
  me as Mean,
  de as PowInt,
  le as Reshape,
  fe as Sum,
  u as Tensor,
  k as TorchFunction,
  xe as Transpose,
  _e as Unsqueeze,
  De as __left_index__,
  Pe as __right_index__,
  Xe as abs,
  Ie as add,
  Se as arange,
  st as cos,
  Ge as div,
  _t as eq,
  O as eventBus,
  q as events,
  Je as exp,
  je as fmod,
  lt as ge,
  ht as gt,
  dt as le,
  Le as linspace,
  Ve as log,
  ct as lt,
  ut as matmul,
  ze as maximum,
  it as mean,
  $e as minimum,
  We as mul,
  pt as ne,
  Ye as neg,
  ft as nn,
  se as ones,
  Ue as ones_like,
  gt as optim,
  Ke as pow,
  K as rand,
  Ce as randint,
  Te as randn,
  Ze as reciprocal,
  et as reshape,
  ce as sign,
  rt as sin,
  He as sqrt,
  Qe as square,
  Ne as sub,
  at as sum,
  nt as tan,
  ot as transpose,
  tt as unsqueeze,
  ne as zeros,
  R as zeros_like
};
