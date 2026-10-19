"""
A tiny expression DSL over Blender shader nodes.

Writing a numerical integrator directly with ``nodes.new(...)`` and
``links.new(...)`` is unreadable and unmaintainable -- the lensing shader alone
is roughly a hundred nodes.  This module wraps sockets in an ``Expr`` object
with Python operators, so the shader source ends up looking like the maths:

    r      = P.length()
    r_hat  = P / r
    h2     = P.cross(v).dot(P.cross(v))
    accel  = r_hat * (-1.5 * h2 / r**4)

Every operation appends the appropriate Math / Vector Math node and returns a
new ``Expr``.  Scalars and tuples are promoted automatically, and the builder
lays the graph out left-to-right so the resulting .blend is actually readable
when the user opens it.
"""

from __future__ import annotations

from typing import Iterable

import bpy


# ---------------------------------------------------------------------------

class Expr:
    """A node socket you can do arithmetic on."""

    __slots__ = ("b", "socket", "depth")

    def __init__(self, builder: "NodeBuilder", socket, depth: int = 0):
        self.b = builder
        self.socket = socket
        self.depth = depth

    # -- type helpers -------------------------------------------------------
    @property
    def is_vector(self) -> bool:
        return self.socket.type == "VECTOR"

    # -- arithmetic ---------------------------------------------------------
    def _bin(self, op_v: str, op_s: str, other, swap: bool = False) -> "Expr":
        other_e = self.b.wrap(other)
        vector = self.is_vector or other_e.is_vector
        a, c = (other_e, self) if swap else (self, other_e)
        return self.b.vmath(op_v, a, c) if vector else self.b.math(op_s, a, c)

    def __add__(self, o):  return self._bin("ADD", "ADD", o)
    def __radd__(self, o): return self._bin("ADD", "ADD", o, swap=True)
    def __sub__(self, o):  return self._bin("SUBTRACT", "SUBTRACT", o)
    def __rsub__(self, o): return self._bin("SUBTRACT", "SUBTRACT", o, swap=True)
    def __mul__(self, o):  return self._bin("MULTIPLY", "MULTIPLY", o)
    def __rmul__(self, o): return self._bin("MULTIPLY", "MULTIPLY", o, swap=True)
    def __truediv__(self, o):  return self._bin("DIVIDE", "DIVIDE", o)
    def __rtruediv__(self, o): return self._bin("DIVIDE", "DIVIDE", o, swap=True)
    def __neg__(self):
        return self.b.vmath("SCALE", self, scale=-1.0) if self.is_vector \
            else self.b.math("MULTIPLY", self, -1.0)

    def __pow__(self, k):
        if self.is_vector:
            raise TypeError("pow() is scalar-only")
        return self.b.math("POWER", self, k)

    # -- vector ops ---------------------------------------------------------
    def dot(self, o) -> "Expr":    return self.b.vmath("DOT_PRODUCT", self, o)
    def cross(self, o) -> "Expr":  return self.b.vmath("CROSS_PRODUCT", self, o)
    def length(self) -> "Expr":    return self.b.vmath("LENGTH", self)
    def normalize(self) -> "Expr": return self.b.vmath("NORMALIZE", self)
    def scale(self, s) -> "Expr":  return self.b.vmath("SCALE", self, scale=s)

    def xyz(self) -> tuple["Expr", "Expr", "Expr"]:
        n = self.b.new("ShaderNodeSeparateXYZ", self.depth + 1)
        self.b.link(self, n.inputs[0])
        d = self.depth + 1
        return (Expr(self.b, n.outputs[0], d),
                Expr(self.b, n.outputs[1], d),
                Expr(self.b, n.outputs[2], d))

    # -- scalar ops ---------------------------------------------------------
    def sqrt(self) -> "Expr":     return self.b.math("SQRT", self)
    def abs(self) -> "Expr":      return self.b.math("ABSOLUTE", self)
    def min(self, o) -> "Expr":   return self.b.math("MINIMUM", self, o)
    def max(self, o) -> "Expr":   return self.b.math("MAXIMUM", self, o)
    def clamp01(self) -> "Expr":
        n = self.b.math("MAXIMUM", self, 0.0)
        return self.b.math("MINIMUM", n, 1.0)

    def gt(self, o) -> "Expr":    return self.b.math("GREATER_THAN", self, o)
    def lt(self, o) -> "Expr":    return self.b.math("LESS_THAN", self, o)

    def smoothstep(self, edge0, edge1) -> "Expr":
        """Hermite smoothstep -- the Math node has no built-in equivalent."""
        t = ((self - edge0) / (self.b.wrap(edge1) - edge0)).clamp01()
        return t * t * (3.0 - t * 2.0)

    def mix(self, a, b_) -> "Expr":
        """Linear blend: ``self`` is the factor, 0 -> a, 1 -> b_."""
        return self.b.mix(self, a, b_)


# ---------------------------------------------------------------------------

class NodeBuilder:
    """Creates nodes in a tree and keeps them roughly laid out by depth."""

    def __init__(self, tree, x0: float = 0.0, y0: float = 0.0,
                 dx: float = 190.0, dy: float = -190.0):
        self.tree = tree
        self.x0, self.y0, self.dx, self.dy = x0, y0, dx, dy
        self._rows: dict[int, int] = {}

    # -- plumbing -----------------------------------------------------------
    def new(self, idname: str, depth: int = 0, **props):
        node = self.tree.nodes.new(idname)
        row = self._rows.get(depth, 0)
        self._rows[depth] = row + 1
        node.location = (self.x0 + depth * self.dx, self.y0 + row * self.dy)
        for k, v in props.items():
            setattr(node, k, v)
        return node

    def link(self, src, dst) -> None:
        socket = src.socket if isinstance(src, Expr) else src
        self.tree.links.new(socket, dst)

    def wrap(self, value) -> Expr:
        """Promote a float / 3-tuple / socket to an ``Expr``."""
        if isinstance(value, Expr):
            return value
        if isinstance(value, (int, float)):
            return self.value(float(value))
        if isinstance(value, Iterable):
            return self.vector(tuple(value))
        return Expr(self, value)                       # a raw bpy socket

    # -- leaves -------------------------------------------------------------
    def value(self, v: float, label: str = "") -> Expr:
        n = self.new("ShaderNodeValue", 0, label=label or f"{v:g}")
        n.outputs[0].default_value = v
        return Expr(self, n.outputs[0], 0)

    def vector(self, v, label: str = "") -> Expr:
        n = self.new("ShaderNodeCombineXYZ", 0, label=label)
        for i in range(3):
            n.inputs[i].default_value = float(v[i])
        return Expr(self, n.outputs[0], 0)

    def combine(self, x, y, z) -> Expr:
        d = 1 + max(getattr(e, "depth", 0) for e in (x, y, z) if isinstance(e, Expr))
        n = self.new("ShaderNodeCombineXYZ", d)
        for i, c in enumerate((x, y, z)):
            if isinstance(c, Expr):
                self.link(c, n.inputs[i])
            else:
                n.inputs[i].default_value = float(c)
        return Expr(self, n.outputs[0], d)

    def geometry(self, out: str) -> Expr:
        node = getattr(self, "_geo", None)
        if node is None:
            node = self._geo = self.new("ShaderNodeNewGeometry", 0)
        return Expr(self, node.outputs[out], 0)

    def attribute(self, name: str, out: str = "Fac") -> Expr:
        n = self.new("ShaderNodeAttribute", 0, attribute_name=name)
        return Expr(self, n.outputs[out], 0)

    # -- operators ----------------------------------------------------------
    def _depth(self, *args) -> int:
        return 1 + max([a.depth for a in args if isinstance(a, Expr)] or [0])

    def math(self, op: str, a, b=None, c=None) -> Expr:
        d = self._depth(a, b, c)
        n = self.new("ShaderNodeMath", d, operation=op)
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, Expr):
                self.link(v, n.inputs[i])
            else:
                n.inputs[i].default_value = float(v)
        return Expr(self, n.outputs[0], d)

    def vmath(self, op: str, a, b=None, c=None, scale=None) -> Expr:
        d = self._depth(a, b, c, scale if isinstance(scale, Expr) else None)
        n = self.new("ShaderNodeVectorMath", d, operation=op)
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            v = self.wrap(v) if not isinstance(v, Expr) else v
            self.link(v, n.inputs[i])
        if scale is not None:
            sock = n.inputs["Scale"]
            if isinstance(scale, Expr):
                self.link(scale, sock)
            else:
                sock.default_value = float(scale)
        out = n.outputs["Value"] if op in {"DOT_PRODUCT", "LENGTH", "DISTANCE"} \
            else n.outputs["Vector"]
        return Expr(self, out, d)

    def mix(self, fac, a, b) -> Expr:
        """Blend two scalars or two vectors by ``fac``."""
        vector = any(isinstance(v, Expr) and v.is_vector for v in (a, b))
        d = self._depth(fac, a, b)
        n = self.new("ShaderNodeMix", d, data_type="VECTOR" if vector else "FLOAT")
        if isinstance(fac, Expr):
            self.link(fac, n.inputs["Factor"])
        else:
            n.inputs["Factor"].default_value = float(fac)
        key = "B" if vector else "A"          # sockets are name-duplicated
        pairs = [s for s in n.inputs if s.name in ("A", "B")]
        pairs = [s for s in pairs if (s.type == "VECTOR") == vector]
        for sock, v in zip(pairs, (a, b)):
            if isinstance(v, Expr):
                self.link(v, sock)
            elif vector:
                sock.default_value = tuple(v)
            else:
                sock.default_value = float(v)
        out = [o for o in n.outputs if (o.type == "VECTOR") == vector][0]
        return Expr(self, out, d)

    def select(self, cond: Expr, if_true, if_false) -> Expr:
        """``cond`` is 0/1; returns ``if_true`` when 1."""
        return self.mix(cond, if_false, if_true)
