"""Code metrics for CADQuery code analysis (AST-based, extended vocabulary).

A more complete operation vocabulary sourced from the official CADQuery documentation -- every public
method of the ``Workplane`` and ``Sketch`` fluent APIs, plus the common
constructors. This covers the Sketch API (``sketch``/``segment``/``arc``/...)
that the original list missed, and many operations that never appear in the
benchmark data but are valid CADQuery (``trapezoid``, ``slot``, ``rarray``,
``hull``, ``combine``, ``split``, ``wedge``, ...).

Source (verified June 2026):
  https://cadquery.readthedocs.io/en/latest/apireference.html
  https://cadquery.readthedocs.io/en/latest/classreference.html

Scope: "everything documented" -- includes selectors, IO, and helper methods,
grouped into dedicated categories so non-modeling calls are still counted.
"""
import ast
import io
import re
import textwrap
import tokenize
from typing import Dict, Any

# Complete CADQuery API, categorized. Fluent Workplane + Sketch API (as in
# code_metrics_extended) PLUS the direct-API and selector tiers.
CADQUERY_OPERATIONS = {
    "referencing_ops": [
        # constructors / construction planes
        "Workplane", "workplane", "workplaneFromTagged", "Plane", "Vector",
        "Sketch", "Location",
        # selectors and stack navigation
        "faces", "edges", "vertices", "wires", "solids", "shells", "compounds",
        "val", "vals", "first", "last", "item", "end", "all", "center",
        "eachpoint", "each", "siblings", "ancestors", "select", "tag",
        "reset", "delete", "BoundingBox", "filterBy", "nearestTo",
    ],
    "solid_ops": [
        "box", "sphere", "cylinder", "wedge", "extrude", "revolve", "loft",
        "sweep", "twistExtrude", "interpPlate", "parametricSurface", "section",
    ],
    "sketch_ops": [
        # sketch lifecycle
        "sketch", "finalize", "assemble", "placeSketch",
        # 2D primitives
        "circle", "rect", "ellipse", "ellipseArc", "polygon", "polyline",
        "regularPolygon", "trapezoid", "slot", "slot2D",
        # pen / path drawing
        "line", "lineTo", "hLine", "hLineTo", "vLine", "vLineTo",
        "polarLine", "polarLineTo", "move", "moveTo",
        "threePointArc", "sagittaArc", "radiusArc", "tangentArcPoint", "arc",
        "spline", "splineApprox", "bezier", "parametricCurve",
        "segment", "close", "wire", "edge", "face", "text",
        # offsets / arrays / point sets
        "offset", "offset2D", "pushPoints", "push", "hull",
        "rarray", "parray", "polarArray", "distribute",
    ],
    "refinement_ops": [
        "fillet", "chamfer", "shell", "hole", "cskHole", "cboreHole", "clean",
    ],
    "boolean_ops": [
        "union", "cut", "cutBlind", "cutThruAll", "cutEach", "intersect",
        "combine", "add", "subtract", "split", "fuse",
    ],
    "transform_ops": [
        "translate", "rotate", "rotateAboutCenter", "mirror", "mirrorX",
        "mirrorY", "transformed", "located", "moved",
    ],
    "constraint_ops": [
        "constrain", "solve",
    ],
    "io_ops": [
        "export", "exportSvg", "toSvg", "importDXF",
    ],
    "helper_ops": [
        "apply", "invoke", "map", "filter", "sort", "copy", "replace",
        "newObject", "toPending", "toOCC", "findSolid", "largestDimension",
        "consolidateWires", "copyWorkplane", "size", "remove",
    ],
    # Direct / low-level geometry-construction API (factory methods on
    # Shape/Solid/Wire/Edge/Face/Compound). These create geometry.
    "direct_api_ops": [
        "makeBox", "makeCone", "makeCylinder", "makeLoft", "makeSolid",
        "makeSphere", "makeTorus", "makeWedge", "makeCircle", "makeEllipse",
        "makeHelix", "makePolygon", "makeFromWires", "makeCompound", "makeText",
        "makeBezier", "makeSpline", "makeSplineApprox", "makeTangentArc",
        "makeThreePointArc", "makeNSidedSurface", "makeRuledSurface", "makeShell",
    ],
    # Selector classes (used inside .faces()/.edges()/.vertices()). Selection,
    # not geometry creation -- but legitimate CADQuery operations.
    "selector_ops": [
        "Selector", "NearestToPointSelector", "NearestToShapeSelector",
        "BoxSelector", "BaseDirSelector", "ParallelDirSelector",
        "DirectionSelector", "PerpendicularDirSelector", "TypeSelector",
        "RadiusNthSelector", "CenterNthSelector", "DirectionMinMaxSelector",
        "DirectionNthSelector", "LengthNthSelector", "AreaNthSelector",
        "BinarySelector", "AndSelector", "SumSelector", "SubtractSelector",
        "InverseSelector", "StringSyntaxSelector",
    ],
}

# Token types that aren't "real" tokens for a size metric.
_SKIP_TOKENS = frozenset({
    tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
    tokenize.INDENT, tokenize.DEDENT,
    tokenize.ENCODING, tokenize.ENDMARKER,
})


def count_tokens(code: str) -> int:
    """Count meaningful tokens using Python's tokenizer.

    Comments and layout tokens (newlines, indentation) are excluded; multi-char
    operators like ``==`` and ``**`` count as a single token. Falls back to a
    regex heuristic if the code is not tokenizable.

    Args:
        code: Python code string

    Returns:
        Token count
    """
    if not code or not isinstance(code, str):
        return 0

    # Snippets often arrive uniformly indented (e.g. extracted from a larger
    # block); strip the common leading whitespace so the tokenizer accepts them.
    code = textwrap.dedent(code)

    count = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type in _SKIP_TOKENS or tok.string == "":
                continue
            count += 1
    except (tokenize.TokenError, IndentationError):
        # Incomplete/invalid code: fall back to the original regex heuristic.
        stripped = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        return len(re.findall(r'\b\w+\b|[(){}[\],.:;=+\-*/%<>!&|]', stripped))

    return count


def count_lines(code: str) -> int:
    """Count the number of non-empty lines in the code.

    Args:
        code: Python code string

    Returns:
        Line count (excluding empty lines)
    """
    if not code or not isinstance(code, str):
        return 0

    lines = code.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]

    return len(non_empty_lines)


def _count_operations_regex(code: str, operations: Dict[str, list]) -> Dict[str, Any]:
    """Best-effort, regex-based operation count.

    Used as a fallback for code that cannot be parsed into an AST (truncated,
    syntactically broken, or mis-indented fragments). This mirrors the original
    text-matching heuristic: it can over-count names that appear in comments or
    strings, but it still returns a useful estimate when parsing is impossible.
    """
    operations_by_category = {category: 0 for category in operations}
    detailed_operations: Dict[str, int] = {}
    total_operations = 0

    for category, ops_list in operations.items():
        for op in ops_list:
            pattern = r'\.{}\s*\(|^{}\s*\(|[^\w]{}\s*\('.format(
                re.escape(op), re.escape(op), re.escape(op)
            )
            count = len(re.findall(pattern, code, re.MULTILINE))
            if count > 0:
                detailed_operations[op] = count
                operations_by_category[category] += count
                total_operations += count

    return {
        "total_operations": total_operations,
        "operations_by_category": operations_by_category,
        "detailed_operations": detailed_operations,
        "fallback": "regex",
    }


def _tally_calls(tree, op_to_category: Dict[str, str], detailed: Dict[str, int]) -> None:
    """Walk an AST and add every CADQuery call it finds into ``detailed``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):      # .extrude(...)
            name = func.attr
        elif isinstance(func, ast.Name):          # box(...), Workplane(...)
            name = func.id
        else:
            continue
        if name in op_to_category:
            detailed[name] = detailed.get(name, 0) + 1


def _has_dynamic_exec(tree) -> bool:
    """True if the AST contains an ``exec``/``eval``/``compile`` call.

    Used by ``mode="auto"`` to decide whether string-embedded code is worth
    scanning -- the presence of these calls is an unambiguous signal that op
    code hidden in strings will actually run.
    """
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {"exec", "eval", "compile"}):
            return True
    return False


def _string_bindings(tree) -> Dict[str, str]:
    """Map simple ``name = "<string literal>"`` assignments to their value.

    Lets us resolve ``exec(src)`` when ``src`` was assigned a string literal
    earlier (lightweight constant propagation).
    """
    bindings: Dict[str, str] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    bindings[tgt.id] = node.value.value
    return bindings


# str.format placeholders ({w}, {0}, {}) -> a literal so templates still parse.
_PLACEHOLDER = re.compile(r'\{[^{}]*\}')


def _parse_embedded(text: str):
    """Return an AST for a string that parses as Python code, else None.

    Format placeholders are scrubbed first so template strings remain valid.
    """
    candidate = _PLACEHOLDER.sub('0', text)
    try:
        return ast.parse(textwrap.dedent(candidate))
    except SyntaxError:
        return None


def _tally_embedded(tree, op_to_category: Dict[str, str], detailed: Dict[str, int]) -> None:
    """Add ops hidden in string literals that the program runs or builds.

    Handles ``exec``/``eval``/``compile`` targets (literal or bound-name) and any
    standalone string literal that itself parses as op-bearing Python code.
    """
    bindings = _string_bindings(tree)
    seen = set()

    def consider(text: str):
        if text in seen:
            return
        seen.add(text)
        # Cheap guard: only attempt a parse if an op name appears as a call.
        if not any(re.search(rf'\b{re.escape(op)}\s*\(', text) for op in op_to_category):
            return
        sub = _parse_embedded(text)
        if sub is not None:
            _tally_calls(sub, op_to_category, detailed)

    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {"exec", "eval", "compile"} and node.args):
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                consider(arg.value)
            elif isinstance(arg, ast.Name) and arg.id in bindings:
                consider(bindings[arg.id])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            consider(node.value)


def _assemble(detailed: Dict[str, int], op_to_category: Dict[str, str],
              operations: Dict[str, list], extra: Dict[str, Any] = None) -> Dict[str, Any]:
    """Build the standard result dict from a {op: count} mapping."""
    operations_by_category = {category: 0 for category in operations}
    for op, count in detailed.items():
        operations_by_category[op_to_category[op]] += count
    result = {
        "total_operations": sum(detailed.values()),
        "operations_by_category": operations_by_category,
        "detailed_operations": dict(detailed),
    }
    if extra:
        result.update(extra)
    return result


def count_operations(code: str, operations: Dict[str, list] = None,
                     mode: str = "auto") -> Dict[str, Any]:
    """Count occurrences of specific CADQuery operations in the code.

    Modes answer different questions:

    - ``"auto"`` (default): behaves like ``"static"``, but automatically
      escalates to ``"effective"`` when the code contains an
      ``exec``/``eval``/``compile`` call -- an unambiguous sign that operations
      hidden in strings will actually run.
    - ``"static"``: how many operations does the code *statically call*?
      AST-based, so names in strings/comments/identifiers are ignored. Falls
      back to a regex estimate (marked ``"fallback": "regex"``) when the code
      cannot be parsed.
    - ``"effective"``: how many operations will the code actually *run or build*?
      Extends ``"static"`` by also resolving operations embedded in strings that
      are ``exec``/``eval``/``compile``-ed or assembled as templates.
    - ``"mentions"``: how many times is an operation *named anywhere* as a call,
      including inside comments, docstrings, and strings? Pure regex over text.

    Args:
        code: Python code string
        operations: Dictionary of operation categories and their operations.
                   If None, uses default CADQUERY_OPERATIONS.
        mode: One of ``"auto"``, ``"static"``, ``"effective"``, ``"mentions"``.

    Returns:
        Dictionary containing:
        - total_operations: Total count of all operations
        - operations_by_category: Dict of counts per category
        - detailed_operations: Dict of counts per individual operation
    """
    if mode not in {"auto", "static", "effective", "mentions"}:
        raise ValueError(f"unknown mode: {mode!r}")

    empty = {
        "total_operations": 0,
        "operations_by_category": {},
        "detailed_operations": {},
    }
    if not code or not isinstance(code, str):
        return empty

    if operations is None:
        operations = CADQUERY_OPERATIONS

    op_to_category = {
        op: category
        for category, ops_list in operations.items()
        for op in ops_list
    }

    # "mentions": textual count over the raw source, comments/strings included.
    if mode == "mentions":
        result = _count_operations_regex(code, operations)
        result.pop("fallback", None)
        result["mode"] = "mentions"
        return result

    # Snippets often arrive uniformly indented; strip the common leading
    # whitespace so otherwise-valid module code parses.
    code = textwrap.dedent(code)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Code isn't valid Python; degrade gracefully to a regex estimate.
        return _count_operations_regex(code, operations)

    detailed: Dict[str, int] = {}
    _tally_calls(tree, op_to_category, detailed)

    # "auto" escalates to effective scanning only when exec/eval/compile is used.
    scan_embedded = mode == "effective" or (mode == "auto" and _has_dynamic_exec(tree))
    if scan_embedded:
        _tally_embedded(tree, op_to_category, detailed)
        return _assemble(detailed, op_to_category, operations, {"mode": "effective"})

    return _assemble(detailed, op_to_category, operations)


def compute_code_metrics(code: str) -> Dict[str, Any]:
    """Compute all code metrics for a given code string.

    Args:
        code: Python code string

    Returns:
        Dictionary containing all code metrics:
        - token_count: Number of tokens
        - line_count: Number of non-empty lines
        - total_operations: Total operation count
        - operations_by_category: Operations grouped by category
    """
    if not code or not isinstance(code, str):
        return {
            "token_count": 0,
            "line_count": 0,
            "total_operations": 0,
            "operations_by_category": {}
        }

    token_count = count_tokens(code)
    line_count = count_lines(code)
    op_metrics = count_operations(code)

    return {
        "token_count": token_count,
        "line_count": line_count,
        "total_operations": op_metrics["total_operations"],
        "operations_by_category": op_metrics["operations_by_category"],
    }
