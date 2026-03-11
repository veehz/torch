import json


class CompactJSONEncoder(json.JSONEncoder):
    """JSON encoder that puts leaf arrays (arrays of non-list items) on single lines."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._indent = kwargs.get("indent", 2)

    def encode(self, o):
        return self._encode(o, 0)

    def _encode(self, o, level):
        indent_str = " " * (self._indent * level)
        next_indent_str = " " * (self._indent * (level + 1))

        if isinstance(o, dict):
            if not o:
                return "{}"
            items = []
            for k, v in o.items():
                key = json.dumps(k)
                val = self._encode(v, level + 1)
                items.append(f"{next_indent_str}{key}: {val}")
            return "{\n" + ",\n".join(items) + f"\n{indent_str}}}"

        elif isinstance(o, list):
            if not o:
                return "[]"
            # Leaf array: no nested lists inside
            if not any(isinstance(item, (list, dict)) for item in o):
                return "[" + ", ".join(self._encode(v, level) for v in o) + "]"
            # Nested array: pretty print
            items = [f"{next_indent_str}{self._encode(v, level + 1)}" for v in o]
            return "[\n" + ",\n".join(items) + f"\n{indent_str}]"

        else:
            return json.dumps(o)
