"""Shell DSL for PyDAC"""


from typing import List, Dict, Tuple


class Shell:

    """Shell function DSL"""

    def __init__(self, name: str):
        """
        Initialize Shell

        Args:
            name: Shell function name
        """
        self.name = name
        self.splits: Dict[str, Tuple[int, int]] = {}  # {name: (size, stride)}
        self.indices: List[str] = []
        self.bindings: Dict[str, str] = {}  # {split_name: index_name}
        self.slices: List[Tuple[str, List[str]]] = []  # [(tensor_name, operators), ...]
        self.params: List[Tuple[str, str, bool]] = []  # [(name, type, is_const), ...]
        self.definition_order: List[Tuple[str, str]] = []  # [("index", "idx"), ("split", "sp"), ...]

    def add_split(self, name: str, size: int, stride: int) -> 'Shell':
        """Add split definition"""

        self.splits[name] = (size, stride)
        self.definition_order.append(("split", name))
        return self

    def add_index(self, name: str) -> 'Shell':
        """Add index definition"""

        if name not in self.indices:
            self.indices.append(name)
            self.definition_order.append(("index", name))
        return self

    def bind(self, split_name: str, index_name: str) -> 'Shell':
        """Bind split and index"""

        if split_name not in self.splits:
            raise ValueError(f"Split '{split_name}' not defined")
        if index_name not in self.indices:
            self.add_index(index_name)
        self.bindings[split_name] = index_name
        return self

    def add_slice(self, tensor_name: str, operators: List[str]) -> 'Shell':
        """Add slice expression"""

        self.slices.append((tensor_name, operators))
        return self
    
    def add_param(self, name: str, param_type: str, is_const: bool = True) -> 'Shell':
        """
        Add parameter definition
        
        Args:
            name: Parameter name
            param_type: Parameter type (e.g., "dacpp::Matrix<int>&")
            is_const: Whether parameter is const (default: True)
        """
        self.params.append((name, param_type, is_const))
        return self

    def to_cpp_code(self) -> str:
        """Generate C++ code"""

        code = f"shell dacpp::list {self.name}("

        # Generate parameter list
        params = []
        if self.params:
            # Use explicitly defined parameters
            for name, param_type, is_const in self.params:
                # Check if param_type already contains 'const'
                if is_const and "const" not in param_type:
                    const_str = "const "
                else:
                    const_str = ""
                params.append(f"{const_str}{param_type} {name}")
        else:
            # Fallback: auto-generate from slices (backward compatibility)
            for tensor_name, _ in self.slices:
                params.append(f"const dacpp::Matrix<float>& {tensor_name}")
        code += ", ".join(params) + ") {\n"

        # Generate definitions in the order they were added
        for def_type, name in self.definition_order:
            if def_type == "split" and name in self.splits:
                size, stride = self.splits[name]
                code += f" dacpp::split {name}({size}, {stride});\n"
            elif def_type == "index" and name in self.indices:
                code += f" dacpp::index {name};\n"

        # Generate bindings
        for split_name, index_name in self.bindings.items():
            code += f" binding({split_name}, {index_name});\n"

        # Generate dataList
        code += " dacpp::list dataList{"
        slice_exprs = []
        for tensor_name, operators in self.slices:
            expr = tensor_name
            for op in operators:
                expr += f"[{op}]"
            slice_exprs.append(expr)
        code += ", ".join(slice_exprs) + "};\n"
        code += " return dataList;\n}\n"

        return code

