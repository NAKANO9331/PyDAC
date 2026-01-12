"""Tensor class for PyDAC"""


import numpy as np

from typing import Union, List, Optional


class Tensor:

    """Python Tensor abstraction for automatic conversion to C++ dacpp::Matrix/Vector"""

    def __init__(
        self,
        data: Union[np.ndarray, List],
        shape: Optional[List[int]] = None,
        dtype: str = "float"
    ):
        """
        Initialize Tensor

        Args:
            data: NumPy array or Python list
            shape: Shape (required if data is a list)
            dtype: Data type
        """
        if isinstance(data, np.ndarray):
            self._data = data
            self._shape = list(data.shape)
            self._dtype = str(data.dtype)
        else:
            if shape is None:
                raise ValueError("shape is required when data is a list")
            self._data = np.array(data, dtype=dtype)
            self._shape = shape
            self._dtype = dtype

    @property
    def shape(self) -> List[int]:
        """Get shape"""

        return self._shape

    @property
    def data(self) -> np.ndarray:
        """Get data"""

        return self._data

    @property
    def dtype(self) -> str:
        """Get data type"""

        return self._dtype

    def _dtype_to_cpp(self) -> str:
        """Convert numpy dtype to C++ type"""

        dtype_str = str(self._dtype)
        # Map numpy dtypes to C++ types
        if 'float32' in dtype_str or dtype_str == 'float32':
            return 'float'
        elif 'float64' in dtype_str or dtype_str == 'float64':
            return 'double'
        elif 'int32' in dtype_str or dtype_str == 'int32':
            return 'int'
        elif 'int64' in dtype_str or dtype_str == 'int64':
            return 'long'
        elif dtype_str in ['float', 'double', 'int', 'long']:
            return dtype_str
        else:
            # Default to float
            return 'float'

    def to_cpp_init(self, var_name: Optional[str] = None) -> str:
        """
        Generate C++ initialization code

        Args:
            var_name: Variable name (None for expression)

        Returns:
            C++ initialization code string
        """
        # Convert dtype to C++ type
        cpp_dtype = self._dtype_to_cpp()

        # Determine type
        if len(self._shape) == 1:
            cpp_type = f"dacpp::Vector<{cpp_dtype}>"
        else:
            cpp_type = f"dacpp::Matrix<{cpp_dtype}>"

        # Generate shape
        shape_str = "{" + ", ".join(map(str, self._shape)) + "}"

        # Generate data
        flat_data = self._data.flatten()
        data_str = "{" + ", ".join(map(str, flat_data)) + "}"

        # Generate code
        if var_name:
            return f"{cpp_type} {var_name}({shape_str}, {data_str});"
        else:
            return f"{cpp_type}({shape_str}, {data_str})"

