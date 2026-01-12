"""Tests for PyDAC Tensor"""


import pytest

import numpy as np

from pydac.tensor import Tensor


class TestTensor:

    """Test cases for Tensor"""

    def test_tensor_from_numpy(self):
        """Test creating Tensor from NumPy array"""

        data = np.random.rand(10, 10).astype(np.float32)
        tensor = Tensor(data)
        assert tensor.shape == [10, 10]
        assert np.array_equal(tensor.data, data)

    def test_tensor_from_list(self):

        """Test creating Tensor from list"""

        data = [1.0, 2.0, 3.0]
        tensor = Tensor(data, shape=[3])
        assert tensor.shape == [3]
        assert len(tensor.data) == 3

    def test_tensor_to_cpp_init(self):

        """Test C++ initialization code generation"""

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = Tensor(data)
        cpp_code = tensor.to_cpp_init("vec")
        assert "dacpp::Vector<float> vec" in cpp_code
        assert "{3}" in cpp_code

    def test_tensor_matrix(self):

        """Test matrix Tensor"""

        data = np.random.rand(5, 5).astype(np.float32)
        tensor = Tensor(data)
        cpp_code = tensor.to_cpp_init("mat")
        assert "dacpp::Matrix<float> mat" in cpp_code
        assert "{5, 5}" in cpp_code

