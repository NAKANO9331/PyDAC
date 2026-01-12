"""Tests for PyDAC DSL"""


import pytest

from pydac.dsl import Shell, Calc


class TestShell:

    """Test cases for Shell DSL"""

    def test_shell_creation(self):
        """Test Shell creation"""

        shell = Shell("testShell")
        assert shell.name == "testShell"

    def test_shell_add_split(self):

        """Test adding split"""

        shell = Shell("testShell")
        shell.add_split("sp1", size=3, stride=1)
        assert "sp1" in shell.splits
        assert shell.splits["sp1"] == (3, 1)

    def test_shell_add_index(self):

        """Test adding index"""

        shell = Shell("testShell")
        shell.add_index("idx1")
        assert "idx1" in shell.indices

    def test_shell_bind(self):

        """Test binding split and index"""

        shell = Shell("testShell")
        shell.add_split("sp1", size=3, stride=1)
        shell.bind("sp1", "idx1")
        assert shell.bindings["sp1"] == "idx1"

    def test_shell_to_cpp(self):

        """Test C++ code generation"""

        shell = Shell("testShell")
        shell.add_split("sp1", size=3, stride=1)
        shell.add_index("idx1")
        shell.bind("sp1", "idx1")
        shell.add_slice("tensor1", ["sp1"])

        cpp_code = shell.to_cpp_code()
        assert "shell dacpp::list testShell" in cpp_code
        assert "dacpp::split sp1(3, 1)" in cpp_code
        assert "binding(sp1, idx1)" in cpp_code


class TestCalc:

    """Test cases for Calc DSL"""


    def test_calc_creation(self):

        """Test Calc creation"""

        calc = Calc("testCalc")
        assert calc.name == "testCalc"
        assert calc.return_type == "void"

    def test_calc_add_param(self):

        """Test adding parameter"""

        calc = Calc("testCalc")
        calc.add_param("x", "float*")
        assert len(calc.params) == 1
        assert calc.params[0] == ("x", "float*")

    def test_calc_set_body(self):

        """Test setting function body"""

        calc = Calc("testCalc")
        calc.set_body("x[0] = 1.0;")
        assert calc.body == "x[0] = 1.0;"

    def test_calc_to_cpp(self):

        """Test C++ code generation"""

        calc = Calc("testCalc")
        calc.add_param("x", "float*")
        calc.set_body("x[0] = 1.0;")

        cpp_code = calc.to_cpp_code()
        assert "calc void testCalc" in cpp_code
        assert "float* x" in cpp_code
        assert "x[0] = 1.0;" in cpp_code

