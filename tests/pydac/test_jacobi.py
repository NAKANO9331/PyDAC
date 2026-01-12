"""
Jacobi Iteration Test Case (DSL Implementation)

Original: jacobi1.0/jacobi.dac.cpp
Reimplemented using PyDAC DSL
"""

from typing import List
from .base_test import BaseDSLTest
from pydac import Shell, Calc, Expression


class JacobiTest(BaseDSLTest):
    """Jacobi iteration test using PyDAC DSL"""
    
    def __init__(self, translator=None):
        super().__init__("jacobi", translator)
        self.N = 100
        self.max_iter = 10000
        self.tolerance = 1e-6
    
    def build(self) -> str:
        """Build complete C++ code with cmath header"""
        # Get DSL definitions
        self.shells = self.define_shells()
        self.calcs = self.define_calcs()
        self.expressions = self.define_expressions()
        self.main_code = self.generate_main_code()
        
        # Build includes - add cmath for std::fabs
        code = """#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

"""
        
        # Add Shell functions
        for shell in self.shells:
            code += shell.to_cpp_code() + "\n"
        
        # Add Calc functions
        for calc in self.calcs:
            code += calc.to_cpp_code() + "\n"
        
        # Add main function
        code += self.main_code + "\n"
        
        return code
        self.N = 100
        self.max_iter = 10000
        self.tolerance = 1e-6
    
    def define_shells(self) -> List[Shell]:
        """Define Shell function for Jacobi iteration"""
        shell = Shell("jacobiShell")
        
        # Define parameters
        shell.add_param("A", "const dacpp::Matrix<float>&", is_const=True)
        shell.add_param("b", "const dacpp::Vector<float>&", is_const=True)
        shell.add_param("x", "const dacpp::Vector<float>&", is_const=True)
        shell.add_param("x_new", "dacpp::Vector<float>&", is_const=False)
        shell.add_param("nums", "const dacpp::Vector<int>&", is_const=True)
        
        # Add index
        shell.add_index("idx1")
        
        # Add slices: A[{idx1}][{}], b[{idx1}], x[{}], x_new[{idx1}], nums[{idx1}]
        shell.add_slice("A", ["{idx1}", "{}"])
        shell.add_slice("b", ["{idx1}"])
        shell.add_slice("x", ["{}"])
        shell.add_slice("x_new", ["{idx1}"])
        shell.add_slice("nums", ["{idx1}"])
        
        return [shell]
    
    def define_calcs(self) -> List[Calc]:
        """Define Calc function for Jacobi iteration"""
        calc = Calc("jacobi")
        
        calc.add_param("a", "float*")
        calc.add_param("b", "float*")
        calc.add_param("x", "float*")
        calc.add_param("x_new", "float*")
        calc.add_param("num", "int*")
        
        calc.set_body(f"""float sigma = 0;
    for(int i = 0; i < {self.N}; ++i) {{
        if(i != num[0]) {{
            sigma += a[i] * x[i];
        }}
    }}
    x_new[0] = (b[0] - sigma) / a[num[0]];""")
        
        return [calc]
    
    def define_expressions(self) -> List[Expression]:
        """Define data association expression"""
        shell = self.define_shells()[0]
        calc = self.define_calcs()[0]
        
        expression = Expression(
            shell=shell,
            calc=calc,
            arguments=["A", "b", "x", "x_new", "tensor_nums"]
        )
        
        return [expression]
    
    def generate_main_code(self) -> str:
        """Generate main() function"""
        code = f"""int main() {{
    // Initialize coefficient matrix A and vector b
    std::vector<float> mat_A({self.N} * {self.N}, 0.0f);
    std::vector<float> vec_b({self.N}, 0.0f);
    std::vector<float> vec_x({self.N}, 0.0f);
    std::vector<float> vec_x_new({self.N}, 0.0f);
    
    // Auto-initialize A and b
    for (int i = 0; i < {self.N}; ++i) {{
        mat_A[i * {self.N} + i] = 4.0f;
        if (i > 0) {{
            mat_A[i * {self.N} + i - 1] = -1.0f;
        }}
        if (i < {self.N} - 1) {{
            mat_A[i * {self.N} + i + 1] = -1.0f;
        }}
        vec_b[i] = 1.0f;
    }}
    
    dacpp::Matrix<float> A({{ {self.N}, {self.N} }}, mat_A);
    dacpp::Vector<float> b(vec_b);
    dacpp::Vector<float> x(vec_x);
    dacpp::Vector<float> x_new(vec_x_new);
    
    bool converged = false;
    int iter = 0;
    std::vector<int> nums({self.N});
    for(int i = 0; i < {self.N}; i++){{
        nums[i] = i;
    }}
    dacpp::Vector<int> tensor_nums(nums);
    float* data = new float[1 * {self.N}];
    float* data2 = new float[1 * {self.N}];
    
    while (!converged && iter < {self.max_iter}) {{
        jacobiShell(A, b, x, x_new, tensor_nums) <-> jacobi;
        
        x.tensor2Array(data);
        x_new.tensor2Array(data2);
        
        float max_error = 0.0f;
        for (int i = 0; i < {self.N}; ++i) {{
            max_error = std::max(max_error, std::fabs(data2[i] - data[i]));
        }}
        
        if (max_error < {self.tolerance}) {{
            converged = true;
        }}
        
        x = x_new;
        iter++;
    }}
    
    // Output results (match original format: space-separated values)
    x.tensor2Array(data2);
    for (int i = 0; i < {self.N}; ++i) {{
        std::cout << data2[i] << " ";
    }}
    std::cout << std::endl;
    
    return 0;
}}
"""
        return code
