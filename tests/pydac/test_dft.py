"""
Discrete Fourier Transform Test Case (DSL Implementation)

Original: DFT1.0/DFT.dac.cpp
Reimplemented using PyDAC DSL
"""

from typing import List
from .base_test import BaseDSLTest
from pydac import Shell, Calc, Expression


class DFTTest(BaseDSLTest):
    """DFT test using PyDAC DSL"""
    
    def __init__(self, translator=None):
        super().__init__("DFT", translator)
        self.N = 8
    
    def build(self) -> str:
        """Build complete C++ code with complex header"""
        # Get DSL definitions
        self.shells = self.define_shells()
        self.calcs = self.define_calcs()
        self.expressions = self.define_expressions()
        self.main_code = self.generate_main_code()
        
        # Build includes - add complex and cmath
        code = """#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;
using Complex = std::complex<double>;

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
        self.N = 8
    
    def define_shells(self) -> List[Shell]:
        """Define Shell function for DFT"""
        shell = Shell("DFT")
        
        shell.add_param("input", "const dacpp::Vector<std::complex<double>>&", is_const=True)
        shell.add_param("output", "dacpp::Vector<std::complex<double>>&", is_const=False)
        shell.add_param("vec", "const dacpp::Vector<int>&", is_const=True)
        
        shell.add_index("i")
        
        shell.add_slice("input", ["{}"])
        shell.add_slice("output", ["i"])
        shell.add_slice("vec", ["i"])
        
        return [shell]
    
    def define_calcs(self) -> List[Calc]:
        """Define Calc function for DFT"""
        calc = Calc("dft")
        
        calc.add_param("input", "std::complex<double>*")
        calc.add_param("output", "std::complex<double>*")
        calc.add_param("vec", "int*")
        
        calc.set_body(f"""Complex sum(0, 0);
    for (int n = 0; n < {self.N}; ++n) {{
        double angle = -2.0 * M_PI * vec[0] * n / {self.N};
        Complex W_n(std::cos(angle), std::sin(angle));
        sum += input[n] * W_n;
    }}
    output[0] = sum;""")
        
        return [calc]
    
    def define_expressions(self) -> List[Expression]:
        """Define data association expression"""
        shell = self.define_shells()[0]
        calc = self.define_calcs()[0]
        
        expression = Expression(
            shell=shell,
            calc=calc,
            arguments=["input_tensor", "output_tensor", "vec_tensor"]
        )
        
        return [expression]
    
    def generate_main_code(self) -> str:
        """Generate main() function"""
        code = f"""int main() {{
    using Complex = std::complex<double>;
    
    vector<std::complex<double>> input({self.N});
    for (int i = 0; i < {self.N}; ++i) {{
        input[i] = Complex(i, 0);
    }}
    
    vector<Complex> output({self.N});
    std::vector<int> vec({self.N});
    for (int i = 0; i < {self.N}; ++i) {{
        vec[i] = i;
    }}
    
    dacpp::Vector<int> vec_tensor(vec);
    dacpp::Vector<std::complex<double>> input_tensor(input);
    dacpp::Vector<std::complex<double>> output_tensor(output);
    
    DFT(input_tensor, output_tensor, vec_tensor) <-> dft;
    output_tensor.print();
    
    return 0;
}}
"""
        return code
