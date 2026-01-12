"""
Mandelbrot Set Test Case (DSL Implementation)

Original: mandel1.0/mandel.dac.cpp
Reimplemented using PyDAC DSL
"""

from typing import List
from .base_test import BaseDSLTest
from pydac import Shell, Calc, Expression


class MandelTest(BaseDSLTest):
    """Mandelbrot set test using PyDAC DSL"""
    
    def __init__(self, translator=None):
        super().__init__("mandel", translator)
        self.row_count = 8
        self.col_count = 8
        self.max_iterations = 1000
    
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
    
    def define_shells(self) -> List[Shell]:
        """Define Shell function for Mandelbrot"""
        shell = Shell("MANDEL")
        
        shell.add_param("complex_points", "const dacpp::Vector<complex<float>>&", is_const=True)
        shell.add_param("mandelbrot_flags", "dacpp::Vector<int>&", is_const=False)
        
        shell.add_index("i")
        
        shell.add_slice("complex_points", ["i"])
        shell.add_slice("mandelbrot_flags", ["i"])
        
        return [shell]
    
    def define_calcs(self) -> List[Calc]:
        """Define Calc function for Mandelbrot"""
        calc = Calc("mandel")
        
        calc.add_param("complex_points", "complex<float>*")
        calc.add_param("mandelbrot_flags", "int*")
        
        max_iter = self.max_iterations
        calc.set_body(f"""const complex<float>& c = complex_points[0];
    complex<float> z = 0;
    int iterations = 0;
    for (int i = 0; i < {max_iter}; ++i) {{
        if (std::sqrt(z.real()*z.real() + z.imag()*z.imag()) > 2.0f) {{
            iterations = i;
            break;
        }}
        z = z * z + c;
        iterations = {max_iter};
    }}
    if (iterations == {max_iter}) {{
        mandelbrot_flags[0] = 1;
    }}""")
        
        return [calc]
    
    def define_expressions(self) -> List[Expression]:
        """Define data association expression"""
        shell = self.define_shells()[0]
        calc = self.define_calcs()[0]
        
        expression = Expression(
            shell=shell,
            calc=calc,
            arguments=["complex_points_tensor", "mandelbrot_flags_tensor"]
        )
        
        return [expression]
    
    def generate_main_code(self) -> str:
        """Generate main() function"""
        code = f"""int main() {{
    int total_points = {self.row_count} * {self.col_count};
    vector<complex<float>> complex_points(total_points);
    
    for (int i = 0; i < {self.row_count}; ++i) {{
        for (int j = 0; j < {self.col_count}; ++j) {{
            int index = i * {self.col_count} + j;
            float real = -1.5f + (i * (2.0f / {self.row_count}));
            float imag = -1.0f + (j * (2.0f / {self.col_count}));
            complex_points[index] = complex<float>(real, imag);
        }}
    }}
    
    vector<int> mandelbrot_flags(total_points, 0);
    dacpp::Vector<complex<float>> complex_points_tensor(complex_points);
    dacpp::Vector<int> mandelbrot_flags_tensor(mandelbrot_flags);
    
    MANDEL(complex_points_tensor, mandelbrot_flags_tensor) <-> mandel;
    
    int mandelbrot_count = 0;
    for (int i = 0; i < total_points; i++){{
        if (mandelbrot_flags_tensor[i] == 1) mandelbrot_count++;
    }}
    
    // Print statistics (match original format)
    std::cout << "Mandelbrot Set Statistics:" << std::endl;
    std::cout << "Total points: " << total_points << std::endl;
    std::cout << "Points in the Mandelbrot set: " << mandelbrot_count << std::endl;
    
    return 0;
}}
"""
        return code
