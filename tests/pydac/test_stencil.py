"""
Stencil Test Case (DSL Implementation)

Original: stencil1.0/stencil.dac.cpp
Reimplemented using PyDAC DSL
"""

from typing import List
from .base_test import BaseDSLTest
from pydac import Shell, Calc, Expression


class StencilTest(BaseDSLTest):
    """Stencil computation test using PyDAC DSL"""
    
    def __init__(self, translator=None):
        super().__init__("stencil", translator)
        self.NX = 32
        self.NY = 32
        self.Lx = 10.0
        self.Ly = 10.0
        self.alpha = 0.01
        self.TIME_STEPS = 1000
        self.dx = self.Lx / (self.NX - 1)
        self.dy = self.Ly / (self.NY - 1)
        self.dt_stability = (self.dx * self.dx * self.dy * self.dy) / (2.0 * self.alpha * (self.dx * self.dx + self.dy * self.dy))
        self.delta_t = 0.4 * self.dt_stability
    
    def build(self) -> str:
        """Build complete C++ code with cmath header"""
        # Get DSL definitions
        self.shells = self.define_shells()
        self.calcs = self.define_calcs()
        self.expressions = self.define_expressions()
        self.main_code = self.generate_main_code()
        
        # Build includes - add cmath for std::exp
        code = """#include <iostream>
#include <vector>
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
        self.NX = 32
        self.NY = 32
        self.Lx = 10.0
        self.Ly = 10.0
        self.alpha = 0.01
        self.TIME_STEPS = 1000
        self.dx = self.Lx / (self.NX - 1)
        self.dy = self.Ly / (self.NY - 1)
        self.dt_stability = (self.dx * self.dx * self.dy * self.dy) / (2.0 * self.alpha * (self.dx * self.dx + self.dy * self.dy))
        self.delta_t = 0.4 * self.dt_stability
    
    def define_shells(self) -> List[Shell]:
        """Define Shell function for stencil"""
        shell = Shell("stencilShell")
        
        # Define parameters
        shell.add_param("matIn", "const dacpp::Matrix<double>&", is_const=True)
        shell.add_param("matOut", "dacpp::Matrix<double>&", is_const=False)
        
        # Add splits and indices
        shell.add_split("sp1", size=3, stride=1)
        shell.add_split("sp2", size=3, stride=1)
        shell.add_index("idx1")
        shell.add_index("idx2")
        shell.bind("sp1", "idx1")
        shell.bind("sp2", "idx2")
        
        # Add slices: matIn[sp1][sp2], matOut[idx1][idx2]
        shell.add_slice("matIn", ["sp1", "sp2"])
        shell.add_slice("matOut", ["idx1", "idx2"])
        
        return [shell]
    
    def define_calcs(self) -> List[Calc]:
        """Define Calc function for stencil"""
        calc = Calc("stencil")
        
        calc.add_param("mat", "dacpp::Matrix<double>&")
        calc.add_param("out", "double*")
        
        alpha_val = self.alpha
        delta_t_val = self.delta_t
        dx_val = self.dx
        dy_val = self.dy
        calc.set_body(f"""out[0] = mat[1][1] + {alpha_val} * {delta_t_val} * (((mat[2][1] - 2.0f * mat[1][1] + mat[0][1]) / ({dx_val} * {dx_val}))+ ((mat[1][2] - 2.0f * mat[1][1] + mat[1][0]) / ({dy_val} * {dy_val})));""")
        
        return [calc]
    
    def define_expressions(self) -> List[Expression]:
        """Define data association expression"""
        shell = self.define_shells()[0]
        calc = self.define_calcs()[0]
        
        expression = Expression(
            shell=shell,
            calc=calc,
            arguments=["u_curr_tensor", "middle_tensor"]
        )
        
        return [expression]
    
    def generate_main_code(self) -> str:
        """Generate main() function"""
        nx = self.NX
        ny = self.NY
        nx_minus_1 = nx - 1
        ny_minus_1 = ny - 1
        nx_minus_2 = nx - 2
        ny_minus_2 = ny - 2
        dx_val = self.dx
        dy_val = self.dy
        lx_val = self.Lx
        ly_val = self.Ly
        
        # Use string formatting to avoid f-string nesting issues
        slice1 = f"{{1,{nx_minus_1}}}"
        slice2 = f"{{1,{ny_minus_1}}}"
        
        code = f"""int main() {{
    // Initialize temperature field
    vector<double> u_curr({nx} * {ny}, 0.0f);
    vector<double> u_next({nx} * {ny}, 0.0f);
    
    // Initial condition: Gaussian distribution heat source at center
    int cx = {nx} / 2;
    int cy = {ny} / 2;
    double sigma = 1.0f;
    for(int i = 0; i < {nx}; ++i) {{
        for(int j = 0; j < {ny}; ++j) {{
            double x = i * {dx_val};
            double y = j * {dy_val};
            u_curr[i * {ny} + j] = std::exp(-((x - {lx_val}/2.0f)*(x - {lx_val}/2.0f) + (y - {ly_val}/2.0f)*(y - {ly_val}/2.0f)) / (2.0f * sigma * sigma));
        }}
    }}
    
    dacpp::Matrix<double> u_curr_tensor({{ {nx}, {ny} }}, u_curr);
    dacpp::Matrix<double> u_next_tensor({{ {nx}, {ny} }}, u_next);
    
    for(int i=0; i<{self.TIME_STEPS}; i++) {{
        dacpp::Matrix<double> middle_tensor = u_next_tensor[{slice1}][{slice2}];
        stencilShell(u_curr_tensor, middle_tensor) <-> stencil;
        
        for (int i = 1; i <= {nx_minus_2}; i++) {{
            for(int j = 1; j <= {ny_minus_2}; j++){{
                u_curr_tensor[i][j] = middle_tensor[i-1][j-1];
            }}
        }}
        
        // Boundary conditions (adiabatic: zero derivative)
        for (int j = 0; j < {ny}; ++j) {{
            u_curr_tensor[0][j] = u_curr_tensor[1][j];
            u_curr_tensor[{nx_minus_1}][j] = u_curr_tensor[{nx_minus_2}][j];
        }}
        for (int i = 0; i < {nx}; ++i) {{
            u_curr_tensor[i][0] = u_curr_tensor[i][1];
            u_curr_tensor[i][{ny_minus_1}] = u_curr_tensor[i][{ny_minus_2}];
        }}
    }}
    u_curr_tensor[0].print();
    
    return 0;
}}
"""
        return code
