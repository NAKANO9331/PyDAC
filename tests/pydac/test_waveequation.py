"""
Wave Equation Test Case (DSL Implementation)

Original: waveEquation1.0/waveEquation.dac.cpp
Reimplemented using PyDAC DSL
"""

from typing import List
from .base_test import BaseDSLTest
from pydac import Shell, Calc, Expression


class WaveEquationTest(BaseDSLTest):
    """Wave equation test using PyDAC DSL"""
    
    def __init__(self, translator=None):
        super().__init__("waveEquation", translator)
        self.NX = 8
        self.NY = 8
        self.Lx = 10.0
        self.Ly = 10.0
        self.c = 1.0
        self.TIME_STEPS = 1000
        self.dx = self.Lx / (self.NX - 1)
        self.dy = self.Ly / (self.NY - 1)
        self.dt = 0.5 * min(self.dx, self.dy) / self.c
    
    def build(self) -> str:
        """Build complete C++ code with cmath header"""
        # Get DSL definitions
        self.shells = self.define_shells()
        self.calcs = self.define_calcs()
        self.expressions = self.define_expressions()
        self.main_code = self.generate_main_code()
        
        # Build includes - add cmath for std::exp and std::fmin
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
        self.NX = 8
        self.NY = 8
        self.Lx = 10.0
        self.Ly = 10.0
        self.c = 1.0
        self.TIME_STEPS = 1000
        self.dx = self.Lx / (self.NX - 1)
        self.dy = self.Ly / (self.NY - 1)
        self.dt = 0.5 * min(self.dx, self.dy) / self.c
    
    def define_shells(self) -> List[Shell]:
        """Define Shell function for wave equation"""
        shell = Shell("waveEqShell")
        
        shell.add_param("matCur", "const dacpp::Matrix<double>&", is_const=True)
        shell.add_param("matPrev", "const dacpp::Matrix<double>&", is_const=True)
        shell.add_param("matNext", "dacpp::Matrix<double>&", is_const=False)
        
        shell.add_split("sp1", size=3, stride=1)
        shell.add_split("sp2", size=3, stride=1)
        shell.add_index("idx1")
        shell.add_index("idx2")
        shell.bind("sp1", "idx1")
        shell.bind("sp2", "idx2")
        
        shell.add_slice("matCur", ["sp1", "sp2"])
        shell.add_slice("matPrev", ["idx1", "idx2"])
        shell.add_slice("matNext", ["idx1", "idx2"])
        
        return [shell]
    
    def define_calcs(self) -> List[Calc]:
        """Define Calc function for wave equation"""
        calc = Calc("waveEq")
        
        calc.add_param("cur", "dacpp::Matrix<double>&")
        calc.add_param("prev", "double*")
        calc.add_param("next", "double*")
        
        dx_val = self.dx
        dy_val = self.dy
        c_val = self.c
        calc.set_body(f"""double dt = 0.5f * std::fmin({dx_val}, {dy_val}) / {c_val};
    double u_xx = (cur[2][1] - 2.0f * cur[1][1] + cur[0][1])/ ({dx_val} * {dx_val});
    double u_yy = (cur[1][2] - 2.0f * cur[1][1] + cur[1][0])/ ({dy_val} * {dy_val});
    next[0]=2.0f*cur[1][1]-prev[0]+({c_val} * {c_val})*dt*dt*(u_xx+u_yy);""")
        
        return [calc]
    
    def define_expressions(self) -> List[Expression]:
        """Define data association expression"""
        shell = self.define_shells()[0]
        calc = self.define_calcs()[0]
        
        expression = Expression(
            shell=shell,
            calc=calc,
            arguments=["u_curr_tensor", "u_prev_middle_tensor", "u_next_middle_tensor"]
        )
        
        return [expression]
    
    def generate_main_code(self) -> str:
        """Generate main() function"""
        code = f"""int main() {{
    vector<double> u_prev({self.NX} * {self.NY}, 0.0f);
    vector<double> u_curr({self.NX} * {self.NY}, 0.0f);
    vector<double> u_next({self.NX} * {self.NY}, 0.0f);
    
    double sigma = 0.5f;
    for(int i = 0; i < {self.NX}; ++i) {{
        for(int j = 0; j < {self.NY}; ++j) {{
            double x = i * {self.dx};
            double y = j * {self.dy};
            u_prev[i*{self.NX}+j] = std::exp(-((x - {self.Lx}/2)*(x - {self.Lx}/2) + (y - {self.Ly}/2)*(y - {self.Ly}/2)) / (2 * sigma * sigma));
        }}
    }}
    
    dacpp::Matrix<double> u_curr_tensor({{ {self.NX}, {self.NY} }}, u_curr);
    dacpp::Matrix<double> u_prev_tensor({{ {self.NX}, {self.NY} }}, u_prev);
    dacpp::Matrix<double> u_next_tensor({{ {self.NX}, {self.NY} }}, u_next);
    dacpp::Matrix<double> u_prev_middle_tensor = u_prev_tensor[{{1,{self.NX}-1}}][{{1,{self.NY}-1}}];
    
    for(int i = 0; i < {self.TIME_STEPS}; i++) {{
        dacpp::Matrix<double> u_next_middle_tensor = u_next_tensor[{{1,{self.NX}-1}}][{{1,{self.NY}-1}}];
        waveEqShell(u_curr_tensor, u_prev_middle_tensor, u_next_middle_tensor) <-> waveEq;
        
        for (int i = 1; i <= {self.NX}-2; i++) {{
            for(int j = 1; j <= {self.NY}-2; j++){{
                u_prev_middle_tensor[i-1][j-1] = u_curr_tensor[i][j];
            }}
        }}
        
        for (int i = 1; i <= {self.NX}-2; i++) {{
            for(int j = 1; j <= {self.NY}-2; j++){{
                u_curr_tensor[i][j] = u_next_middle_tensor[i-1][j-1];
            }}
        }}
        
        // Boundary conditions (set to zero)
        for (int i = 0; i < {self.NX}; ++i) {{
            u_curr_tensor[i][{self.NY}-1] = 0;
            u_curr_tensor[i][0] = 0;
        }}
        for (int j = 0; j < {self.NY}; ++j) {{
            u_curr_tensor[{self.NX} - 1][j] = 0;
            u_curr_tensor[0][j] = 0;
        }}
    }}
    
    u_curr_tensor.print();
    
    return 0;
}}
"""
        return code
