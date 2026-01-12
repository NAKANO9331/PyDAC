"""
Markov Decision Process Test Case (DSL Implementation)

Original: MDP1.0/mdp.dac.cpp
Reimplemented using PyDAC DSL
"""

from typing import List
from .base_test import BaseDSLTest
from pydac import Shell, Calc, Expression


class MDPTest(BaseDSLTest):
    """MDP test using PyDAC DSL"""
    
    def __init__(self, translator=None):
        super().__init__("MDP", translator)
        self.A = 1.0
        self.D = 0.1
        self.dx = 0.1
        self.dt = 0.01
        self.N = 100
        self.T = 1000
    
    def build(self) -> str:
        """Build complete C++ code with cmath header"""
        # Get DSL definitions
        self.shells = self.define_shells()
        self.calcs = self.define_calcs()
        self.expressions = self.define_expressions()
        self.main_code = self.generate_main_code()
        
        # Build includes - add cmath for std::exp and std::pow
        code = f"""#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"

namespace dacpp {{
    typedef std::vector<std::any> list;
}}

// Parameter settings
const double A = {self.A};  // Attraction coefficient
const double D = {self.D};  // Diffusion coefficient
const double dx = {self.dx}; // Spatial step size
const double dt = {self.dt}; // Time step size
const int N = {self.N};     // Spatial grid points
const int T = {self.T};     // Time steps

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
        """Define Shell function for MDP"""
        shell = Shell("mdp_shell")
        
        shell.add_param("p", "const dacpp::Vector<double>&", is_const=True)
        shell.add_param("new_p", "dacpp::Vector<double>&", is_const=False)
        
        # Match original order: idx first, then sp
        shell.add_index("idx")
        shell.add_split("sp", size=3, stride=1)
        
        shell.add_slice("p", ["sp"])
        shell.add_slice("new_p", ["idx"])
        
        return [shell]
    
    def define_calcs(self) -> List[Calc]:
        """Define Calc function for MDP"""
        calc = Calc("mdp")
        
        calc.add_param("p", "dacpp::Vector<double>&")
        calc.add_param("new_p", "double*")
        
        # Use constants directly in calc body (like original code)
        calc.set_body("""double diffusion = D * (p[2] - 2 * p[1] + p[0]) / (dx * dx);
    double drift = (-A) * (p[2] - p[0]) / (2 * dx);
    new_p[0] = p[1] + dt * (diffusion + drift);""")
        
        return [calc]
    
    def define_expressions(self) -> List[Expression]:
        """Define data association expression"""
        shell = self.define_shells()[0]
        calc = self.define_calcs()[0]
        
        expression = Expression(
            shell=shell,
            calc=calc,
            arguments=["p_tensor", "new_p_tensor"]
        )
        
        return [expression]
    
    def generate_main_code(self) -> str:
        """Generate main() function"""
        code = f"""int main() {{
    std::vector<double> p({self.N}, 0.0);
    
    // Initialize preference distribution
    for (int i = 0; i < {self.N}; ++i) {{
        double x = i * {self.dx};
        p[i] = std::exp(-std::pow(x - 5.0, 2) / 2.0);
    }}
    
    std::vector<double> new_p({self.N}-2, 0.0);
    dacpp::Vector<double> p_tensor(p);
    dacpp::Vector<double> new_p_tensor(new_p);
    
    for (int t = 0; t < {self.T}; ++t) {{
        mdp_shell(p_tensor, new_p_tensor) <-> mdp;
        for(int i = 0; i < {self.N}-2; i++){{
            p_tensor[i+1] = new_p_tensor[i];
        }}
    }}
    
    std::cout << p_tensor[2] << std::endl;
    
    return 0;
}}
"""
        return code
