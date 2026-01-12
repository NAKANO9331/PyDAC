"""
Decay Chain Test Case (DSL Implementation)

Original: decay1.0/decay_chain.dac.cpp
Reimplemented using PyDAC DSL
"""

from typing import List
from .base_test import BaseDSLTest
from pydac import Shell, Calc, Expression


class DecayTest(BaseDSLTest):
    
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
#include <cstdlib>
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
    """Decay chain test using PyDAC DSL"""
    
    def __init__(self, translator=None):
        super().__init__("decay", translator)
        self.dt = 0.1
        self.T = 5.0
        self.numIsotopes = 10
    
    def define_shells(self) -> List[Shell]:
        """Define Shell function for decay"""
        shell = Shell("DECAY")
        
        # Define parameters
        shell.add_param("N0s", "const dacpp::Vector<double>&", is_const=True)
        shell.add_param("lambdas", "const dacpp::Vector<double>&", is_const=True)
        shell.add_param("local_A", "dacpp::Vector<double>&", is_const=False)
        shell.add_param("t", "const dacpp::Vector<double>&", is_const=True)
        
        # Add index
        shell.add_index("i")
        
        # Add slices: N0s[i], lambdas[i], local_A[i], t[{}]
        shell.add_slice("N0s", ["i"])
        shell.add_slice("lambdas", ["i"])
        shell.add_slice("local_A", ["i"])
        shell.add_slice("t", ["{}"])
        
        return [shell]
    
    def define_calcs(self) -> List[Calc]:
        """Define Calc function for decay"""
        calc = Calc("decay_calc")
        
        calc.add_param("N0s", "double*")
        calc.add_param("lambdas", "double*")
        calc.add_param("local_A", "double*")
        calc.add_param("t", "double*")
        
        calc.set_body("""local_A[0] = N0s[0] * std::exp(-lambdas[0] * t[0]);""")
        
        return [calc]
    
    def define_expressions(self) -> List[Expression]:
        """Define data association expression"""
        shell = self.define_shells()[0]
        calc = self.define_calcs()[0]
        
        expression = Expression(
            shell=shell,
            calc=calc,
            arguments=["N0s_tensor", "lambdas_tensor", "local_A_tensor", "t_tensor"]
        )
        
        return [expression]
    
    def generate_main_code(self) -> str:
        """Generate main() function"""
        time_steps = int(self.T/self.dt)
        code = f"""int main() {{
    // Random decay constants and initial quantities
    std::vector<double> lambdas({self.numIsotopes});
    std::vector<double> N0s({self.numIsotopes}, 1000.0);  // Initial quantity is 1000
    
    // Random initialization of decay constants (e.g., lambda between 0.01 and 0.2)
    for (size_t i = 0; i < {self.numIsotopes}; ++i) {{
        lambdas[i] = 0.01 + 0.01*i;  // lambda range [0.01, 0.2]
    }}
    
    // Calculate decay
    size_t numIsotopes = lambdas.size();
    std::vector<double> A({time_steps}*numIsotopes, 0.0);
    std::vector<double> t;
    t.push_back(static_cast<double>(0));
    
    std::vector<double> local_A(numIsotopes, 0.0);
    dacpp::Vector<double> local_A_tensor(local_A);
    dacpp::Vector<double> N0s_tensor(N0s);
    dacpp::Vector<double> lambdas_tensor(lambdas);
    dacpp::Vector<double> t_tensor(t);
    dacpp::Matrix<double> A_tensor({{ {time_steps}, static_cast<int>(numIsotopes) }}, A);
    
    while(t_tensor[0] <= {self.T}){{
        DECAY(N0s_tensor, lambdas_tensor, local_A_tensor, t_tensor) <-> decay_calc;
        A_tensor[10*t_tensor[0]] = local_A_tensor;
        t_tensor[0] += {self.dt};
    }}
    A_tensor[1].print();
    
    return 0;
}}
"""
        return code
