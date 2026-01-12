"""
Fourier-Laplace Test Case (DSL Implementation)

Original: FOuLa1.0/FOuLa.dac.cpp
Reimplemented using PyDAC DSL
"""

from typing import List
from .base_test import BaseDSLTest
from pydac import Shell, Calc, Expression


class FOuLaTest(BaseDSLTest):
    """Fourier-Laplace PDE test using PyDAC DSL"""
    
    def __init__(self, translator=None):
        super().__init__("FOuLa", translator)
        self.n = 100
        self.m = 5
        self.r = 0.25
    
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
        self.n = 100
        self.m = 5
        self.r = 0.25
    
    def define_shells(self) -> List[Shell]:
        """Define Shell function for PDE"""
        shell = Shell("PDE")
        
        shell.add_param("u_kin", "const dacpp::Vector<double>&", is_const=True)
        shell.add_param("u_kout", "dacpp::Vector<double>&", is_const=False)
        shell.add_param("r", "const dacpp::Vector<double>&", is_const=True)
        
        shell.add_split("s", size=3, stride=1)
        shell.add_index("i")
        shell.bind("s", "i")
        
        shell.add_slice("u_kin", ["s"])
        shell.add_slice("u_kout", ["i"])
        shell.add_slice("r", ["{}"])
        
        return [shell]
    
    def define_calcs(self) -> List[Calc]:
        """Define Calc function for PDE"""
        calc = Calc("pde")
        
        calc.add_param("u_kin", "dacpp::Vector<double>&")
        calc.add_param("u_kout", "double*")
        calc.add_param("r", "double*")
        
        calc.set_body("""u_kout[0] = r[0] * u_kin[0] + (1 - 2 * r[0]) * u_kin[1] + r[0] * u_kin[2];""")
        
        return [calc]
    
    def define_expressions(self) -> List[Expression]:
        """Define data association expression"""
        shell = self.define_shells()[0]
        calc = self.define_calcs()[0]
        
        expression = Expression(
            shell=shell,
            calc=calc,
            arguments=["u_kin_tensor", "u_kout_tensor", "r_tensor"]
        )
        
        return [expression]
    
    def generate_main_code(self) -> str:
        """Generate main() function"""
        m_val = self.m
        n_val = self.n
        r_val = self.r
        m_plus_1 = m_val + 1
        n_plus_1 = n_val + 1
        m_minus_1 = m_val - 1
        slice1 = f"{{1,{m_val}}}"
        slice2 = "{}"
        
        code = f"""int main() {{
    double h = 1.0 / {m_val};
    double tau = 1.0 / {n_val};
    double r = {r_val};
    double a = 1.0;
    
    double *x = (double*)malloc(sizeof(double)*({m_plus_1}));
    for (int i=0; i<={m_val}; i++) {{
        x[i] = i*h;
    }}
    
    double *t = (double*)malloc(sizeof(double)*({n_plus_1}));
    for (int i = 0; i <= {n_val}; i++) {{
        t[i] = i*tau;
    }}
    
    double **u = (double**)malloc(sizeof(double*)*({m_plus_1}));
    for (int i=0; i<={m_val}; i++) {{
        u[i] = (double*)malloc(sizeof(double)*({n_plus_1}));
    }}
    
    for (int i = 0; i <= {m_val}; i++)
        u[i][0] = x[i]*x[i]*x[i] + x[i];
    for (int i = 1; i <= {n_val}; i++) {{
        u[0][i] = 0.0;
        u[{m_val}][i] = 1.0 + std::exp(t[i]);
    }}
    
    std::vector<double> u_flat;
    for (int i = 0; i <= {m_val}; ++i) {{
        for (int j = 0; j <= {n_val}; ++j) {{
            u_flat.push_back(static_cast<double>(u[i][j]));
        }}
    }}
    
    dacpp::Matrix<double> u_tensor({{ {m_plus_1}, {n_plus_1} }}, u_flat);
    
    for (int k = 0; k < {n_val}; k++) {{
        dacpp::Vector<double> middle_tensor = u_tensor[{slice1}][k+1];
        std::vector<double> r_data;
        r_data.push_back(r);
        dacpp::Vector<double> R(r_data);
        dacpp::Vector<double> u_test1 = u_tensor[{slice2}][k];
        PDE(u_test1, middle_tensor, R) <-> pde;
        
        for (int i = 1; i <= {m_minus_1}; i++) {{
            u_tensor[i][k+1] = middle_tensor[i-1];
        }}
    }}
    
    u_tensor[1].print();
    
    return 0;
}}
"""
        return code
