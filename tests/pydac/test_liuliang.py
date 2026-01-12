"""
Traffic Flow (LWR) Test Case (DSL Implementation)

Original: liuliang1.0/liuliang.dac.cpp
Reimplemented using PyDAC DSL
"""

from typing import List
from .base_test import BaseDSLTest
from pydac import Shell, Calc, Expression


class LiuliangTest(BaseDSLTest):
    """Traffic flow (LWR) test using PyDAC DSL"""
    
    def __init__(self, translator=None):
        super().__init__("liuliang", translator)
        self.WIDTH = 100
        self.TIME_STEPS = 200
        self.DELTA_T = 0.01
        self.DELTA_X = 1.0
    
    def define_shells(self) -> List[Shell]:
        """Define Shell function for LWR"""
        shell = Shell("LWR_shell")
        
        shell.add_param("rho", "const dacpp::Vector<double>&", is_const=True)
        shell.add_param("new_rho", "dacpp::Vector<double>&", is_const=False)
        
        shell.add_split("S1", size=2, stride=1)
        shell.add_index("idx1")
        shell.bind("S1", "idx1")
        
        shell.add_slice("rho", ["S1"])
        shell.add_slice("new_rho", ["idx1"])
        
        return [shell]
    
    def define_calcs(self) -> List[Calc]:
        """Define Calc function for LWR"""
        calc = Calc("lwr")
        
        calc.add_param("rho", "dacpp::Vector<double>&")
        calc.add_param("new_rho", "double*")
        
        calc.set_body(f"""double V_max = 30;
    double rho_max = 50;
    double q0 = rho[0] * V_max * (1 - rho[0] / rho_max);
    double q1 = rho[1] * V_max * (1 - rho[1] / rho_max);
    new_rho[0] = rho[1] - ({self.DELTA_T} / {self.DELTA_X}) * (q1 - q0);
    new_rho[0] = std::max(0.0, new_rho[0]);""")
        
        return [calc]
    
    def define_expressions(self) -> List[Expression]:
        """Define data association expression"""
        shell = self.define_shells()[0]
        calc = self.define_calcs()[0]
        
        expression = Expression(
            shell=shell,
            calc=calc,
            arguments=["middle_in_tensor", "middle_out_tensor"]
        )
        
        return [expression]
    
    def generate_main_code(self) -> str:
        """Generate main() function"""
        code = f"""int main() {{
    std::vector<double> rho({self.WIDTH}, 0.0);
    std::vector<double> new_rho({self.WIDTH}, 0.0);
    
    // Initialize density
    for (int i = 0; i < {self.WIDTH}; ++i) {{
        if (i < {self.WIDTH} / 4) {{
            rho[i] = 40;
        }} else if (i < 3 * {self.WIDTH} / 4) {{
            rho[i] = 20;
        }} else {{
            rho[i] = 10;
        }}
    }}
    
    dacpp::Vector<double> rho_tensor(rho);
    dacpp::Vector<double> new_rho_tensor(new_rho);
    dacpp::Vector<double> middle_out_tensor = new_rho_tensor[{{1,{self.WIDTH}-1}}];
    dacpp::Vector<double> middle_in_tensor = rho_tensor[{{0,{self.WIDTH}-1}}];
    
    for (int t = 0; t < {self.TIME_STEPS}; ++t) {{
        LWR_shell(middle_in_tensor, middle_out_tensor) <-> lwr;
        for (int i = 1; i <= {self.WIDTH}-2; i++) {{
            middle_in_tensor[i] = middle_out_tensor[i-1];
        }}
        middle_in_tensor[0] = middle_out_tensor[0];
    }}
    
    std::cout << middle_in_tensor[15] << std::endl;
    
    return 0;
}}
"""
        return code
