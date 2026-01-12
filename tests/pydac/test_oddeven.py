"""
Odd-Even Sort Test Case (DSL Implementation)

Original: oddeven0.1/oddEven.dac.cpp
Reimplemented using PyDAC DSL
"""

from typing import List
from .base_test import BaseDSLTest
from pydac import Shell, Calc, Expression


class OddEvenTest(BaseDSLTest):
    """Odd-even sort test using PyDAC DSL"""
    
    def __init__(self, translator=None):
        super().__init__("oddeven", translator)
        self.N = 8
    
    def define_shells(self) -> List[Shell]:
        """Define Shell function for odd-even sort"""
        shell = Shell("ODDEVEN")
        
        # Define parameters
        shell.add_param("array", "const dacpp::Vector<int>&", is_const=True)
        shell.add_param("array_out", "dacpp::Vector<int>&", is_const=False)
        
        # Add split
        shell.add_split("S1", size=2, stride=2)
        
        # Add slices: array[{S1}], array_out[{S1}]
        shell.add_slice("array", ["{S1}"])
        shell.add_slice("array_out", ["{S1}"])
        
        return [shell]
    
    def define_calcs(self) -> List[Calc]:
        """Define Calc function for odd-even sort"""
        calc = Calc("oddeven")
        
        calc.add_param("array", "int*")
        calc.add_param("array_out", "int*")
        
        calc.set_body("""if (array[0] > array[1]) {{
        array_out[0] = array[1];
        array_out[1] = array[0];
    }}else{{
        array_out[0] = array[0];
        array_out[1] = array[1];
    }}""")
        
        return [calc]
    
    def define_expressions(self) -> List[Expression]:
        """Define data association expression"""
        shell = self.define_shells()[0]
        calc = self.define_calcs()[0]
        
        expression = Expression(
            shell=shell,
            calc=calc,
            arguments=["array_tensor", "array_out_tensor"]
        )
        
        return [expression]
    
    def generate_main_code(self) -> str:
        """Generate main() function"""
        n_val = self.N
        n_minus_1 = n_val - 1
        slice_expr = f"{{1,{n_minus_1}}}"
        
        code = f"""int main() {{
    vector<int> array({n_val});
    
    // Initialize data (decreasing array)
    for (int i = 0; i < {n_val}; i++) {{
        array[i] = {n_val} - i;
    }}
    
    // Odd-even merge sort
    dacpp::Tensor<int, 1> array_tensor(array);
    vector<int> array_out({n_val});
    dacpp::Tensor<int, 1> array_out_tensor(array_out);
    
    for (int phase = 0; phase < {n_val}; phase++) {{
        ODDEVEN(array_tensor, array_out_tensor) <-> oddeven;
        
        dacpp::Tensor<int, 1> array2_tensor = array_out_tensor[{slice_expr}];
        vector<int> array_out2({n_minus_1}-1, 0);
        dacpp::Tensor<int, 1> array_out2_tensor(array_out2);
        
        ODDEVEN(array2_tensor, array_out2_tensor) <-> oddeven;
        
        for(int i = 1; i < {n_minus_1}; i++){{
            array_tensor[i] = array_out2_tensor[i-1];
        }}
        array_tensor[0] = array_out_tensor[0];
        array_tensor[{n_minus_1}] = array_out_tensor[{n_minus_1}];
    }}
    array_tensor.print();
    
    return 0;
}}
"""
        return code
