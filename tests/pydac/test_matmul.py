"""
Matrix Multiplication Test Case (DSL Implementation)

Original: matMul1.0/matMul.dac.cpp
Reimplemented using PyDAC DSL
"""

from typing import List
import numpy as np
from .base_test import BaseDSLTest
from pydac import Shell, Calc, Expression


class MatMulTest(BaseDSLTest):
    """Matrix multiplication test using PyDAC DSL"""
    
    def __init__(self, translator=None):
        super().__init__("matMul", translator)
        # Initialize test data
        self.matA_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        self.matA_shape = [4, 5]
        self.matB_data = [1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19, 4, 8, 12, 16, 20]
        self.matB_shape = [5, 4]
        self.matC_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.matC_shape = [4, 4]
    
    def define_shells(self) -> List[Shell]:
        """Define Shell function for matrix multiplication"""
        shell = Shell("matrixMultiply_shell")
        
        # Define parameters explicitly (matching original C++ code)
        shell.add_param("matA", "dacpp::Matrix<int>&", is_const=True)
        shell.add_param("matB", "dacpp::Matrix<int>&", is_const=True)
        shell.add_param("matC", "dacpp::Matrix<int>&", is_const=False)  # matC is not const
        
        # Add indices
        shell.add_index("idx1")
        shell.add_index("idx2")
        
        # Add slices for matA[idx1][{}], matB[{}][idx2], matC[idx1][idx2]
        shell.add_slice("matA", ["idx1", "{}"])
        shell.add_slice("matB", ["{}", "idx2"])
        shell.add_slice("matC", ["idx1", "idx2"])
        
        return [shell]
    
    def define_calcs(self) -> List[Calc]:
        """Define Calc function for matrix multiplication"""
        calc = Calc("matrixMultiply_calc")
        
        # Parameters: vecA, vecB, dotProduct
        calc.add_param("vecA", "dacpp::Vector<int>&")
        calc.add_param("vecB", "dacpp::Vector<int>&")
        calc.add_param("dotProduct", "int*")
        
        # Function body: compute dot product
        calc.set_body("""for (int i = 0; i < 5; i++) {
        dotProduct[0] += vecA[i] * vecB[i];
    }""")
        
        return [calc]
    
    def define_expressions(self) -> List[Expression]:
        """Define data association expression"""
        shell = self.define_shells()[0]
        calc = self.define_calcs()[0]
        
        expression = Expression(
            shell=shell,
            calc=calc,
            arguments=["matA", "matB", "matC"]
        )
        
        return [expression]
    
    def generate_main_code(self) -> str:
        """Generate main() function"""
        # Generate data initialization
        code = """int main() {
    // Initialize matrix A
    std::vector<int> dataA{"""
        code += ", ".join(map(str, self.matA_data)) + "};\n"
        code += f"    dacpp::Matrix<int> matA({{{self.matA_shape[0]}, {self.matA_shape[1]}}}, dataA);\n\n"
        
        code += "    // Initialize matrix B\n"
        code += "    std::vector<int> dataB{"
        code += ", ".join(map(str, self.matB_data)) + "};\n"
        code += f"    dacpp::Matrix<int> matB({{{self.matB_shape[0]}, {self.matB_shape[1]}}}, dataB);\n\n"
        
        code += "    // Initialize matrix C\n"
        code += "    std::vector<int> dataC{"
        code += ", ".join(map(str, self.matC_data)) + "};\n"
        code += f"    dacpp::Matrix<int> matC({{{self.matC_shape[0]}, {self.matC_shape[1]}}}, dataC);\n\n"
        
        # Generate expression call
        code += "    matrixMultiply_shell(matA, matB, matC) <-> matrixMultiply_calc;\n"
        code += "    matC.print();\n\n"
        code += "    return 0;\n"
        code += "}\n"
        
        return code


if __name__ == "__main__":
    from pathlib import Path
    # Test the DSL implementation
    test = MatMulTest()
    code = test.build()
    print("Generated C++ code:")
    print("=" * 60)
    print(code)
    print("=" * 60)
    
    # Save to file
    output_file = test.save_to_file(Path("matMul.dac.cpp"))
    print(f"\nCode saved to: {output_file}")
