"""
Base test case class for PyDAC DSL tests
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import tempfile
import numpy as np

from pydac import PyDAC, Shell, Calc, Tensor, Expression


class BaseDSLTest(ABC):
    """Base class for all DSL test cases"""
    
    def __init__(self, name: str, translator: Optional[PyDAC] = None):
        """
        Initialize test case
        
        Args:
            name: Test case name
            translator: PyDAC translator instance (optional)
        """
        self.name = name
        self.translator = translator or PyDAC(verbose=False)
        self.shells: List[Shell] = []
        self.calcs: List[Calc] = []
        self.expressions: List[Expression] = []
        self.main_code: str = ""
        
    @abstractmethod
    def define_shells(self) -> List[Shell]:
        """Define Shell functions for this test case"""
        pass
    
    @abstractmethod
    def define_calcs(self) -> List[Calc]:
        """Define Calc functions for this test case"""
        pass
    
    @abstractmethod
    def define_expressions(self) -> List[Expression]:
        """Define data association expressions"""
        pass
    
    @abstractmethod
    def generate_main_code(self) -> str:
        """Generate main() function code"""
        pass
    
    def build(self) -> str:
        """
        Build complete C++ code from DSL definitions
        
        Returns:
            Complete C++ code string
        """
        # Get DSL definitions
        self.shells = self.define_shells()
        self.calcs = self.define_calcs()
        self.expressions = self.define_expressions()
        self.main_code = self.generate_main_code()
        
        # Build includes - base includes, test can add more if needed
        code = """#include <iostream>
#include <vector>
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
    
    def save_to_file(self, output_path: Optional[Path] = None) -> Path:
        """
        Save generated code to file
        
        Args:
            output_path: Output file path (optional, will use temp file if not provided)
            
        Returns:
            Path to saved file
        """
        code = self.build()
        
        if output_path is None:
            # Use temp file
            fd, temp_path = tempfile.mkstemp(suffix='.dac.cpp', prefix=f'{self.name}_')
            output_path = Path(temp_path)
            with open(fd, 'w') as f:
                f.write(code)
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(code)
        
        return output_path
    
    def translate(self, mode: str = "usm", output_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Translate test case using PyDAC
        
        Args:
            mode: Translation mode (usm/buffer)
            output_file: Output file path (optional)
            
        Returns:
            Translation result dictionary
        """
        # Save code to temp file
        temp_file = self.save_to_file()
        
        try:
            # Translate
            result = self.translator.translate(str(temp_file), mode=mode)
            
            return {
                "success": result.success,
                "input_file": str(temp_file),
                "output_file": result.output_file,
                "mode": mode,
                "warnings": result.warnings,
                "errors": result.errors,
                "duration": result.duration
            }
        finally:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
    
    def translate_compile_and_run(
        self, 
        mode: str = "usm",
        timeout: Optional[float] = None,
        input_data: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete workflow: translate, compile, and run
        
        Args:
            mode: Translation mode
            timeout: Execution timeout
            input_data: Input data for program execution
            
        Returns:
            Complete result dictionary
        """
        # Translate
        translate_result = self.translate(mode=mode)
        
        if not translate_result["success"]:
            return {
                "translate": translate_result,
                "compile": None,
                "run": None,
                "overall_success": False
            }
        
        # Compile
        compile_result = self.translator.Compile(translate_result["output_file"])
        
        if not compile_result.success:
            return {
                "translate": translate_result,
                "compile": {
                    "success": False,
                    "errors": compile_result.stderr
                },
                "run": None,
                "overall_success": False
            }
        
        # Run
        run_result = self.translator.run(
            compile_result.binary_file,
            timeout=timeout,
            input_data=input_data
        )
        
        return {
            "translate": translate_result,
            "compile": {
                "success": compile_result.success,
                "binary_file": compile_result.binary_file
            },
            "run": {
                "success": run_result.success,
                "return_code": run_result.return_code,
                "stdout": run_result.stdout,
                "stderr": run_result.stderr
            },
            "overall_success": run_result.success
        }
