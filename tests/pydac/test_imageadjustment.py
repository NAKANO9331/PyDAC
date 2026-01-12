"""
Image Adjustment Test Case (DSL Implementation)

Original: imageAdjustment1.0/imageAdjustment.dac.cpp
Reimplemented using PyDAC DSL
"""

from typing import List
from .base_test import BaseDSLTest
from pydac import Shell, Calc, Expression


class ImageAdjustmentTest(BaseDSLTest):
    """Image adjustment test using PyDAC DSL"""
    
    def __init__(self, translator=None):
        super().__init__("imageAdjustment", translator)
    
    def define_shells(self) -> List[Shell]:
        """Define Shell function for image adjustment"""
        shell = Shell("imageAdjustment")
        
        shell.add_param("image_tensor", "const dacpp::Matrix<Pixel>&", is_const=True)
        shell.add_param("image_tensor2", "dacpp::Matrix<Pixel>&", is_const=False)
        
        shell.add_index("idx1")
        shell.add_index("idx2")
        
        shell.add_slice("image_tensor", ["idx1", "idx2"])
        shell.add_slice("image_tensor2", ["idx1", "idx2"])
        
        return [shell]
    
    def define_calcs(self) -> List[Calc]:
        """Define Calc function for image adjustment"""
        calc1 = Calc("image_1")
        calc1.add_param("image_tensor", "Pixel*")
        calc1.add_param("image_tensor2", "Pixel*")
        calc1.set_body("""image_tensor2[0].r = std::min(255, image_tensor[0].r + 50);""")
        
        calc2 = Calc("image_2")
        calc2.add_param("image_tensor2", "Pixel*")
        calc2.add_param("image_tensor3", "Pixel*")
        calc2.set_body("""int value = 30;
    image_tensor3[0].r = std::min(255, image_tensor2[0].r + value);
    image_tensor3[0].g = std::min(255, image_tensor2[0].g + value);
    image_tensor3[0].b = std::min(255, image_tensor2[0].b + value);""")
        
        return [calc1, calc2]
    
    def define_expressions(self) -> List[Expression]:
        """Define data association expression"""
        shell = self.define_shells()[0]
        calc1 = self.define_calcs()[0]
        calc2 = self.define_calcs()[1]
        
        expr1 = Expression(shell, calc1, ["image_tensor", "image_tensor2"])
        # Note: image_2 uses different shell, would need separate shell definition
        # For now, just return expr1
        return [expr1]
    
    def build(self) -> str:
        """Build complete C++ code with Pixel struct"""
        # Get DSL definitions
        self.shells = self.define_shells()
        self.calcs = self.define_calcs()
        self.expressions = self.define_expressions()
        self.main_code = self.generate_main_code()
        
        # Build includes
        code = """#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

// Define Pixel structure
struct Pixel {
    int r, g, b;
    friend std::ostream& operator<<(std::ostream& os, const Pixel& pixel) {
        os << "(" << pixel.r << ", " << pixel.g << ", " << pixel.b << ")";
        return os;
    }
};

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
    
    def generate_main_code(self) -> str:
        """Generate main() function"""
        code = """int main() {
    int width, height;
    std::cout << "Enter width: ";
    std::cin >> width;
    std::cout << "Enter height: ";
    std::cin >> height;
    
    std::vector<Pixel> image(height*width, {100, 100, 100});
    std::vector<Pixel> image2(height*width, {100, 100, 100});
    
    // Print initial image
    std::cout << "Original Image:" << std::endl;
    
    dacpp::Matrix<Pixel> image_tensor({height, width}, image);
    dacpp::Matrix<Pixel> image_tensor2({height, width}, image2);
    
    // Execute color adjustment
    imageAdjustment(image_tensor, image_tensor2) <-> image_1;
    std::cout << "\\nImage After Color Adjustment:" << std::endl;
    
    std::vector<Pixel> image3 = image2;
    dacpp::Matrix<Pixel> image_tensor3({height, width}, image3);
    
    // Execute brightness enhancement
    imageAdjustment(image_tensor2, image_tensor3) <-> image_2;
    std::cout << "\\nImage After Brightness Enhancement:" << std::endl;
    image_tensor3.print();
    
    return 0;
}
"""
        return code
