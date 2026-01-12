"""
Translation Result Comparator

Compare SYCL code generated from DSL with SYCL code translated from original code via DACPP
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


class TranslationComparator:
    """Compare SYCL code translated by two different methods"""
    
    def __init__(self):
        """Initialize the comparator"""
        pass
    
    def compare_sycl_files(
        self,
        dsl_sycl_file: Path,
        original_sycl_file: Path
    ) -> Dict[str, Any]:
        """
        Compare two SYCL files
        
        Args:
            dsl_sycl_file: Path to SYCL file translated from DSL-generated code
            original_sycl_file: Path to SYCL file translated from original code
            
        Returns:
            Comparison result dictionary
        """
        result = {
            "dsl_file": str(dsl_sycl_file),
            "original_file": str(original_sycl_file),
            "identical": False,
            "similarity": 0.0,
            "differences": [],
            "difference_count": 0,
            "key_functions_match": False,
            "dsl_line_count": 0,
            "original_line_count": 0
        }
        
        # Check if files exist
        if not dsl_sycl_file.exists():
            result["error"] = f"DSL SYCL file not found: {dsl_sycl_file}"
            return result
        
        if not original_sycl_file.exists():
            result["error"] = f"Original SYCL file not found: {original_sycl_file}"
            return result
        
        # Read file contents
        try:
            with open(dsl_sycl_file, 'r', encoding='utf-8') as f:
                dsl_content = f.read()
            
            with open(original_sycl_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            result["error"] = f"Failed to read files: {e}"
            return result
        
        # Exact match comparison
        if dsl_content == original_content:
            result["identical"] = True
            result["similarity"] = 1.0
            result["key_functions_match"] = True
            result["dsl_line_count"] = len(dsl_content.split('\n'))
            result["original_line_count"] = len(original_content.split('\n'))
            return result
        
        # Normalized comparison
        dsl_normalized = self.normalize_code(dsl_content)
        original_normalized = self.normalize_code(original_content)
        
        if dsl_normalized == original_normalized:
            result["identical"] = False
            result["similarity"] = 0.95  # Only formatting differences
            result["key_functions_match"] = True
        else:
            # Calculate similarity
            result["similarity"] = self.calculate_similarity(
                dsl_normalized,
                original_normalized
            )
            
            # Extract and compare key functions
            dsl_key_functions = self.extract_key_functions(dsl_content)
            original_key_functions = self.extract_key_functions(original_content)
            result["key_functions_match"] = self.compare_key_functions(
                dsl_key_functions,
                original_key_functions
            )
        
        # Calculate differences
        dsl_lines = dsl_content.split('\n')
        original_lines = original_content.split('\n')
        result["dsl_line_count"] = len(dsl_lines)
        result["original_line_count"] = len(original_lines)
        
        differences = self.find_differences(dsl_lines, original_lines)
        result["differences"] = differences[:20]  # Keep only first 20 differences
        result["difference_count"] = len(differences)
        
        return result
    
    def normalize_code(self, code: str) -> str:
        """
        Normalize code (remove comments, whitespace, etc.)
        
        Args:
            code: Original code
            
        Returns:
            Normalized code
        """
        lines = code.split('\n')
        normalized_lines = []
        
        for line in lines:
            # Remove line-end comments (// comments)
            if '//' in line:
                line = line[:line.index('//')]
            
            # Remove inline comments (/* */ comments, simple handling)
            line = re.sub(r'/\*.*?\*/', '', line)
            
            # Remove leading and trailing whitespace
            line = line.strip()
            
            # Skip empty lines
            if line:
                # Normalize whitespace (multiple spaces to single space)
                line = re.sub(r'\s+', ' ', line)
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def calculate_similarity(self, code1: str, code2: str) -> float:
        """
        Calculate similarity between two code snippets (based on longest common subsequence)
        
        Args:
            code1: First code snippet
            code2: Second code snippet
            
        Returns:
            Similarity score (0-1)
        """
        lines1 = code1.split('\n')
        lines2 = code2.split('\n')
        
        # Simple similarity calculation: ratio of identical lines
        set1 = set(lines1)
        set2 = set(lines2)
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def extract_key_functions(self, code: str) -> Dict[str, str]:
        """
        Extract key functions (kernel functions, main computation functions, etc.)
        
        Args:
            code: SYCL code
            
        Returns:
            Dictionary mapping function names to function bodies
        """
        key_functions = {}
        
        # Find SYCL kernel functions (using q.submit or parallel_for)
        kernel_pattern = r'(?:q\.submit|parallel_for).*?\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        kernels = re.findall(kernel_pattern, code, re.DOTALL)
        
        for i, kernel in enumerate(kernels):
            key_functions[f"kernel_{i}"] = self.normalize_code(kernel)
        
        # Find main function definitions
        function_pattern = r'(?:void|int|float|double)\s+(\w+)\s*\([^)]*\)\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        functions = re.finditer(function_pattern, code, re.DOTALL)
        
        for match in functions:
            func_name = match.group(1)
            func_body = match.group(0)
            if func_name not in ['main']:  # Exclude main function
                key_functions[func_name] = self.normalize_code(func_body)
        
        return key_functions
    
    def compare_key_functions(
        self,
        funcs1: Dict[str, str],
        funcs2: Dict[str, str]
    ) -> bool:
        """
        Compare whether key functions match
        
        Args:
            funcs1: First function set
            funcs2: Second function set
            
        Returns:
            Whether functions match
        """
        if len(funcs1) != len(funcs2):
            return False
        
        # Compare each function
        for name, body1 in funcs1.items():
            if name not in funcs2:
                return False
            
            body2 = funcs2[name]
            if body1 != body2:
                # Allow small differences (e.g., variable names)
                similarity = self.calculate_similarity(body1, body2)
                if similarity < 0.9:  # Similarity threshold
                    return False
        
        return True
    
    def find_differences(
        self,
        lines1: List[str],
        lines2: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Find differences between two code snippets
        
        Args:
            lines1: List of lines from first code
            lines2: List of lines from second code
            
        Returns:
            List of differences
        """
        differences = []
        max_len = max(len(lines1), len(lines2))
        
        for i in range(max_len):
            line1 = lines1[i] if i < len(lines1) else None
            line2 = lines2[i] if i < len(lines2) else None
            
            if line1 != line2:
                # Compare again after normalization
                norm1 = self.normalize_code(line1 or "")
                norm2 = self.normalize_code(line2 or "")
                
                if norm1 != norm2:
                    differences.append({
                        "line_number": i + 1,
                        "dsl_line": line1,
                        "original_line": line2,
                        "dsl_normalized": norm1,
                        "original_normalized": norm2
                    })
        
        return differences
