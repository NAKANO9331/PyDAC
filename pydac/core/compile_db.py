"""Compilation database generation for PyDAC"""


import json

from pathlib import Path

from typing import List, Optional


class CompileDBGenerator:

    """Generate Compile_commands.json for translator"""

    def __init__(self):
        """Initialize generator"""
        pass

    def generate(
        self,
        source_file: str,
        include_dirs: Optional[List[str]] = None,
        Compiler: str = "clang++",
        flags: Optional[List[str]] = None,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate Compile_commands.json

        Args:
            source_file: Source file path
            include_dirs: Include directories
            Compiler: Compiler name
            flags: Compilation flags
            output_file: Output file path (None for Compile_commands.json in source directory)

        Returns:
            Path to generated Compile_commands.json
        """
        source_path = Path(source_file).resolve()
        source_dir = source_path.parent

        # Default include directories
        if include_dirs is None:
            include_dirs = []

        # Add project-specific includes
        project_root = source_path
        max_depth = 10  # Prevent infinite loop
        depth = 0
        while project_root.parent != project_root and depth < max_depth:
            project_root = project_root.parent
            depth += 1
        # Check for project-internal translator directory
        compile_db_file = Path(__file__).resolve()
        pydac_root = compile_db_file.parent.parent.parent  # PyDAC
        translator_root = pydac_root / "translator"  # PyDAC/translator
        for include_name in ["dpcppLib/include", "dacppLib/include", "rewriter/include", "parser/include"]:
            include_path = translator_root / include_name
            if include_path.exists() and str(include_path) not in include_dirs:
                include_dirs.append(str(include_path))

        # Default flags
        if flags is None:
            flags = ["-std=c++17", "-I" + str(source_dir)]

        # Add include directories to flags
        for include_dir in include_dirs:
            if include_dir not in flags:
                flags.append(f"-I{include_dir}")

        # Create Compile command
        Compile_command = {
            "directory": str(source_dir),
            "command": f"{Compiler} {' '.join(flags)} -c {source_path.name}",
            "file": str(source_path)
        }

        # Write compile_commands.json (lowercase, as required by clang tools)
        if output_file is None:
            output_file = source_dir / "compile_commands.json"
        else:
            output_file = Path(output_file)

        # Read existing Compile_commands.json if exists
        commands = []
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    commands = json.load(f)
            except (json.JSONDecodeError, IOError):
                commands = []

        # Add or update entry for this file
        updated = False
        for i, cmd in enumerate(commands):
            if cmd.get("file") == str(source_path):
                commands[i] = Compile_command
                updated = True
                break

        if not updated:
            commands.append(Compile_command)

        # Write Compile_commands.json
        with open(output_file, 'w') as f:
            json.dump(commands, f, indent=2)

        return str(output_file)

    def generate_for_test_case(
        self,
        test_file: str,
        test_dir: Optional[str] = None
    ) -> str:
        """
        Generate Compile_commands.json for a test case

        Args:
            test_file: Test file path
            test_dir: Test directory (for finding includes)

        Returns:
            Path to generated Compile_commands.json
        """
        test_path = Path(test_file).resolve()
        test_case_dir = test_path.parent

        # Find include directories
        include_dirs = []

        # Look for ReconTensor.h in project-internal translator directory
        compile_db_file = Path(__file__).resolve()
        pydac_root = compile_db_file.parent.parent.parent  # PyDAC
        translator_root = pydac_root / "translator"  # PyDAC/translator
        recon_tensor = translator_root / "dacppLib" / "include" / "ReconTensor.h"
        if recon_tensor.exists():
            include_dirs.append(str(recon_tensor.parent))
            include_dirs.append(str(translator_root / "dpcppLib" / "include"))
            include_dirs.append(str(translator_root / "rewriter" / "include"))
            include_dirs.append(str(translator_root / "parser" / "include"))

        # Add common includes
        include_dirs.extend([
            str(test_case_dir),
        ])

        return self.generate(
            str(test_path),
            include_dirs=include_dirs,
            Compiler="clang++",
            flags=["-std=c++17", "-I/usr/include", "-I/usr/local/include"]
        )
