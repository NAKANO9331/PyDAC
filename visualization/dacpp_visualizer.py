"""
DACPP Test Result Visualization Module

Generate 5 performance analysis charts according to VISUALIZATION_README.md
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None


class DACPPVisualizer:
    """DACPP test result visualizer"""
    
    def __init__(self, output_dir: str = "result"):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Output directory path
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib and seaborn are required. Please install: pip install matplotlib seaborn"
            )
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set academic style
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'sans-serif',
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'patch.linewidth': 1.2,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
        
        # Color scheme (according to documentation)
        self.colors = {
            'primary': '#2E86AB',      # Blue - USM mode, translation time
            'secondary': '#A23B72',     # Purple - Buffer mode, performance distribution
            'success': '#06A77D',      # Green - Execution time
            'warning': '#F18F01',      # Orange - Compilation time, complexity scatter plot
            'danger': '#C73E1D',       # Red - Mean line, performance equality line
        }
    
    def load_data(self, json_path: str) -> Dict[str, Any]:
        """Load JSON test report"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def plot_stage_comparison(self, data: Dict[str, Any]) -> str:
        """
        Generate stage comparison chart (performance_analysis_stage_comparison.png)
        Contains 2 subplots: stage time breakdown (stacked bar chart) and USM vs Buffer performance comparison (scatter plot)
        """
        results = data.get('results', [])
        
        # Extract data
        test_names = []
        translation_times = []
        compilation_times = []
        execution_times_usm = []
        execution_times_buffer = []
        
        for result in results:
            name = result.get('name', 'Unknown')
            test_names.append(name)
            
            # Translation time (USM and Buffer should be the same, use USM)
            trans_usm = result.get('translation_usm', {})
            trans_time = trans_usm.get('duration', 0) if trans_usm else 0
            translation_times.append(trans_time)
            
            # Compilation time
            comp_usm = result.get('compilation_usm', {})
            comp_time = comp_usm.get('duration', 0) if comp_usm else 0
            compilation_times.append(comp_time)
            
            # Execution time
            exec_usm = result.get('execution_usm', {})
            exec_time_usm = exec_usm.get('duration', 0) if exec_usm and exec_usm.get('success') else 0
            execution_times_usm.append(exec_time_usm)
            
            exec_buffer = result.get('execution_buffer', {})
            exec_time_buffer = exec_buffer.get('duration', 0) if exec_buffer and exec_buffer.get('success') else 0
            execution_times_buffer.append(exec_time_buffer)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Stage time breakdown (stacked bar chart)
        x_pos = np.arange(len(test_names))
        width = 0.6
        
        bars1 = ax1.bar(x_pos, translation_times, width, 
                       label='Translation Time', color=self.colors['primary'], alpha=0.8)
        bars2 = ax1.bar(x_pos, compilation_times, width, 
                       bottom=translation_times,
                       label='Compilation Time', color=self.colors['warning'], alpha=0.8)
        bars3 = ax1.bar(x_pos, execution_times_usm, width,
                       bottom=np.array(translation_times) + np.array(compilation_times),
                       label='Execution Time', color=self.colors['success'], alpha=0.8)
        
        ax1.set_xlabel('Test Case', fontweight='bold')
        ax1.set_ylabel('Time (seconds)', fontweight='bold')
        ax1.set_title('Stage Time Breakdown', fontweight='bold', pad=10)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(test_names, rotation=45, ha='right')
        ax1.legend(loc='upper left', frameon=True)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Right plot: USM vs Buffer performance comparison (scatter plot)
        # Filter out data points with execution time of 0
        valid_data = [(usm, buf) for usm, buf in zip(execution_times_usm, execution_times_buffer) 
                     if usm > 0 and buf > 0]
        
        if valid_data:
            usm_times, buffer_times = zip(*valid_data)
            ax2.scatter(usm_times, buffer_times, alpha=0.6, s=100, 
                       color=self.colors['secondary'], edgecolors='black', linewidth=1)
            
            # Add performance equality line (y=x)
            max_time = max(max(usm_times), max(buffer_times))
            ax2.plot([0, max_time], [0, max_time], 'r--', linewidth=2, 
                    label='Performance Equality Line', alpha=0.7)
            
            # Add test case labels
            for i, (usm, buf) in enumerate(zip(execution_times_usm, execution_times_buffer)):
                if usm > 0 and buf > 0:
                    ax2.annotate(test_names[i], (usm, buf), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=8, alpha=0.7)
        
        ax2.set_xlabel('USM Execution Time (seconds)', fontweight='bold')
        ax2.set_ylabel('Buffer Execution Time (seconds)', fontweight='bold')
        ax2.set_title('USM vs Buffer Performance Comparison', fontweight='bold', pad=10)
        ax2.legend(loc='upper left', frameon=True)
        ax2.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save image
        output_path = self.output_dir / "performance_analysis_stage_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def plot_distribution_complexity(self, data: Dict[str, Any]) -> str:
        """
        Generate distribution and complexity chart (performance_analysis_distribution_complexity.png)
        Contains 2 subplots: performance distribution (histogram) and code complexity vs performance (scatter plot)
        """
        results = data.get('results', [])
        
        # Extract execution times
        execution_times = []
        file_sizes = []
        execution_times_for_complexity = []
        
        for result in results:
            exec_usm = result.get('execution_usm', {})
            if exec_usm and exec_usm.get('success'):
                exec_time = exec_usm.get('duration', 0)
                if exec_time > 0:
                    execution_times.append(exec_time)
                    execution_times_for_complexity.append(exec_time)
                    
                    # File size
                    file_stats = result.get('file_stats', {})
                    file_size_kb = file_stats.get('file_size_kb', 0)
                    file_sizes.append(file_size_kb)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Performance distribution (histogram)
        if execution_times:
            ax1.hist(execution_times, bins=min(20, len(execution_times)), 
                    color=self.colors['secondary'], alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add mean line
            mean_time = np.mean(execution_times)
            ax1.axvline(mean_time, color=self.colors['danger'], linestyle='--', 
                       linewidth=2, label=f'Mean: {mean_time:.3f}s')
            
            ax1.set_xlabel('Execution Time (seconds)', fontweight='bold')
            ax1.set_ylabel('Frequency', fontweight='bold')
            ax1.set_title('Performance Distribution', fontweight='bold', pad=10)
            ax1.legend(loc='upper right', frameon=True)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Right plot: Code complexity vs performance (scatter plot)
        if file_sizes and execution_times_for_complexity:
            ax2.scatter(file_sizes, execution_times_for_complexity, alpha=0.6, s=100,
                       color=self.colors['warning'], edgecolors='black', linewidth=1)
            
            ax2.set_xlabel('File Size (KB)', fontweight='bold')
            ax2.set_ylabel('Execution Time (seconds)', fontweight='bold')
            ax2.set_title('Code Complexity vs Performance', fontweight='bold', pad=10)
            ax2.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save image
        output_path = self.output_dir / "performance_analysis_distribution_complexity.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def plot_performance_trends(self, data: Dict[str, Any]) -> str:
        """
        Generate performance trends chart (performance_analysis_performance_trends.png)
        Contains 1 subplot: performance trend timeline
        """
        results = data.get('results', [])
        
        # Extract data
        test_indices = []
        usm_times = []
        buffer_times = []
        
        for i, result in enumerate(results):
            exec_usm = result.get('execution_usm', {})
            exec_buffer = result.get('execution_buffer', {})
            
            usm_time = exec_usm.get('duration', 0) if exec_usm and exec_usm.get('success') else None
            buffer_time = exec_buffer.get('duration', 0) if exec_buffer and exec_buffer.get('success') else None
            
            if usm_time is not None or buffer_time is not None:
                test_indices.append(i)
                usm_times.append(usm_time if usm_time is not None else np.nan)
                buffer_times.append(buffer_time if buffer_time is not None else np.nan)
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        
        # Draw trend lines
        if usm_times:
            valid_usm = [(idx, t) for idx, t in zip(test_indices, usm_times) if not np.isnan(t)]
            if valid_usm:
                idxs, times = zip(*valid_usm)
                ax.plot(idxs, times, 'o-', label='USM', color=self.colors['primary'], 
                       linewidth=2, markersize=8, markerfacecolor='white', 
                       markeredgewidth=2, markeredgecolor=self.colors['primary'])
        
        if buffer_times:
            valid_buffer = [(idx, t) for idx, t in zip(test_indices, buffer_times) if not np.isnan(t)]
            if valid_buffer:
                idxs, times = zip(*valid_buffer)
                ax.plot(idxs, times, 's-', label='Buffer', color=self.colors['secondary'],
                       linewidth=2, markersize=8, markerfacecolor='white',
                       markeredgewidth=2, markeredgecolor=self.colors['secondary'])
        
        ax.set_xlabel('Test Case Sequence', fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontweight='bold')
        ax.set_title('Performance Trend Timeline', fontweight='bold', pad=10)
        ax.legend(loc='best', frameon=True)
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save image
        output_path = self.output_dir / "performance_analysis_performance_trends.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def plot_complexity_performance(self, data: Dict[str, Any]) -> str:
        """
        Generate complexity and performance relationship chart (code_complexity_analysis_complexity_performance.png)
        Contains 3 subplots: file size vs performance, code lines vs performance, operation count vs performance
        """
        results = data.get('results', [])
        
        # Extract data
        file_sizes = []
        code_lines = []
        total_operations = []
        execution_times = []
        test_indices = []
        
        for i, result in enumerate(results):
            exec_usm = result.get('execution_usm', {})
            if exec_usm and exec_usm.get('success'):
                exec_time = exec_usm.get('duration', 0)
                if exec_time > 0:
                    file_stats = result.get('file_stats', {})
                    analysis = result.get('analysis', {})
                    
                    file_sizes.append(file_stats.get('file_size_kb', 0))
                    code_lines.append(file_stats.get('code_lines', 0))
                    
                    shell_count = analysis.get('shell_count', 0)
                    calc_count = analysis.get('calc_count', 0)
                    total_operations.append(shell_count + calc_count)
                    
                    execution_times.append(exec_time)
                    test_indices.append(i)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Left plot: File size vs performance
        if file_sizes and execution_times:
            scatter = ax1.scatter(file_sizes, execution_times, c=test_indices, 
                                 cmap='viridis', alpha=0.6, s=100, 
                                 edgecolors='black', linewidth=1)
            ax1.set_xlabel('File Size (KB)', fontweight='bold')
            ax1.set_ylabel('USM Execution Time (seconds)', fontweight='bold')
            ax1.set_title('File Size vs Performance', fontweight='bold', pad=10)
            ax1.grid(alpha=0.3, linestyle='--')
            plt.colorbar(scatter, ax=ax1, label='Test Case Index')
        
        # Middle plot: Code lines vs performance
        if code_lines and execution_times:
            scatter = ax2.scatter(code_lines, execution_times, c=test_indices,
                                 cmap='plasma', alpha=0.6, s=100,
                                 edgecolors='black', linewidth=1)
            ax2.set_xlabel('Code Lines', fontweight='bold')
            ax2.set_ylabel('USM Execution Time (seconds)', fontweight='bold')
            ax2.set_title('Code Lines vs Performance', fontweight='bold', pad=10)
            ax2.grid(alpha=0.3, linestyle='--')
            plt.colorbar(scatter, ax=ax2, label='Test Case Index')
        
        # Right plot: Operation count vs performance
        if total_operations and execution_times:
            scatter = ax3.scatter(total_operations, execution_times, c=test_indices,
                                 cmap='coolwarm', alpha=0.6, s=100,
                                 edgecolors='black', linewidth=1)
            ax3.set_xlabel('Total Operations (Shell + Calc)', fontweight='bold')
            ax3.set_ylabel('USM Execution Time (seconds)', fontweight='bold')
            ax3.set_title('Operation Count vs Performance', fontweight='bold', pad=10)
            ax3.grid(alpha=0.3, linestyle='--')
            plt.colorbar(scatter, ax=ax3, label='Test Case Index')
        
        plt.tight_layout()
        
        # Save image
        output_path = self.output_dir / "code_complexity_analysis_complexity_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def plot_correlation_distribution(self, data: Dict[str, Any]) -> str:
        """
        Generate correlation matrix and distribution chart (code_complexity_analysis_correlation_distribution.png)
        Contains 2 subplots: complexity-performance correlation matrix (heatmap) and complexity distribution (grouped bar chart)
        """
        results = data.get('results', [])
        
        # Extract data
        file_sizes = []
        code_lines = []
        shell_counts = []
        calc_counts = []
        execution_times = []
        
        for result in results:
            exec_usm = result.get('execution_usm', {})
            if exec_usm and exec_usm.get('success'):
                exec_time = exec_usm.get('duration', 0)
                if exec_time > 0:
                    file_stats = result.get('file_stats', {})
                    analysis = result.get('analysis', {})
                    
                    file_sizes.append(file_stats.get('file_size_kb', 0))
                    code_lines.append(file_stats.get('code_lines', 0))
                    shell_counts.append(analysis.get('shell_count', 0))
                    calc_counts.append(analysis.get('calc_count', 0))
                    execution_times.append(exec_time)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Correlation matrix (heatmap)
        if len(file_sizes) > 1:
            # Build data matrix
            data_matrix = np.array([
                file_sizes,
                code_lines,
                shell_counts,
                calc_counts,
                execution_times
            ])
            
            # Calculate correlation coefficient matrix
            corr_matrix = np.corrcoef(data_matrix)
            
            # Labels
            labels = ['File Size', 'Code Lines', 'Shell Count', 'Calc Count', 'USM Execution Time']
            
            # Draw heatmap
            im = ax1.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            # Set ticks
            ax1.set_xticks(np.arange(len(labels)))
            ax1.set_yticks(np.arange(len(labels)))
            ax1.set_xticklabels(labels, rotation=45, ha='right')
            ax1.set_yticklabels(labels)
            
            # Add value annotations
            for i in range(len(labels)):
                for j in range(len(labels)):
                    text = ax1.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontweight='bold')
            
            ax1.set_title('Complexity-Performance Correlation Matrix', fontweight='bold', pad=10)
            plt.colorbar(im, ax=ax1, label='Correlation Coefficient')
        
        # Right plot: Complexity distribution (grouped bar chart)
        if file_sizes and code_lines and shell_counts and calc_counts:
            # Normalize data to 0-1 range
            file_sizes_norm = np.array(file_sizes) / max(file_sizes) if max(file_sizes) > 0 else np.array(file_sizes)
            code_lines_norm = np.array(code_lines) / max(code_lines) if max(code_lines) > 0 else np.array(code_lines)
            total_ops = np.array(shell_counts) + np.array(calc_counts)
            total_ops_norm = total_ops / max(total_ops) if max(total_ops) > 0 else total_ops
            
            # Calculate mean and standard deviation
            mean_file_size = np.mean(file_sizes_norm)
            mean_code_lines = np.mean(code_lines_norm)
            mean_ops = np.mean(total_ops_norm)
            
            std_file_size = np.std(file_sizes_norm)
            std_code_lines = np.std(code_lines_norm)
            std_ops = np.std(total_ops_norm)
            
            # Raw mean values (for annotation)
            mean_file_size_raw = np.mean(file_sizes)
            mean_code_lines_raw = np.mean(code_lines)
            mean_ops_raw = np.mean(total_ops)
            
            # Draw bar chart
            categories = ['File Size', 'Code Lines', 'Operations']
            means = [mean_file_size, mean_code_lines, mean_ops]
            stds = [std_file_size, std_code_lines, std_ops]
            raw_means = [mean_file_size_raw, mean_code_lines_raw, mean_ops_raw]
            colors = [self.colors['primary'], self.colors['secondary'], self.colors['success']]
            
            x_pos = np.arange(len(categories))
            bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8,
                          color=colors, edgecolor='black', linewidth=1.2)
            
            # Add raw mean value annotations
            for bar, raw_mean in zip(bars, raw_means):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + bar.get_yerr()[1] + 0.02,
                        f'{raw_mean:.1f}', ha='center', va='bottom', fontweight='bold')
            
            ax2.set_ylabel('Normalized Value (0-1)', fontweight='bold')
            ax2.set_title('Complexity Distribution', fontweight='bold', pad=10)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(categories)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save image
        output_path = self.output_dir / "code_complexity_analysis_correlation_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def generate_all_visualizations(self, json_path: str) -> Dict[str, str]:
        """
        Generate all visualization charts
        
        Args:
            json_path: JSON test report path
            
        Returns:
            Dictionary of generated image paths
        """
        data = self.load_data(json_path)
        
        output_paths = {}
        
        print("Generating stage comparison chart...")
        output_paths['stage_comparison'] = self.plot_stage_comparison(data)
        
        print("Generating distribution and complexity chart...")
        output_paths['distribution_complexity'] = self.plot_distribution_complexity(data)
        
        print("Generating performance trends chart...")
        output_paths['performance_trends'] = self.plot_performance_trends(data)
        
        print("Generating complexity and performance relationship chart...")
        output_paths['complexity_performance'] = self.plot_complexity_performance(data)
        
        print("Generating correlation matrix and distribution chart...")
        output_paths['correlation_distribution'] = self.plot_correlation_distribution(data)
        
        print(f"\nAll visualization charts have been generated to: {self.output_dir}")
        
        return output_paths
