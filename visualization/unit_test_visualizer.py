"""
Unit Test Result Visualization Module

Generate 3 unit test performance analysis charts according to VISUALIZATION_README.md
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

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


class UnitTestVisualizer:
    """Unit test result visualizer"""
    
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
        
        # Color scheme
        self.colors = {
            'primary': '#2E86AB',      # Blue
            'secondary': '#A23B72',     # Purple
            'success': '#06A77D',      # Green
            'warning': '#F18F01',      # Orange
            'danger': '#C73E1D',       # Red
            'neutral': '#6C757D',      # Gray
        }
    
    def load_data(self, json_path: str) -> Dict[str, Any]:
        """Load JSON test report"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def plot_slowest_tests(self, data: Dict[str, Any]) -> str:
        """
        Generate slowest test cases chart (unit_test_results_slowest_tests.png)
        Contains 1 subplot: slowest test cases (horizontal bar chart)
        """
        tests = data.get('tests', [])
        
        # Extract test data
        test_data = []
        for test in tests:
            if test.get('outcome') == 'passed':
                call = test.get('call', {})
                duration = call.get('duration', 0) * 1000  # Convert to milliseconds
                if duration > 0:
                    nodeid = test.get('nodeid', 'Unknown')
                    test_data.append((nodeid, duration))
        
        # Sort by execution time, take top 15
        test_data.sort(key=lambda x: x[1], reverse=True)
        top_15 = test_data[:15]
        
        if not top_15:
            # If no data, create an empty chart
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.text(0.5, 0.5, 'No test data available', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Slowest Test Cases', fontweight='bold', pad=10)
        else:
            test_names, durations = zip(*top_15)
            
            # Create horizontal bar chart
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Use red gradient
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(test_names)))
            
            bars = ax.barh(range(len(test_names)), durations, color=colors, 
                          edgecolor='black', linewidth=1.2, alpha=0.8)
            
            # Add value labels
            for i, (bar, duration) in enumerate(zip(bars, durations)):
                ax.text(bar.get_width() + max(durations) * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{duration:.1f} ms', ha='left', va='center', fontweight='bold')
            
            ax.set_yticks(range(len(test_names)))
            ax.set_yticklabels(test_names)
            ax.set_xlabel('Execution Time (milliseconds)', fontweight='bold')
            ax.set_title('Slowest Test Cases', fontweight='bold', pad=10)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save image
        output_path = self.output_dir / "unit_test_results_slowest_tests.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def plot_module_performance(self, data: Dict[str, Any]) -> str:
        """
        Generate module performance comparison chart (unit_test_results_module_performance.png)
        Contains 2 subplots: average execution time per module and total execution time per module
        """
        tests = data.get('tests', [])
        
        # Group by module
        module_data = defaultdict(list)
        for test in tests:
            if test.get('outcome') == 'passed':
                call = test.get('call', {})
                duration = call.get('duration', 0) * 1000  # Convert to milliseconds
                if duration > 0:
                    nodeid = test.get('nodeid', '')
                    # Extract module name (e.g., tests/test_config.py::TestPyDACConfig::test_default_config)
                    if '::' in nodeid:
                        module_name = nodeid.split('::')[0]
                        # Simplify module name (keep only filename)
                        module_name = Path(module_name).stem
                    else:
                        module_name = nodeid.split('/')[-1] if '/' in nodeid else nodeid
                    
                    module_data[module_name].append(duration)
        
        if not module_data:
            # If no data, create empty charts
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            ax1.text(0.5, 0.5, 'No test data available', 
                   ha='center', va='center', fontsize=14, transform=ax1.transAxes)
            ax1.set_title('Average Execution Time per Module', fontweight='bold', pad=10)
            ax2.text(0.5, 0.5, 'No test data available',
                   ha='center', va='center', fontsize=14, transform=ax2.transAxes)
            ax2.set_title('Total Execution Time per Module', fontweight='bold', pad=10)
        else:
            # Calculate statistics
            module_stats = []
            for module, durations in module_data.items():
                avg_time = np.mean(durations)
                total_time = np.sum(durations)
                test_count = len(durations)
                module_stats.append({
                    'module': module,
                    'avg_time': avg_time,
                    'total_time': total_time,
                    'test_count': test_count
                })
            
            # Sort by average time
            module_stats.sort(key=lambda x: x['avg_time'], reverse=True)
            modules_avg = [s['module'] for s in module_stats]
            avg_times = [s['avg_time'] for s in module_stats]
            
            # Sort by total time
            module_stats.sort(key=lambda x: x['total_time'], reverse=True)
            modules_total = [s['module'] for s in module_stats]
            total_times = [s['total_time'] for s in module_stats]
            test_counts = [s['test_count'] for s in module_stats]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Left plot: Average execution time per module
            colors1 = plt.cm.viridis(np.linspace(0.2, 0.8, len(modules_avg)))
            bars1 = ax1.bar(range(len(modules_avg)), avg_times, color=colors1,
                          edgecolor='black', linewidth=1.2, alpha=0.8)
            
            # Add value labels
            for bar, avg_time in zip(bars1, avg_times):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(avg_times) * 0.01,
                        f'{avg_time:.1f} ms', ha='center', va='bottom', fontweight='bold')
            
            ax1.set_xticks(range(len(modules_avg)))
            ax1.set_xticklabels(modules_avg, rotation=45, ha='right')
            ax1.set_ylabel('Average Execution Time (milliseconds)', fontweight='bold')
            ax1.set_title('Average Execution Time per Module', fontweight='bold', pad=10)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Right plot: Total execution time per module
            colors2 = plt.cm.plasma(np.linspace(0.2, 0.8, len(modules_total)))
            bars2 = ax2.bar(range(len(modules_total)), total_times, color=colors2,
                          edgecolor='black', linewidth=1.2, alpha=0.8)
            
            # Add value labels (including test count)
            for bar, total_time, count in zip(bars2, total_times, test_counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(total_times) * 0.01,
                        f'{total_time:.1f} ms\n({count} tests)', ha='center', va='bottom', 
                        fontweight='bold', fontsize=9)
            
            ax2.set_xticks(range(len(modules_total)))
            ax2.set_xticklabels(modules_total, rotation=45, ha='right')
            ax2.set_ylabel('Total Execution Time (milliseconds)', fontweight='bold')
            ax2.set_title('Total Execution Time per Module', fontweight='bold', pad=10)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save image
        output_path = self.output_dir / "unit_test_results_module_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def plot_time_distribution(self, data: Dict[str, Any]) -> str:
        """
        Generate execution time distribution chart (unit_test_results_time_distribution.png)
        Contains 1 subplot: test execution time distribution (histogram)
        """
        tests = data.get('tests', [])
        
        # Extract execution times
        durations = []
        for test in tests:
            if test.get('outcome') == 'passed':
                call = test.get('call', {})
                duration = call.get('duration', 0) * 1000  # Convert to milliseconds
                if duration > 0:
                    durations.append(duration)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        if durations:
            # Draw histogram
            n, bins, patches = ax.hist(durations, bins=min(30, len(durations)), 
                                      color=self.colors['primary'], alpha=0.7,
                                      edgecolor='white', linewidth=1.2)
            
            # Calculate statistics
            mean_time = np.mean(durations)
            median_time = np.median(durations)
            p95_time = np.percentile(durations, 95)
            p99_time = np.percentile(durations, 99)
            
            # Add statistical lines
            ax.axvline(mean_time, color=self.colors['danger'], linestyle='--',
                      linewidth=2, label=f'Mean: {mean_time:.2f} ms')
            ax.axvline(median_time, color=self.colors['warning'], linestyle='--',
                      linewidth=2, label=f'Median: {median_time:.2f} ms')
            ax.axvline(p95_time, color=self.colors['secondary'], linestyle=':',
                      linewidth=2, label=f'95th Percentile: {p95_time:.2f} ms')
            ax.axvline(p99_time, color=self.colors['success'], linestyle=':',
                      linewidth=2, label=f'99th Percentile: {p99_time:.2f} ms')
            
            ax.set_xlabel('Execution Time (milliseconds)', fontweight='bold')
            ax.set_ylabel('Test Count', fontweight='bold')
            ax.set_title('Test Execution Time Distribution', fontweight='bold', pad=10)
            ax.legend(loc='upper right', frameon=True)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        else:
            ax.text(0.5, 0.5, 'No test data available',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Test Execution Time Distribution', fontweight='bold', pad=10)
        
        plt.tight_layout()
        
        # Save image
        output_path = self.output_dir / "unit_test_results_time_distribution.png"
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
        
        print("Generating slowest test cases chart...")
        output_paths['slowest_tests'] = self.plot_slowest_tests(data)
        
        print("Generating module performance comparison chart...")
        output_paths['module_performance'] = self.plot_module_performance(data)
        
        print("Generating execution time distribution chart...")
        output_paths['time_distribution'] = self.plot_time_distribution(data)
        
        print(f"\nAll visualization charts have been generated to: {self.output_dir}")
        
        return output_paths
