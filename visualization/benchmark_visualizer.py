"""Performance visualization for PyDAC benchmarks"""


import os

from pathlib import Path

from typing import List, Dict, Optional, Tuple

from dataclasses import dataclass


# Optional dependency for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    np = None

from pydac.utils.benchmark import BenchmarkResult, ComparisonResult


class PerformanceVisualizer:

    """Visualize performance benchmark results"""

    def __init__(self, output_dir: str = "./performance_plots"):
        """
        Initialize visualizer

 Args:
     output_dir: Directory to save plots
     """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set academic style
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times', 'serif'],
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
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'xtick.minor.width': 0.8,
            'ytick.minor.width': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
        
        # Use academic color palette (colorblind-friendly)
        self.colors = {
            'primary': '#2E86AB',      # Blue
            'secondary': '#A23B72',   # Purple
            'success': '#06A77D',      # Green
            'warning': '#F18F01',      # Orange
            'danger': '#C73E1D',       # Red
            'neutral': '#6C757D',      # Gray
            'light_blue': '#6BB6FF',
            'light_green': '#4ECDC4',
        }

    def plot_comparison(
        self,
        comparison: ComparisonResult,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> str:
        """
        Plot performance comparison with academic styling

        Args:
            comparison: ComparisonResult object
            save_path: Path to save plot (optional)
            show: Whether to display plot

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Performance Comparison: {comparison.operation}', 
                     fontsize=14, fontweight='bold', y=0.995)

        methods = ['Direct', 'PyDAC']
        bar_colors = [self.colors['primary'], self.colors['secondary']]

        # 1. Duration comparison (bar chart)
        ax1 = axes[0, 0]
        durations = [comparison.direct_avg, comparison.pydac_avg]
        bars = ax1.bar(methods, durations, color=bar_colors, alpha=0.85, 
                       edgecolor='black', linewidth=1.2, width=0.6)
        ax1.set_ylabel('Average Duration (s)', fontsize=11, fontweight='bold')
        ax1.set_title('(a) Execution Time', fontsize=12, fontweight='bold', pad=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Add value labels on bars
        for bar, duration in zip(bars, durations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                     f'{duration:.3f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Add overhead annotation
        overhead_text = f'Overhead: {comparison.overhead:.1f}%'
        if comparison.overhead > 0:
            overhead_text += f' ({comparison.overhead_abs:.3f} s)'
            ax1.text(0.98, 0.98, overhead_text, transform=ax1.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
            edgecolor='gray', alpha=0.8))

        # 2. Speedup ratio
        ax2 = axes[0, 1]
        speedup = comparison.speedup
        if speedup < 1.1:
            color = self.colors['success']
        elif speedup < 1.5:
            color = self.colors['warning']
        else:
            color = self.colors['danger']
        
        bars = ax2.bar(['Speedup'], [speedup], color=color, alpha=0.85, 
                       edgecolor='black', linewidth=1.2, width=0.5)
        ax2.set_ylabel('Ratio (PyDAC / Direct)', fontsize=11, fontweight='bold')
        ax2.set_title('(b) Performance Ratio', fontsize=12, fontweight='bold', pad=10)
        ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, 
                    alpha=0.7, label='Baseline (1.0)')
        ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Add value label
        ax2.text(0, speedup + (speedup*0.05 if speedup > 1 else -speedup*0.05),
                 f'{speedup:.3f}×',
                 ha='center', va='bottom' if speedup > 1 else 'top',
                 fontsize=11, fontweight='bold')

        # 3. CPU usage comparison
        ax3 = axes[1, 0]
        cpu_normalized = [100, 100 + comparison.cpu_diff]
        bars = ax3.bar(methods, cpu_normalized, color=bar_colors, alpha=0.85, 
                       edgecolor='black', linewidth=1.2, width=0.6)
        ax3.set_ylabel('CPU Usage (%)', fontsize=11, fontweight='bold')
        ax3.set_title('(c) CPU Usage', fontsize=12, fontweight='bold', pad=10)
        ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # Add value labels
        for bar, cpu in zip(bars, cpu_normalized):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                     f'{cpu:.1f}%',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

        if comparison.cpu_diff != 0:
            diff_text = f'Δ = {comparison.cpu_diff:+.1f}%'
            ax3.text(0.98, 0.98, diff_text, transform=ax3.transAxes,
                     ha='right', va='top', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                               edgecolor='gray', alpha=0.8))

        # 4. Memory usage comparison
        ax4 = axes[1, 1]
        memory_values = [100, 100 + comparison.memory_diff]
        bars = ax4.bar(methods, memory_values, color=bar_colors, alpha=0.85, 
                       edgecolor='black', linewidth=1.2, width=0.6)
        ax4.set_ylabel('Memory Usage (relative)', fontsize=11, fontweight='bold')
        ax4.set_title('(d) Memory Usage', fontsize=12, fontweight='bold', pad=10)
        ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)

        # Add value labels
        for bar, mem in zip(bars, memory_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                     f'{mem:.1f}%',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

        if comparison.memory_diff != 0:
            diff_text = f'Δ = {comparison.memory_diff:+.1f}%'
            ax4.text(0.98, 0.98, diff_text, transform=ax4.transAxes,
                     ha='right', va='top', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                               edgecolor='gray', alpha=0.8))

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"comparison_{comparison.operation}.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', pad_inches=0.1)

        if show:
            plt.show()
        else:
            plt.close()

        return str(save_path)

    def plot_benchmark_results(
        self,
        results: List[BenchmarkResult],
        save_path: Optional[str] = None,
        show: bool = False
    ) -> str:
        """
        Plot detailed benchmark results with academic styling

        Args:
            results: List of BenchmarkResult objects
            save_path: Path to save plot (optional)
            show: Whether to display plot

        Returns:
            Path to saved plot
        """
        # Separate results by method
        direct_results = [r for r in results if r.method == "direct"]
        pydac_results = [r for r in results if r.method == "pydac"]

        if not direct_results or not pydac_results:
            raise ValueError("Need both direct and pydac results")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        operation = results[0].operation if results else "benchmark"
        fig.suptitle(f'Detailed Benchmark Results: {operation}', 
                     fontsize=14, fontweight='bold', y=0.995)

        # 1. Duration distribution
        ax1 = axes[0, 0]
        direct_durations = [r.duration for r in direct_results]
        pydac_durations = [r.duration for r in pydac_results]

        bins = max(10, min(len(direct_durations), 20))
        ax1.hist(direct_durations, bins=bins, alpha=0.7, label='Direct', 
                 color=self.colors['primary'], edgecolor='black', linewidth=1.0)
        ax1.hist(pydac_durations, bins=bins, alpha=0.7, label='PyDAC', 
                 color=self.colors['secondary'], edgecolor='black', linewidth=1.0)
        ax1.set_xlabel('Duration (s)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('(a) Duration Distribution', fontsize=12, fontweight='bold', pad=10)
        ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # 2. Duration over iterations (line plot)
        ax2 = axes[0, 1]
        iterations = range(1, len(direct_results) + 1)
        ax2.plot(iterations, direct_durations, 'o-', label='Direct', 
                 color=self.colors['primary'], linewidth=2, markersize=6, 
                 markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=self.colors['primary'])
        ax2.plot(iterations, pydac_durations, 's-', label='PyDAC', 
                 color=self.colors['secondary'], linewidth=2, markersize=6,
                 markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=self.colors['secondary'])
        ax2.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Duration (s)', fontsize=11, fontweight='bold')
        ax2.set_title('(b) Duration Over Iterations', fontsize=12, fontweight='bold', pad=10)
        ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax2.grid(alpha=0.3, linestyle='--', linewidth=0.8)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # 3. CPU usage comparison
        ax3 = axes[1, 0]
        direct_cpu = [r.cpu_percent for r in direct_results]
        pydac_cpu = [r.cpu_percent for r in pydac_results]

        bp = ax3.boxplot([direct_cpu, pydac_cpu], labels=['Direct', 'PyDAC'],
                         patch_artist=True, widths=0.6,
                         boxprops=dict(facecolor=self.colors['light_blue'], alpha=0.7, 
                                       linewidth=1.2, edgecolor='black'),
                         medianprops=dict(color=self.colors['danger'], linewidth=2),
                         whiskerprops=dict(linewidth=1.2, color='black'),
                         capprops=dict(linewidth=1.2, color='black'),
                         flierprops=dict(marker='o', markersize=5, alpha=0.5))
        
        # Set second box color
        bp['boxes'][1].set_facecolor(self.colors['light_green'])
        
        ax3.set_ylabel('CPU Usage (%)', fontsize=11, fontweight='bold')
        ax3.set_title('(c) CPU Usage Distribution', fontsize=12, fontweight='bold', pad=10)
        ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # 4. Memory usage comparison
        ax4 = axes[1, 1]
        direct_memory = [r.memory_mb for r in direct_results]
        pydac_memory = [r.memory_mb for r in pydac_results]

        bp = ax4.boxplot([direct_memory, pydac_memory], labels=['Direct', 'PyDAC'],
                         patch_artist=True, widths=0.6,
                         boxprops=dict(facecolor=self.colors['light_blue'], alpha=0.7, 
                                       linewidth=1.2, edgecolor='black'),
                         medianprops=dict(color=self.colors['danger'], linewidth=2),
                         whiskerprops=dict(linewidth=1.2, color='black'),
                         capprops=dict(linewidth=1.2, color='black'),
                         flierprops=dict(marker='o', markersize=5, alpha=0.5))
        
        # Set second box color
        bp['boxes'][1].set_facecolor(self.colors['light_green'])
        
        ax4.set_ylabel('Memory Usage (MB)', fontsize=11, fontweight='bold')
        ax4.set_title('(d) Memory Usage Distribution', fontsize=12, fontweight='bold', pad=10)
        ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"benchmark_{operation}.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', pad_inches=0.1)

        if show:
            plt.show()
        else:
            plt.close()

        return str(save_path)

    def plot_multiple_comparisons(
        self,
        comparisons: List[ComparisonResult],
        save_path: Optional[str] = None,
        show: bool = False
    ) -> str:
        """
        Plot multiple comparison results with academic styling

        Args:
            comparisons: List of ComparisonResult objects
            save_path: Path to save plot (optional)
            show: Whether to display plot

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle('Multiple Performance Comparisons', 
                     fontsize=14, fontweight='bold', y=0.995)

        operations = [c.operation for c in comparisons]
        x_pos = np.arange(len(operations))
        width = 0.35

        # 1. Duration comparison
        ax1 = axes[0, 0]
        direct_avgs = [c.direct_avg for c in comparisons]
        pydac_avgs = [c.pydac_avg for c in comparisons]

        bars1 = ax1.bar(x_pos - width/2, direct_avgs, width, label='Direct', 
                        color=self.colors['primary'], alpha=0.85, edgecolor='black', linewidth=1.2)
        bars2 = ax1.bar(x_pos + width/2, pydac_avgs, width, label='PyDAC', 
                        color=self.colors['secondary'], alpha=0.85, edgecolor='black', linewidth=1.2)
        ax1.set_xlabel('Operation', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Average Duration (s)', fontsize=11, fontweight='bold')
        ax1.set_title('(a) Duration Comparison', fontsize=12, fontweight='bold', pad=10)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(operations, rotation=45, ha='right', fontsize=10)
        ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # 2. Overhead percentage
        ax2 = axes[0, 1]
        overheads = [c.overhead for c in comparisons]
        bar_colors = [self.colors['success'] if o < 10 else 
                      self.colors['warning'] if o < 50 else 
                      self.colors['danger'] for o in overheads]
        bars = ax2.bar(x_pos, overheads, color=bar_colors, alpha=0.85, 
                       edgecolor='black', linewidth=1.2, width=0.6)
        ax2.set_xlabel('Operation', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Overhead (%)', fontsize=11, fontweight='bold')
        ax2.set_title('(b) Overhead Percentage', fontsize=12, fontweight='bold', pad=10)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(operations, rotation=45, ha='right', fontsize=10)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.7)
        ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Add value labels
        for bar, overhead in zip(bars, overheads):
            height = bar.get_height()
            y_pos = height + (abs(height) * 0.05) if height != 0 else 0.5
            ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                     f'{overhead:.1f}%',
                     ha='center', va='bottom' if height > 0 else 'top',
                     fontsize=9, fontweight='bold')

        # 3. Speedup ratio
        ax3 = axes[1, 0]
        speedups = [c.speedup for c in comparisons]
        bar_colors = [self.colors['success'] if s < 1.1 else 
                      self.colors['warning'] if s < 1.5 else 
                      self.colors['danger'] for s in speedups]
        bars = ax3.bar(x_pos, speedups, color=bar_colors, alpha=0.85, 
                       edgecolor='black', linewidth=1.2, width=0.6)
        ax3.set_xlabel('Operation', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Speedup Ratio', fontsize=11, fontweight='bold')
        ax3.set_title('(c) Speedup Ratio (PyDAC / Direct)', fontsize=12, fontweight='bold', pad=10)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(operations, rotation=45, ha='right', fontsize=10)
        ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, 
                    alpha=0.7, label='Baseline (1.0)')
        ax3.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # Add value labels
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            y_pos = height + (height * 0.05) if height > 1 else height - (height * 0.05)
            ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                     f'{speedup:.2f}×',
                     ha='center', va='bottom' if height > 1 else 'top',
                     fontsize=9, fontweight='bold')

        # 4. Resource usage comparison
        ax4 = axes[1, 1]
        cpu_diffs = [c.cpu_diff for c in comparisons]
        memory_diffs = [c.memory_diff for c in comparisons]

        bars1 = ax4.bar(x_pos - width/2, cpu_diffs, width, label='CPU Δ (%)', 
                        color=self.colors['danger'], alpha=0.85, edgecolor='black', linewidth=1.2)
        bars2 = ax4.bar(x_pos + width/2, memory_diffs, width, label='Memory Δ (%)', 
                        color=self.colors['secondary'], alpha=0.85, edgecolor='black', linewidth=1.2)
        ax4.set_xlabel('Operation', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Difference (%)', fontsize=11, fontweight='bold')
        ax4.set_title('(d) Resource Usage Difference', fontsize=12, fontweight='bold', pad=10)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(operations, rotation=45, ha='right', fontsize=10)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.7)
        ax4.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        # Save plot
        if save_path is None:
            save_path = self.output_dir / "multiple_comparisons.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', pad_inches=0.1)

        if show:
            plt.show()
        else:
            plt.close()

        return str(save_path)

