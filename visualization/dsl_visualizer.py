"""
DSL Test Result Visualization Module

This is the most important test visualization module in the project, providing comprehensive performance analysis for DSL tests
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


class DSLVisualizer:
    """DSL test result visualizer - Core test visualization for the project"""
    
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
            'primary': '#2E86AB',      # Blue - USM mode
            'secondary': '#A23B72',     # Purple - Buffer mode
            'success': '#06A77D',      # Green - Success
            'warning': '#F18F01',      # Orange - Warning
            'danger': '#C73E1D',       # Red - Failure/Error
            'neutral': '#6C757D',      # Gray
            'info': '#17A2B8',        # Cyan - Info
        }
    
    def load_data(self, json_path: str) -> Dict[str, Any]:
        """Load JSON test report"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def plot_success_overview(self, data: Dict[str, Any]) -> str:
        """
        Generate test success rate overview chart (dsl_test_success_overview.png)
        Contains multiple subplots: overall statistics, stage success rates, mode comparison
        """
        summary = data.get('summary', {})
        results = data.get('results', [])
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall success rate (pie chart)
        ax1 = fig.add_subplot(gs[0, 0])
        total = data.get('total_tests', 0)
        overall_success = summary.get('overall_success', 0)
        overall_fail = total - overall_success
        
        if total > 0:
            sizes = [overall_success, overall_fail]
            colors_pie = [self.colors['success'], self.colors['danger']]
            labels = [f'Success\n{overall_success}', f'Failure\n{overall_fail}']
            explode = (0.05, 0) if overall_success > 0 else (0, 0.05)
            
            wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, 
                                               colors=colors_pie, autopct='%1.1f%%',
                                               startangle=90, textprops={'fontweight': 'bold'})
            ax1.set_title('Overall Success Rate', fontweight='bold', pad=10)
        
        # 2. Stage success rates (stacked bar chart)
        ax2 = fig.add_subplot(gs[0, 1])
        translate_success = summary.get('translate_success', 0)
        compile_success = summary.get('compile_success', 0)
        run_success = summary.get('run_success', 0)
        
        stages = ['Translation', 'Compilation', 'Execution']
        success_counts = [translate_success, compile_success, run_success]
        fail_counts = [total - s for s in success_counts]
        
        x = np.arange(len(stages))
        width = 0.6
        
        bars1 = ax2.bar(x, success_counts, width, label='Success', 
                       color=self.colors['success'], alpha=0.8, edgecolor='black')
        bars2 = ax2.bar(x, fail_counts, width, bottom=success_counts, label='Failure',
                       color=self.colors['danger'], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for i, (bar1, bar2, total_val) in enumerate(zip(bars1, bars2, success_counts)):
            if total_val > 0:
                ax2.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height()/2,
                        f'{total_val}', ha='center', va='center', fontweight='bold', color='white')
            if fail_counts[i] > 0:
                ax2.text(bar2.get_x() + bar2.get_width()/2., 
                        bar2.get_y() + bar2.get_height()/2,
                        f'{fail_counts[i]}', ha='center', va='center', 
                        fontweight='bold', color='white')
        
        ax2.set_ylabel('Test Count', fontweight='bold')
        ax2.set_title('Stage Success Rates', fontweight='bold', pad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(stages)
        ax2.legend(loc='upper right', frameon=True)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 3. USM vs Buffer mode comparison (grouped bar chart)
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Group statistics by mode
        usm_results = [r for r in results if r.get('mode') == 'usm']
        buffer_results = [r for r in results if r.get('mode') == 'buffer']
        
        usm_success = sum(1 for r in usm_results if r.get('overall_success', False))
        buffer_success = sum(1 for r in buffer_results if r.get('overall_success', False))
        
        modes = ['USM', 'Buffer']
        success_rates = [usm_success/len(usm_results)*100 if usm_results else 0,
                         buffer_success/len(buffer_results)*100 if buffer_results else 0]
        
        bars = ax3.bar(modes, success_rates, color=[self.colors['primary'], self.colors['secondary']],
                      alpha=0.8, edgecolor='black', linewidth=1.2, width=0.6)
        
        # Add value labels
        for bar, rate, count, total_count in zip(bars, success_rates, 
                                                [usm_success, buffer_success],
                                                [len(usm_results), len(buffer_results)]):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%\n({count}/{total_count})', ha='center', va='bottom',
                    fontweight='bold')
        
        ax3.set_ylabel('Success Rate (%)', fontweight='bold')
        ax3.set_title('USM vs Buffer Mode Comparison', fontweight='bold', pad=10)
        ax3.set_ylim([0, 105])
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 4. Test case success rates (horizontal bar chart)
        ax4 = fig.add_subplot(gs[1, :2])
        
        # Group by test name
        test_stats = defaultdict(lambda: {'usm': {'success': 0, 'total': 0},
                                         'buffer': {'success': 0, 'total': 0}})
        
        for result in results:
            name = result.get('name', 'Unknown')
            mode = result.get('mode', 'unknown')
            success = result.get('overall_success', False)
            
            if mode in ['usm', 'buffer']:
                test_stats[name][mode]['total'] += 1
                if success:
                    test_stats[name][mode]['success'] += 1
        
        # Calculate total success rate for each test
        test_names = sorted(test_stats.keys())
        success_rates = []
        for name in test_names:
            stats = test_stats[name]
            total_success = stats['usm']['success'] + stats['buffer']['success']
            total_count = stats['usm']['total'] + stats['buffer']['total']
            rate = total_success / total_count * 100 if total_count > 0 else 0
            success_rates.append(rate)
        
        # Sort by success rate
        sorted_data = sorted(zip(test_names, success_rates), key=lambda x: x[1])
        test_names_sorted, rates_sorted = zip(*sorted_data) if sorted_data else ([], [])
        
        if test_names_sorted:
            colors_bar = [self.colors['success'] if r == 100 else 
                         self.colors['warning'] if r >= 50 else 
                         self.colors['danger'] for r in rates_sorted]
            
            bars = ax4.barh(range(len(test_names_sorted)), rates_sorted, 
                          color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.2)
            
            # Add value labels
            for i, (bar, rate) in enumerate(zip(bars, rates_sorted)):
                ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{rate:.1f}%', ha='left', va='center', fontweight='bold')
            
            ax4.set_yticks(range(len(test_names_sorted)))
            ax4.set_yticklabels(test_names_sorted)
            ax4.set_xlabel('Success Rate (%)', fontweight='bold')
            ax4.set_title('Test Case Success Rates', fontweight='bold', pad=10)
            ax4.set_xlim([0, 105])
            ax4.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 5. Stage time comparison (box plot)
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Extract stage times
        translate_times = []
        compile_times = []
        run_times = []
        
        for result in results:
            translate = result.get('translate', {})
            compile_info = result.get('compile', {})
            run_info = result.get('run', {})
            
            if translate.get('success') and translate.get('duration'):
                translate_times.append(translate['duration'])
            if compile_info.get('success') and compile_info.get('duration'):
                compile_times.append(compile_info['duration'])
            if run_info.get('success') and run_info.get('duration'):
                run_times.append(run_info['duration'])
        
        if translate_times or compile_times or run_times:
            data_to_plot = []
            labels_to_plot = []
            
            if translate_times:
                data_to_plot.append(translate_times)
                labels_to_plot.append('Translation')
            if compile_times:
                data_to_plot.append(compile_times)
                labels_to_plot.append('Compilation')
            if run_times:
                data_to_plot.append(run_times)
                labels_to_plot.append('Execution')
            
            bp = ax5.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                            boxprops=dict(facecolor=self.colors['primary'], alpha=0.7, 
                                        linewidth=1.2, edgecolor='black'),
                            medianprops=dict(color=self.colors['danger'], linewidth=2),
                            whiskerprops=dict(linewidth=1.2, color='black'),
                            capprops=dict(linewidth=1.2, color='black'))
            
            ax5.set_ylabel('Time (seconds)', fontweight='bold')
            ax5.set_title('Stage Time Distribution', fontweight='bold', pad=10)
            ax5.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.suptitle('DSL Test Success Rate Overview', fontsize=16, fontweight='bold', y=0.98)
        
        # Save image
        output_path = self.output_dir / "dsl_test_success_overview.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def plot_mode_comparison(self, data: Dict[str, Any]) -> str:
        """
        Generate USM vs Buffer mode detailed comparison chart (dsl_test_mode_comparison.png)
        Contains performance comparison, success rate comparison, time distribution, etc.
        """
        results = data.get('results', [])
        
        # Separate USM and Buffer results
        usm_results = [r for r in results if r.get('mode') == 'usm']
        buffer_results = [r for r in results if r.get('mode') == 'buffer']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Execution time comparison (scatter plot)
        ax1 = axes[0, 0]
        
        # Match USM and Buffer results by test name
        usm_by_name = {r.get('name'): r for r in usm_results}
        buffer_by_name = {r.get('name'): r for r in buffer_results}
        
        common_tests = set(usm_by_name.keys()) & set(buffer_by_name.keys())
        
        usm_times = []
        buffer_times = []
        test_names = []
        
        for name in sorted(common_tests):
            usm_run = usm_by_name[name].get('run', {})
            buffer_run = buffer_by_name[name].get('run', {})
            
            if usm_run.get('success') and usm_run.get('duration') and \
               buffer_run.get('success') and buffer_run.get('duration'):
                usm_times.append(usm_run['duration'])
                buffer_times.append(buffer_run['duration'])
                test_names.append(name)
        
        if usm_times and buffer_times:
            ax1.scatter(usm_times, buffer_times, alpha=0.6, s=100,
                       color=self.colors['primary'], edgecolors='black', linewidth=1)
            
            # Add test name labels
            for name, usm_t, buf_t in zip(test_names, usm_times, buffer_times):
                ax1.annotate(name, (usm_t, buf_t), xytext=(5, 5),
                           textcoords='offset points', fontsize=8, alpha=0.7)
            
            # Add performance equality line
            max_time = max(max(usm_times), max(buffer_times))
            ax1.plot([0, max_time], [0, max_time], 'r--', linewidth=2,
                    label='Performance Equality Line', alpha=0.7)
            
            ax1.set_xlabel('USM Execution Time (seconds)', fontweight='bold')
            ax1.set_ylabel('Buffer Execution Time (seconds)', fontweight='bold')
            ax1.set_title('USM vs Buffer Execution Time Comparison', fontweight='bold', pad=10)
            ax1.legend(loc='upper left', frameon=True)
            ax1.grid(alpha=0.3, linestyle='--')
        
        # 2. Stage time comparison (grouped bar chart)
        ax2 = axes[0, 1]
        
        # Calculate average time
        usm_translate_avg = np.mean([r.get('translate', {}).get('duration', 0) 
                                     for r in usm_results 
                                     if r.get('translate', {}).get('success')]) or 0
        buffer_translate_avg = np.mean([r.get('translate', {}).get('duration', 0)
                                       for r in buffer_results
                                       if r.get('translate', {}).get('success')]) or 0
        
        usm_compile_avg = np.mean([r.get('compile', {}).get('duration', 0)
                                  for r in usm_results
                                  if r.get('compile', {}).get('success')]) or 0
        buffer_compile_avg = np.mean([r.get('compile', {}).get('duration', 0)
                                     for r in buffer_results
                                     if r.get('compile', {}).get('success')]) or 0
        
        usm_run_avg = np.mean([r.get('run', {}).get('duration', 0)
                             for r in usm_results
                             if r.get('run', {}).get('success')]) or 0
        buffer_run_avg = np.mean([r.get('run', {}).get('duration', 0)
                                 for r in buffer_results
                                 if r.get('run', {}).get('success')]) or 0
        
        stages = ['Translation', 'Compilation', 'Execution']
        x = np.arange(len(stages))
        width = 0.35
        
        usm_values = [usm_translate_avg, usm_compile_avg, usm_run_avg]
        buffer_values = [buffer_translate_avg, buffer_compile_avg, buffer_run_avg]
        
        bars1 = ax2.bar(x - width/2, usm_values, width, label='USM',
                       color=self.colors['primary'], alpha=0.8, edgecolor='black')
        bars2 = ax2.bar(x + width/2, buffer_values, width, label='Buffer',
                       color=self.colors['secondary'], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bars, values in [(bars1, usm_values), (bars2, buffer_values)]:
            for bar, val in zip(bars, values):
                if val > 0:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + max(max(usm_values), max(buffer_values)) * 0.01,
                            f'{val:.3f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2.set_ylabel('Average Time (seconds)', fontweight='bold')
        ax2.set_title('Average Stage Time Comparison', fontweight='bold', pad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(stages)
        ax2.legend(loc='upper left', frameon=True)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 3. Success rate comparison (stacked bar chart)
        ax3 = axes[1, 0]
        
        # Count success rates by stage
        usm_translate_success = sum(1 for r in usm_results 
                                   if r.get('translate', {}).get('success', False))
        usm_compile_success = sum(1 for r in usm_results
                                 if r.get('compile', {}).get('success', False))
        usm_run_success = sum(1 for r in usm_results
                            if r.get('run', {}).get('success', False))
        
        buffer_translate_success = sum(1 for r in buffer_results
                                      if r.get('translate', {}).get('success', False))
        buffer_compile_success = sum(1 for r in buffer_results
                                    if r.get('compile', {}).get('success', False))
        buffer_run_success = sum(1 for r in buffer_results
                               if r.get('run', {}).get('success', False))
        
        total_usm = len(usm_results)
        total_buffer = len(buffer_results)
        
        usm_rates = [usm_translate_success/total_usm*100 if total_usm > 0 else 0,
                    usm_compile_success/total_usm*100 if total_usm > 0 else 0,
                    usm_run_success/total_usm*100 if total_usm > 0 else 0]
        buffer_rates = [buffer_translate_success/total_buffer*100 if total_buffer > 0 else 0,
                       buffer_compile_success/total_buffer*100 if total_buffer > 0 else 0,
                       buffer_run_success/total_buffer*100 if total_buffer > 0 else 0]
        
        x = np.arange(len(stages))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, usm_rates, width, label='USM',
                       color=self.colors['primary'], alpha=0.8, edgecolor='black')
        bars2 = ax3.bar(x + width/2, buffer_rates, width, label='Buffer',
                       color=self.colors['secondary'], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bars, values in [(bars1, usm_rates), (bars2, buffer_rates)]:
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax3.set_ylabel('Success Rate (%)', fontweight='bold')
        ax3.set_title('Stage Success Rate Comparison', fontweight='bold', pad=10)
        ax3.set_xticks(x)
        ax3.set_xticklabels(stages)
        ax3.set_ylim([0, 105])
        ax3.legend(loc='upper right', frameon=True)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 4. Performance improvement/degradation analysis (bar chart)
        ax4 = axes[1, 1]
        
        if usm_times and buffer_times:
            # Calculate performance ratio (Buffer/USM)
            performance_ratios = [buf/usm if usm > 0 else 0 
                                 for usm, buf in zip(usm_times, buffer_times)]
            
            # Calculate improvement percentage
            improvements = [(buf - usm) / usm * 100 if usm > 0 else 0
                          for usm, buf in zip(usm_times, buffer_times)]
            
            colors_improve = [self.colors['success'] if imp < 0 else 
                             self.colors['warning'] if imp < 20 else 
                             self.colors['danger'] for imp in improvements]
            
            bars = ax4.barh(range(len(test_names)), improvements, 
                          color=colors_improve, alpha=0.8, edgecolor='black', linewidth=1.2)
            
            # Add value labels
            for i, (bar, imp, ratio) in enumerate(zip(bars, improvements, performance_ratios)):
                ax4.text(bar.get_width() + (1 if imp > 0 else -1), 
                        bar.get_y() + bar.get_height()/2,
                        f'{imp:+.1f}% ({ratio:.2f}×)', ha='left' if imp > 0 else 'right',
                        va='center', fontsize=9, fontweight='bold')
            
            ax4.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax4.set_yticks(range(len(test_names)))
            ax4.set_yticklabels(test_names)
            ax4.set_xlabel('Performance Change (%)', fontweight='bold')
            ax4.set_title('Buffer Performance Change Relative to USM', fontweight='bold', pad=10)
            ax4.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.suptitle('USM vs Buffer Mode Detailed Comparison', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save image
        output_path = self.output_dir / "dsl_test_mode_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def plot_performance_analysis(self, data: Dict[str, Any]) -> str:
        """
        Generate performance analysis chart (dsl_test_performance_analysis.png)
        Contains stage time breakdown, performance trends, time distribution, etc.
        """
        results = data.get('results', [])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Stage time breakdown (stacked bar chart)
        ax1 = axes[0, 0]
        
        # Organize data by test name and mode
        test_data = defaultdict(lambda: {'usm': {}, 'buffer': {}})
        
        for result in results:
            name = result.get('name', 'Unknown')
            mode = result.get('mode', 'unknown')
            if mode in ['usm', 'buffer']:
                translate = result.get('translate', {})
                compile_info = result.get('compile', {})
                run_info = result.get('run', {})
                
                test_data[name][mode] = {
                    'translate': translate.get('duration', 0) if translate.get('success') else 0,
                    'compile': compile_info.get('duration', 0) if compile_info.get('success') else 0,
                    'run': run_info.get('duration', 0) if run_info.get('success') else 0,
                }
        
        # Select tests with data (prefer USM)
        test_names = []
        translate_times = []
        compile_times = []
        run_times = []
        
        for name in sorted(test_data.keys()):
            mode_data = test_data[name].get('usm') or test_data[name].get('buffer')
            if mode_data and (mode_data['translate'] > 0 or mode_data['compile'] > 0 or mode_data['run'] > 0):
                test_names.append(name)
                translate_times.append(mode_data['translate'])
                compile_times.append(mode_data['compile'])
                run_times.append(mode_data['run'])
        
        if test_names:
            x = np.arange(len(test_names))
            width = 0.6
            
            bars1 = ax1.bar(x, translate_times, width, label='Translation Time',
                           color=self.colors['primary'], alpha=0.8, edgecolor='black')
            bars2 = ax1.bar(x, compile_times, width, bottom=translate_times, label='Compilation Time',
                           color=self.colors['warning'], alpha=0.8, edgecolor='black')
            bars3 = ax1.bar(x, run_times, width,
                           bottom=np.array(translate_times) + np.array(compile_times),
                           label='Execution Time', color=self.colors['success'], alpha=0.8, edgecolor='black')
            
            ax1.set_xlabel('Test Case', fontweight='bold')
            ax1.set_ylabel('Time (seconds)', fontweight='bold')
            ax1.set_title('Stage Time Breakdown', fontweight='bold', pad=10)
            ax1.set_xticks(x)
            ax1.set_xticklabels(test_names, rotation=45, ha='right')
            ax1.legend(loc='upper left', frameon=True)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 2. Execution time distribution (histogram)
        ax2 = axes[0, 1]
        
        run_times_all = [r.get('run', {}).get('duration', 0)
                         for r in results
                         if r.get('run', {}).get('success') and r.get('run', {}).get('duration')]
        
        if run_times_all:
            ax2.hist(run_times_all, bins=min(20, len(run_times_all)),
                    color=self.colors['secondary'], alpha=0.7, edgecolor='black', linewidth=1.2)
            
            # Add statistical lines
            mean_time = np.mean(run_times_all)
            median_time = np.median(run_times_all)
            ax2.axvline(mean_time, color=self.colors['danger'], linestyle='--',
                       linewidth=2, label=f'Mean: {mean_time:.3f}s')
            ax2.axvline(median_time, color=self.colors['warning'], linestyle='--',
                       linewidth=2, label=f'Median: {median_time:.3f}s')
            
            ax2.set_xlabel('Execution Time (seconds)', fontweight='bold')
            ax2.set_ylabel('Frequency', fontweight='bold')
            ax2.set_title('Execution Time Distribution', fontweight='bold', pad=10)
            ax2.legend(loc='upper right', frameon=True)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 3. Performance trend (by test order)
        ax3 = axes[1, 0]
        
        # Sort by timestamp or order
        sorted_results = sorted(results, key=lambda x: x.get('timestamp', ''))
        
        usm_times_trend = []
        buffer_times_trend = []
        indices = []
        
        for i, result in enumerate(sorted_results):
            mode = result.get('mode', '')
            run_info = result.get('run', {})
            if run_info.get('success') and run_info.get('duration'):
                if mode == 'usm':
                    usm_times_trend.append((i, run_info['duration']))
                elif mode == 'buffer':
                    buffer_times_trend.append((i, run_info['duration']))
        
        if usm_times_trend:
            idxs, times = zip(*usm_times_trend)
            ax3.plot(idxs, times, 'o-', label='USM', color=self.colors['primary'],
                    linewidth=2, markersize=8, markerfacecolor='white',
                    markeredgewidth=2, markeredgecolor=self.colors['primary'])
        
        if buffer_times_trend:
            idxs, times = zip(*buffer_times_trend)
            ax3.plot(idxs, times, 's-', label='Buffer', color=self.colors['secondary'],
                    linewidth=2, markersize=8, markerfacecolor='white',
                    markeredgewidth=2, markeredgecolor=self.colors['secondary'])
        
        ax3.set_xlabel('Test Sequence', fontweight='bold')
        ax3.set_ylabel('Execution Time (seconds)', fontweight='bold')
        ax3.set_title('Performance Trend', fontweight='bold', pad=10)
        ax3.legend(loc='best', frameon=True)
        ax3.grid(alpha=0.3, linestyle='--')
        
        # 4. Total time comparison (USM vs Buffer)
        ax4 = axes[1, 1]
        
        # Calculate total time for each test (translation + compilation + execution)
        test_total_times = defaultdict(lambda: {'usm': 0, 'buffer': 0})
        
        for result in results:
            name = result.get('name', 'Unknown')
            mode = result.get('mode', '')
            if mode in ['usm', 'buffer']:
                translate = result.get('translate', {})
                compile_info = result.get('compile', {})
                run_info = result.get('run', {})
                
                total = (translate.get('duration', 0) if translate.get('success') else 0) + \
                       (compile_info.get('duration', 0) if compile_info.get('success') else 0) + \
                       (run_info.get('duration', 0) if run_info.get('success') else 0)
                
                test_total_times[name][mode] = total
        
        # Find tests with both USM and Buffer data
        common_tests = [name for name in test_total_times.keys()
                       if test_total_times[name]['usm'] > 0 and test_total_times[name]['buffer'] > 0]
        
        if common_tests:
            usm_totals = [test_total_times[name]['usm'] for name in sorted(common_tests)]
            buffer_totals = [test_total_times[name]['buffer'] for name in sorted(common_tests)]
            
            x = np.arange(len(common_tests))
            width = 0.35
            
            bars1 = ax4.bar(x - width/2, usm_totals, width, label='USM',
                           color=self.colors['primary'], alpha=0.8, edgecolor='black')
            bars2 = ax4.bar(x + width/2, buffer_totals, width, label='Buffer',
                           color=self.colors['secondary'], alpha=0.8, edgecolor='black')
            
            ax4.set_xlabel('Test Case', fontweight='bold')
            ax4.set_ylabel('Total Time (seconds)', fontweight='bold')
            ax4.set_title('Total Time Comparison (Translation + Compilation + Execution)', fontweight='bold', pad=10)
            ax4.set_xticks(x)
            ax4.set_xticklabels(sorted(common_tests), rotation=45, ha='right')
            ax4.legend(loc='upper left', frameon=True)
            ax4.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.suptitle('DSL Test Performance Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save image
        output_path = self.output_dir / "dsl_test_performance_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def plot_comparison_analysis(self, data: Dict[str, Any]) -> str:
        """
        Generate comparison analysis chart with original tests (dsl_test_comparison_analysis.png)
        Contains output consistency, interface validation, etc.
        """
        results = data.get('results', [])
        summary = data.get('summary', {})
        comparison_stats = summary.get('comparison_stats', {})
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Output consistency statistics (pie chart)
        ax1 = axes[0, 0]
        
        if comparison_stats:
            output_identical = comparison_stats.get('output_identical', 0)
            output_total = comparison_stats.get('output_total', 0)
            output_diff = output_total - output_identical
            
            if output_total > 0:
                sizes = [output_identical, output_diff]
                colors_pie = [self.colors['success'], self.colors['danger']]
                labels = [f'Consistent\n{output_identical}', f'Inconsistent\n{output_diff}']
                explode = (0.05, 0) if output_identical > 0 else (0, 0.05)
                
                wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels,
                                                   colors=colors_pie, autopct='%1.1f%%',
                                                   startangle=90, textprops={'fontweight': 'bold'})
                ax1.set_title('Output Consistency Statistics', fontweight='bold', pad=10)
            else:
                ax1.text(0.5, 0.5, 'No comparison data', ha='center', va='center',
                        fontsize=14, transform=ax1.transAxes)
                ax1.set_title('Output Consistency Statistics', fontweight='bold', pad=10)
        else:
            ax1.text(0.5, 0.5, 'No comparison data', ha='center', va='center',
                    fontsize=14, transform=ax1.transAxes)
            ax1.set_title('Output Consistency Statistics', fontweight='bold', pad=10)
        
        # 2. Interface validation status (bar chart)
        ax2 = axes[0, 1]
        
        if comparison_stats:
            interface_valid = comparison_stats.get('interface_valid', False)
            
            status = ['Interface Validation']
            values = [1 if interface_valid else 0]
            colors_bar = [self.colors['success'] if interface_valid else self.colors['danger']]
            labels_bar = ['Pass' if interface_valid else 'Fail']
            
            bars = ax2.bar(status, values, color=colors_bar, alpha=0.8,
                          edgecolor='black', linewidth=1.2, width=0.6)
            
            # Add labels
            for bar, label in zip(bars, labels_bar):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                        label, ha='center', va='center', fontweight='bold',
                        fontsize=14, color='white')
            
            ax2.set_ylabel('Status', fontweight='bold')
            ax2.set_title('Interface Validation Status', fontweight='bold', pad=10)
            ax2.set_ylim([0, 1.2])
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['Fail', 'Pass'])
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
        else:
            ax2.text(0.5, 0.5, 'No validation data', ha='center', va='center',
                    fontsize=14, transform=ax2.transAxes)
            ax2.set_title('Interface Validation Status', fontweight='bold', pad=10)
        
        # 3. Test case comparison results (horizontal bar chart)
        ax3 = axes[1, :]
        
        # Extract tests with comparison data
        comparison_data = []
        for result in results:
            comparison = result.get('comparison')
            if comparison:
                output_comp = comparison.get('output_comparison', {})
                if output_comp:
                    name = result.get('name', 'Unknown')
                    identical = output_comp.get('identical', False)
                    comparison_data.append((name, identical))
        
        if comparison_data:
            test_names, identical_flags = zip(*comparison_data)
            
            # Sort by consistency
            sorted_data = sorted(zip(test_names, identical_flags), 
                               key=lambda x: (not x[1], x[0]))
            test_names_sorted, flags_sorted = zip(*sorted_data)
            
            colors_bar = [self.colors['success'] if flag else self.colors['danger']
                         for flag in flags_sorted]
            values = [1 if flag else 0 for flag in flags_sorted]
            
            bars = ax3.barh(range(len(test_names_sorted)), values,
                          color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.2)
            
            # Add labels
            for i, (bar, flag, name) in enumerate(zip(bars, flags_sorted, test_names_sorted)):
                label = 'Consistent' if flag else 'Inconsistent'
                ax3.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                        label, ha='center', va='center', fontweight='bold',
                        fontsize=10, color='white')
            
            ax3.set_yticks(range(len(test_names_sorted)))
            ax3.set_yticklabels(test_names_sorted)
            ax3.set_xlabel('Consistency Status', fontweight='bold')
            ax3.set_title('Test Case Output Consistency', fontweight='bold', pad=10)
            ax3.set_xlim([0, 1.2])
            ax3.set_xticks([0, 1])
            ax3.set_xticklabels(['Inconsistent', 'Consistent'])
            ax3.grid(axis='x', alpha=0.3, linestyle='--')
        else:
            ax3.text(0.5, 0.5, 'No comparison data', ha='center', va='center',
                    fontsize=14, transform=ax3.transAxes)
            ax3.set_title('Test Case Output Consistency', fontweight='bold', pad=10)
        
        plt.suptitle('DSL Test vs Original Test Comparison Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save image
        output_path = self.output_dir / "dsl_test_comparison_analysis.png"
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
        
        print("Generating test success rate overview chart...")
        output_paths['success_overview'] = self.plot_success_overview(data)
        
        print("Generating USM vs Buffer mode comparison chart...")
        output_paths['mode_comparison'] = self.plot_mode_comparison(data)
        
        print("Generating performance analysis chart...")
        output_paths['performance_analysis'] = self.plot_performance_analysis(data)
        
        print("Generating comparison analysis chart...")
        output_paths['comparison_analysis'] = self.plot_comparison_analysis(data)
        
        print(f"\nAll visualization charts have been generated to: {self.output_dir}")
        
        return output_paths
