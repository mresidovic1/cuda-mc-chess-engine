#!/usr/bin/env python3

import re
import sys
import os
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import argparse


@dataclass
class FunctionSample:
    name: str
    samples: int
    source_file: str
    line: int
    percentage: float = 0.0


@dataclass
class PerformanceMetrics:
    depth: int = 0
    nodes: int = 0
    nps: int = 0
    time_ms: int = 0
    threads: int = 0
    best_move: str = ""
    expected_move: str = ""
    test_passed: bool = False


@dataclass
class MemoryInfo:
    physical_footprint_kb: float = 0
    peak_footprint_mb: float = 0
    nodes_malloced: int = 0
    malloc_size_kb: float = 0
    leaks_count: int = 0
    leaked_bytes: int = 0


class ProfileAnalyzer:
    def __init__(self, profile_dir: str):
        self.profile_dir = profile_dir
        self.time_profile_path = os.path.join(profile_dir, "time_profile.txt")
        self.memory_leaks_path = os.path.join(profile_dir, "memory_leaks.txt")
        self.summary_path = os.path.join(profile_dir, "summary_report.txt")
        self.function_samples: dict[str, FunctionSample] = {}
        self.total_samples = 0
        self.metrics = PerformanceMetrics()
        self.memory = MemoryInfo()
        self.threads_detected: set[str] = set()
        
    def analyze(self):
        print("=" * 80)
        print("                    CHESS ENGINE PROFILE ANALYSIS")
        print("=" * 80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Profile:   {os.path.basename(self.profile_dir)}")
        print()
        
        self._parse_time_profile()
        self._parse_memory_leaks()
        self._calculate_percentages()
        
        self._print_glossary()
        self._print_executive_summary()
        self._print_cpu_intensive_functions()
        self._print_category_breakdown()
        self._print_parallelization_analysis()
        self._print_memory_analysis()
        self._print_performance_metrics()
        self._print_potential_improvements()
        
    def _parse_time_profile(self):
        if not os.path.exists(self.time_profile_path):
            print(f"Warning: Time profile not found at {self.time_profile_path}")
            return
            
        func_pattern = re.compile(
            r'^\s*\+?\s*[!:\s|]*\s*(\d+)\s+(.+?)\s+\(in\s+([^)]+)\)\s+\+\s*[\d,]+\s+\[[^\]]+\](?:\s+(\S+):(\d+))?',
            re.MULTILINE
        )
        simple_pattern = re.compile(r'(\d+)\s+(\w[\w:<>(),\s&*]+?)\s+\(in\s+(\S+)\)')
        thread_pattern = re.compile(r'Thread_(\d+)')
        
        with open(self.time_profile_path, 'r') as f:
            content = f.read()
            
        for match in thread_pattern.finditer(content):
            self.threads_detected.add(match.group(1))
            
        for line in content.split('\n'):
            if 'Thread_' in line and 'DispatchQueue' in line:
                continue
                
            match = re.search(r'(\d+)\s+(\w[\w:<>(),\s&*]+)\s+\(in\s+([^)]+)\)', line)
            if match:
                samples = int(match.group(1))
                func_name = match.group(2).strip()
                binary = match.group(3).strip()
                
                source_match = re.search(r'(\S+\.(?:cpp|hpp|h)):(\d+)', line)
                source_file = source_match.group(1) if source_match else ""
                line_num = int(source_match.group(2)) if source_match else 0
                
                if binary in ['libomp.dylib', 'libsystem_pthread.dylib', 'libsystem_kernel.dylib']:
                    continue
                    
                if func_name.startswith('???') or func_name.startswith('0x'):
                    continue
                
                key = f"{func_name}:{source_file}:{line_num}"
                
                if key not in self.function_samples:
                    self.function_samples[key] = FunctionSample(
                        name=func_name,
                        samples=samples,
                        source_file=source_file,
                        line=line_num
                    )
                else:
                    if samples > self.function_samples[key].samples:
                        self.function_samples[key].samples = samples
                        
        self.total_samples = sum(fs.samples for fs in self.function_samples.values())
        
    def _parse_memory_leaks(self):
        if not os.path.exists(self.memory_leaks_path):
            print(f"Warning: Memory leaks file not found at {self.memory_leaks_path}")
            return
            
        with open(self.memory_leaks_path, 'r') as f:
            content = f.read()
            
        depth_matches = re.findall(r'info depth (\d+) score cp (-?\d+) nodes (\d+) nps (\d+) time (\d+)', content)
        if depth_matches:
            last = depth_matches[-1]
            self.metrics.depth = int(last[0])
            self.metrics.nodes = int(last[2])
            self.metrics.nps = int(last[3])
            self.metrics.time_ms = int(last[4])
            
        best_move_match = re.search(r'Best Move: (\S+)', content)
        if best_move_match:
            self.metrics.best_move = best_move_match.group(1)
            
        expected_match = re.search(r'Expected: (\S+)', content)
        if expected_match:
            self.metrics.expected_move = expected_match.group(1)
            
        threads_match = re.search(r'Threads: (\d+)', content)
        if threads_match:
            self.metrics.threads = int(threads_match.group(1))
            
        self.metrics.test_passed = '[PASS]' in content
        
        footprint_match = re.search(r'Physical footprint:\s+([\d.]+)([KMG]?)', content)
        if footprint_match:
            value = float(footprint_match.group(1))
            unit = footprint_match.group(2)
            if unit == 'M':
                value *= 1024
            elif unit == 'G':
                value *= 1024 * 1024
            self.memory.physical_footprint_kb = value
            
        peak_match = re.search(r'Physical footprint \(peak\):\s+([\d.]+)([KMG]?)', content)
        if peak_match:
            value = float(peak_match.group(1))
            unit = peak_match.group(2)
            if unit == 'K':
                value /= 1024
            elif unit == 'G':
                value *= 1024
            self.memory.peak_footprint_mb = value
            
        malloc_match = re.search(r'Process \d+: (\d+) nodes malloced for (\d+) KB', content)
        if malloc_match:
            self.memory.nodes_malloced = int(malloc_match.group(1))
            self.memory.malloc_size_kb = float(malloc_match.group(2))
            
        leaks_match = re.search(r'(\d+) leaks for (\d+) total leaked bytes', content)
        if leaks_match:
            self.memory.leaks_count = int(leaks_match.group(1))
            self.memory.leaked_bytes = int(leaks_match.group(2))
            
    def _calculate_percentages(self):
        if self.total_samples == 0:
            return
        for key in self.function_samples:
            self.function_samples[key].percentage = (
                self.function_samples[key].samples / self.total_samples * 100
            )
            
    def _get_sorted_functions(self, limit: int = 20) -> list[FunctionSample]:
        return sorted(
            self.function_samples.values(),
            key=lambda x: x.samples,
            reverse=True
        )[:limit]
        
    def _categorize_function(self, func: FunctionSample) -> str:
        name = func.name.lower()
        
        if 'negamax' in name:
            return 'Search (negamax)'
        elif 'quiescence' in name:
            return 'Quiescence Search'
        elif 'evaluate' in name:
            return 'Evaluation'
        elif 'order_moves' in name:
            return 'Move Ordering'
        elif 'legalmoves' in name or 'movegen' in name or 'generatemoves' in name:
            return 'Move Generation'
        elif 'makemove' in name:
            return 'Make Move'
        elif 'unmakemove' in name:
            return 'Unmake Move'
        elif 'see' in name and 'seen' not in name:
            return 'Static Exchange Eval'
        elif 'transposition' in name or 'tt_' in name:
            return 'Transposition Table'
        elif 'killer' in name:
            return 'Killer Moves'
        elif 'sort' in name or 'introsort' in name:
            return 'Sorting'
        elif 'malloc' in name or 'free' in name or 'new' in name:
            return 'Memory Allocation'
        elif 'checkMask' in name or 'isAttacked' in name:
            return 'Attack Detection'
        elif 'find_best_move' in name:
            return 'Best Move Search'
        elif 'omp' in name or 'kmp' in name:
            return 'OpenMP Runtime'
        else:
            return 'Other'

    def _print_glossary(self):
        print("GLOSSARY")
        print("-" * 80)
        print("""
+----------------------+------------------------------------------------------+
| Term                 | Description                                          |
+----------------------+------------------------------------------------------+
| Samples              | Number of times the profiler observed the CPU        |
|                      | executing this function (1 sample = 1 millisecond)   |
+----------------------+------------------------------------------------------+
| NPS                  | Nodes Per Second - positions evaluated per second    |
+----------------------+------------------------------------------------------+
| Nodes                | Total chess positions evaluated during search        |
+----------------------+------------------------------------------------------+
| Depth                | Maximum search depth reached in the game tree        |
+----------------------+------------------------------------------------------+
| Parallel Efficiency  | Ratio of useful work to total work (higher = better) |
+----------------------+------------------------------------------------------+
| Physical Footprint   | Actual memory in use by the process                  |
+----------------------+------------------------------------------------------+
| Peak Footprint       | Maximum memory used during execution                 |
+----------------------+------------------------------------------------------+
| Memory Leaks         | Allocated memory not freed (should be 0)             |
+----------------------+------------------------------------------------------+
""")

    def _print_executive_summary(self):
        print("EXECUTIVE SUMMARY")
        print("-" * 80)
        
        if self.metrics.nps > 0:
            print(f"  Search Performance:  {self.metrics.nps:,} nodes/second")
            print(f"  Search Depth:        {self.metrics.depth} plies")
            print(f"  Total Nodes:         {self.metrics.nodes:,}")
            print(f"  Execution Time:      {self.metrics.time_ms / 1000:.2f} seconds")
            print(f"  Thread Count:        {self.metrics.threads}")
            
            search_samples = sum(
                f.samples for f in self.function_samples.values()
                if 'negamax' in f.name.lower() or 'quiescence' in f.name.lower()
            )
            if self.total_samples > 0:
                efficiency = search_samples / self.total_samples * 100
                print(f"  Search Efficiency:   {efficiency:.1f}% of CPU time in search")
            
            if self.memory.leaks_count == 0:
                print(f"  Memory Status:       No leaks detected")
            else:
                print(f"  Memory Status:       {self.memory.leaks_count} leaks detected")
                
            test_status = "PASS" if self.metrics.best_move == self.metrics.expected_move else "FAIL"
            print(f"  Test Result:         {test_status}")
        else:
            print("  No performance data available.")
            
        print()
            
    def _print_cpu_intensive_functions(self):
        print("MOST CPU-INTENSIVE FUNCTIONS")
        print("-" * 80)
        print(f"{'Rank':<5} {'Samples':<10} {'Time %':<8} {'Function':<40} {'Source Location'}")
        print("-" * 80)
        
        top_funcs = self._get_sorted_functions(15)
        for i, func in enumerate(top_funcs, 1):
            name = func.name[:38] + ".." if len(func.name) > 40 else func.name
            location = f"{func.source_file}:{func.line}" if func.source_file else "-"
            print(f"{i:<5} {func.samples:<10} {func.percentage:<7.2f}% {name:<40} {location}")
            
        print()
        
    def _print_category_breakdown(self):
        print("TIME BREAKDOWN BY CATEGORY")
        print("-" * 80)
        
        categories: dict[str, int] = defaultdict(int)
        for func in self.function_samples.values():
            cat = self._categorize_function(func)
            categories[cat] += func.samples
            
        sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        
        max_samples = max(categories.values()) if categories else 1
        bar_width = 40
        
        for cat, samples in sorted_cats:
            pct = samples / self.total_samples * 100 if self.total_samples > 0 else 0
            bar_len = int(samples / max_samples * bar_width)
            bar = "#" * bar_len + "-" * (bar_width - bar_len)
            print(f"{cat:<25} [{bar}] {pct:5.1f}% ({samples:,} samples)")
            
        print()
        
    def _print_parallelization_analysis(self):
        print("PARALLELIZATION ANALYSIS")
        print("-" * 80)
        
        num_threads = len(self.threads_detected)
        if num_threads == 0:
            num_threads = self.metrics.threads or 1
            
        print(f"Detected threads: {num_threads}")
        print(f"Configured threads: {self.metrics.threads or 'N/A'}")
        
        search_samples = sum(
            f.samples for f in self.function_samples.values()
            if 'negamax' in f.name.lower() or 'quiescence' in f.name.lower()
        )
        
        omp_overhead = sum(
            f.samples for f in self.function_samples.values()
            if 'kmp_' in f.name.lower() or 'omp' in f.name.lower()
        )
        
        memory_overhead = sum(
            f.samples for f in self.function_samples.values()
            if 'malloc' in f.name.lower() or 'free' in f.name.lower() or 'new' in f.name.lower()
        )
        
        if self.total_samples > 0:
            search_pct = search_samples / self.total_samples * 100
            omp_pct = omp_overhead / self.total_samples * 100
            mem_pct = memory_overhead / self.total_samples * 100
            
            print(f"\nTime distribution:")
            print(f"  - Search algorithms: {search_pct:.1f}%")
            print(f"  - OpenMP overhead: {omp_pct:.1f}%")
            print(f"  - Memory operations: {mem_pct:.1f}%")
            
            efficiency = search_pct / (search_pct + omp_pct + mem_pct) * 100 if (search_pct + omp_pct + mem_pct) > 0 else 0
            print(f"\n  Parallel efficiency score: {efficiency:.1f}%")
            
            if omp_pct > 5:
                print("  [!] High OpenMP overhead detected - consider reducing parallel region granularity")
            if mem_pct > 10:
                print("  [!] High memory allocation overhead - consider object pooling or stack allocation")
                
        print()
        
    def _print_memory_analysis(self):
        print("MEMORY ANALYSIS")
        print("-" * 80)
        
        print(f"Physical footprint: {self.memory.physical_footprint_kb:.1f} KB")
        print(f"Peak footprint: {self.memory.peak_footprint_mb:.1f} MB")
        print(f"Malloc nodes: {self.memory.nodes_malloced:,}")
        print(f"Malloc size: {self.memory.malloc_size_kb:.1f} KB")
        print(f"Memory leaks: {self.memory.leaks_count}")
        print(f"Leaked bytes: {self.memory.leaked_bytes}")
        
        if self.memory.leaks_count == 0:
            print("\n[OK] No memory leaks detected.")
        else:
            print(f"\n[WARNING] {self.memory.leaks_count} memory leaks detected!")
            
        if self.metrics.nodes > 0 and self.memory.peak_footprint_mb > 0:
            bytes_per_node = (self.memory.peak_footprint_mb * 1024 * 1024) / self.metrics.nodes
            print(f"\nMemory efficiency: {bytes_per_node:.4f} bytes/node")
            
        print()
        
    def _print_performance_metrics(self):
        print("PERFORMANCE METRICS")
        print("-" * 80)
        
        if self.metrics.depth > 0:
            print(f"Search depth reached: {self.metrics.depth}")
            print(f"Total nodes searched: {self.metrics.nodes:,}")
            print(f"Nodes per second: {self.metrics.nps:,}")
            print(f"Search time (includes profiling overhead): {self.metrics.time_ms / 1000:.2f} seconds")
            print(f"Threads used: {self.metrics.threads}")
            
            if self.metrics.best_move:
                print(f"\nBest move found: {self.metrics.best_move}")
                print(f"Expected move: {self.metrics.expected_move}")
                status = "PASS" if self.metrics.best_move == self.metrics.expected_move else "FAIL"
                print(f"Test result: {status}")
                
            if self.metrics.threads > 1:
                print(f"\nNPS per thread (avg): {self.metrics.nps // self.metrics.threads:,}")
                
        print()
        
    def _print_potential_improvements(self):
        print("POTENTIAL IMPROVEMENTS")
        print("-" * 80)
        
        recommendations = []
        
        move_ordering_samples = sum(
            f.samples for f in self.function_samples.values()
            if 'order_moves' in f.name.lower()
        )
        if self.total_samples > 0 and move_ordering_samples / self.total_samples > 0.1:
            recommendations.append(
                "Move ordering takes >10% of time. Consider:\n"
                "   - Using incremental move scoring\n"
                "   - Implementing lazy move generation\n"
                "   - Caching move scores in transposition table"
            )
            
        movegen_samples = sum(
            f.samples for f in self.function_samples.values()
            if 'legalmoves' in f.name.lower() or 'movegen' in f.name.lower()
        )
        if self.total_samples > 0 and movegen_samples / self.total_samples > 0.15:
            recommendations.append(
                "Move generation takes >15% of time. Consider:\n"
                "   - Lazy move generation (generate moves as needed)\n"
                "   - Better move ordering to cause earlier cutoffs"
            )
            
        eval_samples = sum(
            f.samples for f in self.function_samples.values()
            if 'evaluate' in f.name.lower()
        )
        if self.total_samples > 0 and eval_samples / self.total_samples > 0.2:
            recommendations.append(
                "Evaluation function takes >20% of time. Consider:\n"
                "   - Incremental evaluation updates\n"
                "   - SIMD vectorization\n"
                "   - Simplifying evaluation for non-PV nodes"
            )
            
        alloc_samples = sum(
            f.samples for f in self.function_samples.values()
            if any(x in f.name.lower() for x in ['malloc', 'free', 'new', 'delete'])
        )
        if self.total_samples > 0 and alloc_samples / self.total_samples > 0.05:
            recommendations.append(
                "Memory allocation takes >5% of time. Consider:\n"
                "   - Pre-allocating move lists\n"
                "   - Using thread-local storage for temporary data\n"
                "   - Stack allocation for small objects"
            )
            
        if self.metrics.nps > 0:
            if self.metrics.nps < 5_000_000:
                recommendations.append(
                    f"NPS ({self.metrics.nps:,}) is relatively low. Consider:\n"
                    "   - Profiling with compiler optimizations enabled (-O3)\n"
                    "   - Checking for unnecessary copy operations\n"
                    "   - Optimizing hot loops"
                )
            elif self.metrics.nps > 10_000_000:
                recommendations.append(
                    f"NPS ({self.metrics.nps:,}) is good. Focus on search improvements:\n"
                    "   - Better move ordering\n"
                    "   - Late move reductions\n"
                    "   - Null move pruning tuning"
                )
                
        if not self.metrics.test_passed and self.metrics.best_move:
            recommendations.append(
                f"Test failed (got {self.metrics.best_move}, expected {self.metrics.expected_move}). Consider:\n"
                "   - Increasing search depth\n"
                "   - Checking evaluation function accuracy\n"
                "   - Reviewing search extensions/reductions"
            )
            
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
                print()
        else:
            print("No significant issues identified. Profile appears healthy.")
            
        print()
        print("=" * 80)
        print("                              END OF REPORT")
        print("=" * 80)


def generate_pdf(text_content: str, output_path: str) -> bool:
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as txt_file:
            ascii_content = text_content.replace('█', '#').replace('░', '-')
            ascii_content = ascii_content.replace('•', '*')
            txt_file.write(ascii_content)
            txt_file_path = txt_file.name
        
        ps_file = tempfile.NamedTemporaryFile(suffix='.ps', delete=False).name
        
        result = subprocess.run(
            ['enscript', '-B', '-f', 'Courier8', '-p', ps_file, txt_file_path],
            capture_output=True
        )
        
        if result.returncode == 0:
            result = subprocess.run(
                ['ps2pdf', ps_file, output_path],
                capture_output=True
            )
            os.unlink(txt_file_path)
            os.unlink(ps_file)
            if result.returncode == 0:
                return True
    except FileNotFoundError:
        pass
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as txt_file:
            ascii_content = text_content.replace('█', '#').replace('░', '-')
            ascii_content = ascii_content.replace('•', '*')
            txt_file.write(ascii_content)
            txt_file_path = txt_file.name
        
        result = subprocess.run(
            ['cupsfilter', '-o', 'media=letter', txt_file_path],
            capture_output=True
        )
        
        if result.returncode == 0:
            with open(output_path, 'wb') as f:
                f.write(result.stdout)
            os.unlink(txt_file_path)
            return True
        os.unlink(txt_file_path)
    except FileNotFoundError:
        pass
    
    escaped_content = text_content.replace('<', '&lt;').replace('>', '&gt;')
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Chess Engine Profile Analysis Report</title>
    <style>
        body {{
            font-family: 'SF Mono', 'Monaco', 'Menlo', 'Consolas', monospace;
            font-size: 9pt;
            line-height: 1.5;
            color: #1a1a1a;
            margin: 0;
            padding: 20px;
        }}
        pre {{
            font-family: inherit;
            margin: 0;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
<pre>{escaped_content}</pre>
</body>
</html>"""
    
    html_path = output_path.replace('.pdf', '.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Note: PDF tools not found. Generated HTML: {html_path}")
    return False


def find_latest_profile_dir(base_path: str = ".") -> Optional[str]:
    profile_dirs = [
        d for d in os.listdir(base_path)
        if d.startswith("profile_results_") and os.path.isdir(os.path.join(base_path, d))
    ]
    
    if not profile_dirs:
        return None
        
    profile_dirs.sort(reverse=True)
    return os.path.join(base_path, profile_dirs[0])


def main():
    parser = argparse.ArgumentParser(description="Analyze chess engine profiling results")
    parser.add_argument("profile_dir", nargs="?", help="Profile results directory")
    parser.add_argument("--list", "-l", action="store_true", help="List available profiles")
    parser.add_argument("--output", "-o", help="Save to text file")
    parser.add_argument("--pdf", "-p", help="Generate PDF report")
    
    args = parser.parse_args()
    
    if args.list:
        profile_dirs = [d for d in os.listdir(".") if d.startswith("profile_results_") and os.path.isdir(d)]
        if profile_dirs:
            print("Available profile directories:")
            for d in sorted(profile_dirs, reverse=True):
                print(f"  {d}")
        else:
            print("No profile directories found")
        return
        
    profile_dir = args.profile_dir
    if not profile_dir:
        profile_dir = find_latest_profile_dir()
        if not profile_dir:
            print("Error: No profile directory found. Run chess_profiler.sh first.")
            sys.exit(1)
        print(f"Using latest profile: {profile_dir}\n")
        
    if not os.path.isdir(profile_dir):
        print(f"Error: Directory not found: {profile_dir}")
        sys.exit(1)
    
    if args.pdf or args.output:
        import io
        captured_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output
        
    try:
        analyzer = ProfileAnalyzer(profile_dir)
        analyzer.analyze()
    finally:
        if args.pdf or args.output:
            sys.stdout = original_stdout
            output_text = captured_output.getvalue()
            print(output_text)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output_text)
                print(f"Analysis saved to: {args.output}")
            
            if args.pdf:
                pdf_path = args.pdf
                if not pdf_path.endswith('.pdf'):
                    pdf_path += '.pdf'
                if generate_pdf(output_text, pdf_path):
                    print(f"PDF report generated: {pdf_path}")


if __name__ == "__main__":
    main()

