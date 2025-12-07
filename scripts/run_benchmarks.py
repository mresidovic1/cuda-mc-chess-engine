#!/usr/bin/env python3

import json
import sys
import os
import subprocess
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import Optional


@dataclass
class StageResult:
    name: str
    easy_passed: int
    easy_total: int
    medium_passed: int
    medium_total: int
    hard_passed: int
    hard_total: int
    total_passed: int
    total_tests: int
    total_time_ms: int
    nps: int = 0


def run_test_suite(engine_path: str, engine_name: str, mode: str, depth: int, time_limit: int) -> Optional[dict]:
    if not os.path.exists(engine_path):
        print(f"Error: Engine not found: {engine_path}")
        return None
    
    args = [engine_path, f"--engine={engine_name}", "--json", "--quiet"]
    
    if mode == "depth":
        args.extend([f"--mode=depth", f"--depth={depth}"])
    else:
        args.extend([f"--mode=time", f"--time={time_limit}"])
    
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=600)
        output = result.stdout
        
        json_start = output.find('{"engine"')
        if json_start == -1:
            json_start = output.find('{"results"')
        
        if json_start >= 0:
            json_str = output[json_start:]
            return json.loads(json_str)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"Error running {engine_name}: {e}")
    
    return None


def parse_results(data: dict, name: str) -> StageResult:
    results = data.get('results', [])
    
    easy_passed, easy_total = 0, 0
    medium_passed, medium_total = 0, 0
    hard_passed, hard_total = 0, 0
    
    for r in results:
        diff = r.get('difficulty', '')
        passed = r.get('passed', False)
        
        if diff == 'EASY':
            easy_total += 1
            if passed: easy_passed += 1
        elif diff == 'MEDIUM':
            medium_total += 1
            if passed: medium_passed += 1
        elif diff == 'HARD':
            hard_total += 1
            if passed: hard_passed += 1
    
    return StageResult(
        name=name,
        easy_passed=easy_passed,
        easy_total=easy_total,
        medium_passed=medium_passed,
        medium_total=medium_total,
        hard_passed=hard_passed,
        hard_total=hard_total,
        total_passed=easy_passed + medium_passed + hard_passed,
        total_tests=easy_total + medium_total + hard_total,
        total_time_ms=data.get('total_time_ms', 0)
    )


def print_comparison_table(results: list[StageResult], output_file: Optional[str] = None):
    lines = []
    
    def add(line: str):
        lines.append(line)
        print(line)
    
    add("=" * 100)
    add("                           CHESS ENGINE OPTIMIZATION COMPARISON")
    add("=" * 100)
    add(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"Stages compared: {len(results)}")
    add("")
    
    add("PASS RATE BY DIFFICULTY")
    add("-" * 100)
    header = f"{'Stage':<20} {'Easy':<15} {'Medium':<15} {'Hard':<15} {'Total':<15} {'Time(s)':<10}"
    add(header)
    add("-" * 100)
    
    for r in results:
        easy_str = f"{r.easy_passed}/{r.easy_total}" if r.easy_total > 0 else "-"
        med_str = f"{r.medium_passed}/{r.medium_total}" if r.medium_total > 0 else "-"
        hard_str = f"{r.hard_passed}/{r.hard_total}" if r.hard_total > 0 else "-"
        total_pct = 100 * r.total_passed / r.total_tests if r.total_tests > 0 else 0
        total_str = f"{r.total_passed}/{r.total_tests} ({total_pct:.0f}%)"
        time_str = f"{r.total_time_ms / 1000:.1f}"
        
        add(f"{r.name:<20} {easy_str:<15} {med_str:<15} {hard_str:<15} {total_str:<15} {time_str:<10}")
    
    add("")
    add("PERCENTAGE TABLE")
    add("-" * 100)
    add(f"{'Stage':<20} {'Easy %':<15} {'Medium %':<15} {'Hard %':<15} {'Total %':<15}")
    add("-" * 100)
    
    for r in results:
        easy_pct = f"{100 * r.easy_passed / r.easy_total:.1f}%" if r.easy_total > 0 else "-"
        med_pct = f"{100 * r.medium_passed / r.medium_total:.1f}%" if r.medium_total > 0 else "-"
        hard_pct = f"{100 * r.hard_passed / r.hard_total:.1f}%" if r.hard_total > 0 else "-"
        total_pct = f"{100 * r.total_passed / r.total_tests:.1f}%" if r.total_tests > 0 else "-"
        
        add(f"{r.name:<20} {easy_pct:<15} {med_pct:<15} {hard_pct:<15} {total_pct:<15}")
    
    add("")
    add("IMPROVEMENT ANALYSIS")
    add("-" * 100)
    
    if len(results) >= 2:
        baseline = results[0]
        for r in results[1:]:
            if baseline.total_tests > 0 and r.total_tests > 0:
                base_pct = 100 * baseline.total_passed / baseline.total_tests
                curr_pct = 100 * r.total_passed / r.total_tests
                diff = curr_pct - base_pct
                sign = "+" if diff >= 0 else ""
                
                time_diff = r.total_time_ms - baseline.total_time_ms
                time_sign = "+" if time_diff >= 0 else ""
                
                add(f"{baseline.name} -> {r.name}:")
                add(f"  Pass rate: {base_pct:.1f}% -> {curr_pct:.1f}% ({sign}{diff:.1f}%)")
                add(f"  Time: {baseline.total_time_ms/1000:.1f}s -> {r.total_time_ms/1000:.1f}s ({time_sign}{time_diff/1000:.1f}s)")
                add("")
    
    add("=" * 100)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        print(f"\nReport saved to: {output_file}")


def generate_csv(results: list[StageResult], output_file: str):
    with open(output_file, 'w') as f:
        f.write("Stage,Easy Passed,Easy Total,Easy %,Medium Passed,Medium Total,Medium %,")
        f.write("Hard Passed,Hard Total,Hard %,Total Passed,Total Tests,Total %,Time (ms)\n")
        
        for r in results:
            easy_pct = 100 * r.easy_passed / r.easy_total if r.easy_total > 0 else 0
            med_pct = 100 * r.medium_passed / r.medium_total if r.medium_total > 0 else 0
            hard_pct = 100 * r.hard_passed / r.hard_total if r.hard_total > 0 else 0
            total_pct = 100 * r.total_passed / r.total_tests if r.total_tests > 0 else 0
            
            f.write(f"{r.name},{r.easy_passed},{r.easy_total},{easy_pct:.1f},")
            f.write(f"{r.medium_passed},{r.medium_total},{med_pct:.1f},")
            f.write(f"{r.hard_passed},{r.hard_total},{hard_pct:.1f},")
            f.write(f"{r.total_passed},{r.total_tests},{total_pct:.1f},{r.total_time_ms}\n")
    
    print(f"CSV saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks across multiple engine stages")
    parser.add_argument("--engines", "-e", required=True,
                       help="Comma-separated list of engine:name pairs (e.g., build/v1:Stage1,build/v2:Stage2)")
    parser.add_argument("--mode", "-m", choices=["depth", "time"], default="depth",
                       help="Test mode (default: depth)")
    parser.add_argument("--depth", "-d", type=int, default=20, help="Search depth (default: 20)")
    parser.add_argument("--time", "-t", type=int, default=30, help="Time limit in seconds (default: 30)")
    parser.add_argument("--output", "-o", help="Output report file")
    parser.add_argument("--csv", help="Output CSV file")
    parser.add_argument("--json-dir", help="Directory to save individual JSON results")
    
    args = parser.parse_args()
    
    engine_pairs = []
    for pair in args.engines.split(','):
        if ':' in pair:
            path, name = pair.split(':', 1)
        else:
            path = pair
            name = os.path.basename(pair)
        engine_pairs.append((path.strip(), name.strip()))
    
    if args.json_dir:
        os.makedirs(args.json_dir, exist_ok=True)
    
    all_results = []
    
    for engine_path, engine_name in engine_pairs:
        print(f"\nRunning tests for: {engine_name}")
        print("-" * 50)
        
        data = run_test_suite(engine_path, engine_name, args.mode, args.depth, args.time)
        
        if data:
            result = parse_results(data, engine_name)
            all_results.append(result)
            
            if args.json_dir:
                json_path = os.path.join(args.json_dir, f"{engine_name}_results.json")
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"JSON saved: {json_path}")
        else:
            print(f"Failed to get results for {engine_name}")
    
    if all_results:
        print("\n")
        print_comparison_table(all_results, args.output)
        
        if args.csv:
            generate_csv(all_results, args.csv)


if __name__ == "__main__":
    main()

