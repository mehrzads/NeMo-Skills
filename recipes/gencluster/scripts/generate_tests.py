#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time
import json

def run_generator(gen_binary_path, timeout=10):
    """Run a generator binary and return its stdout output"""
    try:
        result = subprocess.run(
            [str(gen_binary_path)], 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, f"Generator failed with return code {result.returncode}: {result.stderr}"
    
    except subprocess.TimeoutExpired:
        return False, "Generator timed out"
    except Exception as e:
        return False, f"Generator error: {str(e)}"

def run_validator(val_binary_path, test_data, timeout=10):
    """Run a validator binary with test data as stdin"""
    try:
        result = subprocess.run(
            [str(val_binary_path)], 
            input=test_data,
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        
        # Check if validator output contains "passed" or "failed" (case insensitive)
        output_lower = result.stdout.lower().strip()
        
        if "passed" in output_lower:
            return "passed"
        elif "failed" in output_lower:
            return "failed"
        else:
            # If no clear passed/failed, check return code
            return "passed" if result.returncode == 0 else "failed"
    
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception as e:
        return "error"

def validate_dataset(test_data, val_binaries, min_validators=7):
    """Validate test data against all validators and return if it passes threshold"""
    validation_results = []
    passed_count = 0
    
    for val_binary in val_binaries:
        result = run_validator(val_binary, test_data)
        validation_results.append({
            'validator': val_binary.name,
            'result': result
        })
        
        if result == "passed":
            passed_count += 1
    
    if (len(val_binaries) * 0.8) < min_validators:
        is_valid = passed_count >= (len(val_binaries) * 0.8)
    else:
        is_valid = passed_count >= min_validators
 
    return is_valid, passed_count, len(val_binaries), validation_results

def generate_datasets_for_problem(problem_dir, binary_dir, output_dir, n_datasets, min_validators=7):
    """Generate N validated datasets for a single problem"""
    problem_name = problem_dir.name
    
    # Get generator and validator binaries
    gen_dir = binary_dir / problem_name / "gen"
    val_dir = binary_dir / problem_name / "val"
    
    if not gen_dir.exists() or not val_dir.exists():
        print(f"⚠️  Skipping {problem_name}: missing gen or val directory")
        return 0, []
    
    gen_binaries = [f for f in gen_dir.iterdir() if f.is_file() and os.access(f, os.X_OK)]
    val_binaries = [f for f in val_dir.iterdir() if f.is_file() and os.access(f, os.X_OK)]
    
    if len(gen_binaries) == 0:
        print(f"⚠️  Skipping {problem_name}: no generator binaries found")
        return 0, []
    
    if len(val_binaries) == 0:
        print(f"⚠️  Skipping {problem_name}: no validator binaries found")
        return 0, []
    
    print(f"\n=== Problem {problem_name} ===")
    print(f"Generators: {len(gen_binaries)}, Validators: {len(val_binaries)}")
    print(f"Target: {n_datasets} datasets (need ≥{min_validators}/{len(val_binaries)} validator approval)")
    
    # Create output directory for this problem (just the number, not "problem_X")
    problem_number = problem_name.replace("problem_", "")
    problem_output_dir = output_dir / problem_number
    problem_output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_datasets = []
    gen_index = 0  # Round-robin index
    attempts = 0
    max_attempts = n_datasets * 10  # Safety limit
    
    while len(generated_datasets) < n_datasets and attempts < max_attempts:
        if len(gen_binaries) == 0:
            print("  ⚠️ No generators remaining for this problem. Stopping.")
            break
        attempts += 1
        current_gen = gen_binaries[gen_index % len(gen_binaries)]
        
        print(f"Attempt {attempts}: Using generator {current_gen.name}...")
        
        # Generate test data
        success, test_data = run_generator(current_gen)
        
        if not success:
            print(f"  ❌ Generator failed: {test_data}")
            gen_index += 1
            continue
        
        # Validate the generated data
        is_valid, passed_count, total_validators, validation_results = validate_dataset(
            test_data, val_binaries, min_validators
        )
        
        if is_valid:
            # Save the dataset
            dataset_filename = f"dataset_{len(generated_datasets) + 1:03d}.txt"
            dataset_path = problem_output_dir / dataset_filename
            
            with open(dataset_path, 'w') as f:
                f.write(test_data)
            
            # Save validation report
            report_filename = f"dataset_{len(generated_datasets) + 1:03d}_validation.json"
            report_path = problem_output_dir / report_filename
            
            validation_report = {
                'dataset_file': dataset_filename,
                'generator': current_gen.name,
                'passed_validators': passed_count,
                'total_validators': total_validators,
                'validation_results': validation_results,
                'attempt_number': attempts
            }
            
            with open(report_path, 'w') as f:
                json.dump(validation_report, f, indent=2)
            
            generated_datasets.append({
                'dataset_path': dataset_path,
                'validation_report': validation_report
            })
            
            print(f"  ✅ Dataset {len(generated_datasets)}/{n_datasets} saved: {passed_count}/{total_validators} validators passed")
            gen_index += 1
        else:
            print(f"  ❌ Validation failed: only {passed_count}/{total_validators} validators passed")
            # Drop this generator from future round-robin attempts
            print(f"  ⛔ Dropping generator {current_gen.name} due to failed validation")
            gen_binaries = [g for g in gen_binaries if g != current_gen]
            if len(gen_binaries) == 0:
                print("  ⚠️ No generators remaining for this problem. Stopping.")
                break
            if gen_index >= len(gen_binaries):
                gen_index = 0
    
    if len(generated_datasets) < n_datasets:
        print(f"⚠️  Warning: Only generated {len(generated_datasets)}/{n_datasets} datasets after {max_attempts} attempts")
    
    return len(generated_datasets), generated_datasets

def main():
    parser = argparse.ArgumentParser(description='Generate N validated datasets per problem')
    parser.add_argument('n_datasets', type=int, help='Number of datasets to generate per problem')
    parser.add_argument('--min-validators', type=int, default=7, 
                       help='Minimum number of validators that must pass (default: 7)')
    parser.add_argument('--base-dir', type=str, default=None,
                       help='Base directory (default: current directory)')
    
    args = parser.parse_args()
    
    if args.n_datasets <= 0:
        print("Error: N must be a positive integer")
        sys.exit(1)
    
    # Set up directories
    base_dir = Path(args.base_dir) if args.base_dir else Path.cwd()
    binary_dir = base_dir / "compiled_binaries" 
    output_dir = base_dir / "generated_datasets"
    
    if not binary_dir.exists():
        print(f"Error: Binary directory {binary_dir} does not exist")
        print("Please run the extraction and compilation script first")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Get all problem directories
    problem_dirs = []
    for item in binary_dir.iterdir():
        if item.is_dir() and item.name.startswith('problem_'):
            problem_dirs.append(item)
    
    problem_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
    
    if not problem_dirs:
        print(f"Error: No problem directories found in {binary_dir}")
        sys.exit(1)
    
    print(f"🚀 Starting dataset generation")
    print(f"Target: {args.n_datasets} datasets per problem")
    print(f"Minimum validators: {args.min_validators}")
    print(f"Problems found: {len(problem_dirs)}")
    print(f"Output directory: {output_dir}")
    
    # Process each problem
    total_datasets = 0
    successful_problems = 0
    
    start_time = time.time()
    
    for problem_dir in problem_dirs:
        datasets_generated, _ = generate_datasets_for_problem(
            problem_dir, 
            binary_dir, 
            output_dir, 
            args.n_datasets, 
            args.min_validators
        )
        
        total_datasets += datasets_generated
        if datasets_generated > 0:
            successful_problems += 1
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Final summary
    print(f"\n🎯 FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"Problems processed: {len(problem_dirs)}")
    print(f"Problems with datasets: {successful_problems}")
    print(f"Total datasets generated: {total_datasets}")
    print(f"Target datasets: {len(problem_dirs) * args.n_datasets}")
    print(f"Success rate: {total_datasets/(len(problem_dirs) * args.n_datasets)*100:.1f}%")
    print(f"Time taken: {duration:.1f} seconds")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
