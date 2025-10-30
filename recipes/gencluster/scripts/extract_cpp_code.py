#!/usr/bin/env python3

import os
import re
import json
import subprocess
import argparse
from pathlib import Path

def extract_final_cpp_block(text):
    """Extract the final C++ code block from text using the provided pattern"""
    pattern = r"```(?:cpp|Cpp)\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1] if matches else ""

def compile_cpp_file(cpp_file_path, binary_dir):
    """Compile a C++ file and return compilation status"""
    cpp_file = Path(cpp_file_path)
    binary_name = cpp_file.stem  # filename without extension
    binary_path = binary_dir / binary_name
    
    # Compilation command with optimization and warnings
    compile_cmd = [
        'g++', '-std=c++17', '-O2', '-Wall', '-Wextra',
        '-o', str(binary_path), str(cpp_file)
    ]
    
    try:
        # Run compilation
        result = subprocess.run(
            compile_cmd, 
            capture_output=True, 
            text=True, 
            timeout=30  # 30 second timeout
        )
        
        if result.returncode == 0:
            return True, "Success", ""
        else:
            return False, "Compilation failed", result.stderr
            
    except subprocess.TimeoutExpired:
        return False, "Timeout", "Compilation timed out after 30 seconds"
    except Exception as e:
        return False, "Error", str(e)

def process_jsonl_file(jsonl_path, output_dir, binary_dir, folder_name, file_id, source_id):
    """Process a single JSONL file and extract C++ code blocks for a given source (rs0/rs1)"""
    extracted_count = 0
    compiled_count = 0
    compilation_results = []
    
    # Create organized directory structure: problem_X/gen or problem_X/val
    problem_dir = output_dir / f"problem_{file_id}"
    problem_type_dir = problem_dir / ("gen" if folder_name == "ioi_gen" else "val")
    problem_type_dir.mkdir(parents=True, exist_ok=True)
    
    # Create corresponding binary directory structure
    binary_problem_dir = binary_dir / f"problem_{file_id}"
    binary_type_dir = binary_problem_dir / ("gen" if folder_name == "ioi_gen" else "val")
    binary_type_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    # Parse JSON line
                    data = json.loads(line.strip())
                    generation = data.get('generation', '')
                    
                    if generation:
                        # Extract C++ code
                        cpp_code = extract_final_cpp_block(generation)
                        
                        if cpp_code.strip():  # Only save if there's actual code
                            # Create organized filename including source to avoid collisions across rs0/rs1
                            output_filename = f"{source_id}_line_{line_num}.cpp"
                            output_path = problem_type_dir / output_filename
                            
                            # Save the C++ code
                            with open(output_path, 'w', encoding='utf-8') as cpp_file:
                                cpp_file.write(cpp_code.strip())
                            
                            extracted_count += 1
                            relative_path = f"problem_{file_id}/{'gen' if folder_name == 'ioi_gen' else 'val'}/{output_filename}"
                            print(f"Extracted C++ code to: {relative_path}")
                            
                            # Compile the C++ file
                            success, status, error_msg = compile_cpp_file(output_path, binary_type_dir)
                            compilation_results.append({
                                'file': relative_path,
                                'success': success,
                                'status': status,
                                'error': error_msg
                            })
                            
                            if success:
                                compiled_count += 1
                                print(f"  ✓ Compiled successfully")
                            else:
                                print(f"  ✗ Compilation failed: {status}")
                                if error_msg.strip():
                                    print(f"    Error: {error_msg.strip()[:100]}...")  # First 100 chars
                
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in {jsonl_path} line {line_num}: {e}")
                except Exception as e:
                    print(f"Error processing line {line_num} in {jsonl_path}: {e}")
    
    except Exception as e:
        print(f"Error reading file {jsonl_path}: {e}")
    
    return extracted_count, compiled_count, compilation_results

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Extract and compile C++ code from JSONL under generators/ and validators/")
    parser.add_argument("--input_dir", required=True, help="Input directory containing generators/ and validators/ folders")
    args = parser.parse_args()

    # Base directory
    base_dir = Path(args.input_dir)

    # Output directories
    output_dir = base_dir / "extracted_cpp"
    binary_dir = base_dir / "compiled_binaries"
    output_dir.mkdir(exist_ok=True)
    binary_dir.mkdir(exist_ok=True)
    
    # Statistics tracking
    total_extracted = 0
    total_compiled = 0
    processed_files = 0
    all_compilation_results = []
    failed_compilations = []
    
    # Check if g++ is available
    try:
        subprocess.run(['g++', '--version'], capture_output=True, check=True)
        print("✓ g++ compiler found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ g++ compiler not found! Please install g++ to enable compilation.")
        return
    
    # Process both generators and validators folders
    for folder_name in ['generators', 'validators']:
        folder_path = base_dir / folder_name

        if not folder_path.exists():
            print(f"Folder {folder_name} does not exist, skipping...")
            continue

        print(f"\n=== Processing {folder_name} ===")

        # Find all JSONL files recursively under this folder
        jsonl_files = sorted(folder_path.rglob("*.jsonl"), key=lambda p: str(p))
        if not jsonl_files:
            print(f"No JSONL files found in: {folder_path}")
            continue

        for fpath in jsonl_files:
            print(f"\nProcessing: {fpath.relative_to(base_dir)}")
            # Use parent directory name as file_id if numeric, else use filename stem
            parent = fpath.parent.name
            file_id = parent if parent.isdigit() else fpath.stem
            rs = fpath.stem  # best-effort identifier
            extracted, compiled, compilation_results = process_jsonl_file(
                fpath,
                output_dir,
                binary_dir,
                folder_name,
                file_id,
                rs
            )

            total_extracted += extracted
            total_compiled += compiled
            processed_files += 1
            all_compilation_results.extend(compilation_results)

            failed_in_file = [r for r in compilation_results if not r['success']]
            failed_compilations.extend(failed_in_file)

            print(f"  -> Extracted {extracted} C++ files, compiled {compiled}/{extracted} successfully")
            if failed_in_file:
                print(f"  -> {len(failed_in_file)} compilation failures")
    
    # Final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Processed JSONL files: {processed_files}")
    print(f"Total C++ files extracted: {total_extracted}")
    print(f"Total C++ files compiled successfully: {total_compiled}")
    print(f"Compilation success rate: {total_compiled/total_extracted*100:.1f}%" if total_extracted > 0 else "N/A")
    print(f"\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"    problem_*/")
    print(f"      gen/        (from ioi_gen)")
    print(f"      val/        (from ioi_val)")
    print(f"  {binary_dir}/")
    print(f"    problem_*/")
    print(f"      gen/        (compiled binaries)")
    print(f"      val/        (compiled binaries)")
    
    # Detailed failure analysis
    if failed_compilations:
        print(f"\n=== COMPILATION FAILURES ({len(failed_compilations)}) ===")
        failure_types = {}
        for failure in failed_compilations:
            status = failure['status']
            failure_types[status] = failure_types.get(status, 0) + 1
        
        for failure_type, count in sorted(failure_types.items()):
            print(f"  {failure_type}: {count} files")
        
        # Show first few detailed errors
        print(f"\nFirst 5 compilation errors:")
        for i, failure in enumerate(failed_compilations[:5]):
            print(f"  {i+1}. {failure['file']}: {failure['status']}")
            if failure['error'].strip():
                error_lines = failure['error'].strip().split('\n')
                for line in error_lines[:2]:  # Show first 2 lines of error
                    print(f"     {line}")
                if len(error_lines) > 2:
                    print(f"     ... (and {len(error_lines)-2} more lines)")
    else:
        print("\n🎉 All files compiled successfully!")

if __name__ == "__main__":
    main()
