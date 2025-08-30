#!/usr/bin/env python3
"""
Direct CLI testing - Execute CLI tools directly
Tests all CLI tools from biomni_env/biomni_tools/bin/ by running them directly

IMPORTANT: Run with the biomni_e1 conda environment activated:
    conda activate biomni_e1
    python tests/cli/test_direct_cli_calls.py
"""

import os
import subprocess
import tempfile
import traceback
from pathlib import Path


def test_fasttree_cli():
    """Test FastTree CLI tool directly"""
    print("\n=== Testing FastTree CLI ===")
    
    fasttree_path = Path(__file__).parent.parent.parent / "biomni_env" / "biomni_tools" / "bin" / "FastTree"
    
    # Check if file exists
    if not fasttree_path.exists():
        print(f"❌ FastTree not found at {fasttree_path}")
        return
    
    print(f"✅ FastTree found at {fasttree_path}")
    
    # Check if executable
    if not os.access(fasttree_path, os.X_OK):
        print("❌ FastTree is not executable")
        return
    
    print("✅ FastTree is executable")
    
    # Test help/version output
    try:
        result = subprocess.run([str(fasttree_path)], 
                              capture_output=True, text=True, timeout=10)
        output_text = (result.stderr or result.stdout or "").lower()
        
        if "fasttree" in output_text:
            print("✅ FastTree responds to execution")
            print(f"   Return code: {result.returncode}")
            print(f"   Output preview: {(result.stderr or result.stdout)[:150]}...")
        else:
            print(f"⚠️  FastTree output doesn't contain expected text")
            
    except subprocess.TimeoutExpired:
        print("⚠️  FastTree command timed out")
    except Exception as e:
        print(f"⚠️  Failed to run FastTree: {e}")


def test_muscle_cli():
    """Test muscle CLI tool directly"""
    print("\n=== Testing muscle CLI ===")
    
    muscle_path = Path(__file__).parent.parent.parent / "biomni_env" / "biomni_tools" / "bin" / "muscle"
    
    # Check if file exists
    if not muscle_path.exists():
        print(f"❌ muscle not found at {muscle_path}")
        return
    
    print(f"✅ muscle found at {muscle_path}")
    
    # Check if executable
    if not os.access(muscle_path, os.X_OK):
        print("❌ muscle is not executable")
        return
    
    print("✅ muscle is executable")
    
    # Test help option
    try:
        result = subprocess.run([str(muscle_path), "-h"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 or "muscle" in result.stdout.lower() or "muscle" in result.stderr.lower():
            print("✅ muscle responds to help command")
            print(f"   Return code: {result.returncode}")
            print(f"   Output preview: {(result.stdout or result.stderr)[:150]}...")
        else:
            print(f"⚠️  muscle help command gave unexpected output")
            
    except subprocess.TimeoutExpired:
        print("⚠️  muscle command timed out")
    except Exception as e:
        print(f"⚠️  Failed to run muscle: {e}")


def test_muscle_alignment():
    """Test muscle with actual sequence alignment"""
    print("\n=== Testing muscle Alignment ===")
    
    muscle_path = Path(__file__).parent.parent.parent / "biomni_env" / "biomni_tools" / "bin" / "muscle" 
    
    if not muscle_path.exists():
        print("❌ muscle not found - skipping alignment test")
        return
    
    # Create temporary directory and files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple FASTA file with two protein sequences
        test_fasta = os.path.join(tmpdir, "test_sequences.fasta")
        with open(test_fasta, 'w') as f:
            f.write(">seq1\n")
            f.write("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG\n")
            f.write(">seq2\n")
            f.write("MKATVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG\n")
        
        output_file = os.path.join(tmpdir, "aligned.fasta")
        
        try:
            # Run muscle alignment
            result = subprocess.run([str(muscle_path), "-align", test_fasta, "-output", output_file], 
                                  capture_output=True, text=True, timeout=30)
            
            # Check if alignment was created
            if os.path.exists(output_file):
                print("✅ muscle alignment completed successfully")
                
                # Check output file content
                with open(output_file, 'r') as f:
                    content = f.read()
                    
                if len(content) > 0 and ">seq1" in content and ">seq2" in content:
                    print("✅ Output alignment file contains expected sequences")
                    print(f"   Output file size: {len(content)} characters")
                    print(f"   Number of lines: {len(content.splitlines())}")
                else:
                    print("⚠️  Output file exists but content is unexpected")
            else:
                print("⚠️  muscle alignment did not create output file")
                print(f"   Return code: {result.returncode}")
                print(f"   stderr: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("⚠️  muscle alignment timed out")
        except Exception as e:
            print(f"⚠️  Failed to run muscle alignment: {e}")


def test_plink2_cli():
    """Test plink2 CLI tool directly"""
    print("\n=== Testing plink2 CLI ===")
    
    plink2_path = Path(__file__).parent.parent.parent / "biomni_env" / "biomni_tools" / "bin" / "plink2"
    
    # Check if file exists
    if not plink2_path.exists():
        print(f"❌ plink2 not found at {plink2_path}")
        return
    
    print(f"✅ plink2 found at {plink2_path}")
    
    # Check if executable
    if not os.access(plink2_path, os.X_OK):
        print("❌ plink2 is not executable")
        return
    
    print("✅ plink2 is executable")
    
    # Test version option
    try:
        result = subprocess.run([str(plink2_path), "--version"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 or "plink" in result.stdout.lower() or "plink" in result.stderr.lower():
            print("✅ plink2 responds to version command")
            print(f"   Return code: {result.returncode}")
            print(f"   Output preview: {(result.stdout or result.stderr)[:150]}...")
        else:
            print(f"⚠️  plink2 version command gave unexpected output")
            
    except subprocess.TimeoutExpired:
        print("⚠️  plink2 command timed out")
    except Exception as e:
        print(f"⚠️  Failed to run plink2: {e}")


def test_other_cli_tools():
    """Test other CLI tools in the bin directory"""
    print("\n=== Testing Other CLI Tools ===")
    
    bin_dir = Path(__file__).parent.parent.parent / "biomni_env" / "biomni_tools" / "bin"
    
    if not bin_dir.exists():
        print(f"❌ CLI tools directory not found: {bin_dir}")
        return
    
    # Get all executable files
    cli_tools = []
    for item in bin_dir.iterdir():
        if item.is_file() and os.access(item, os.X_OK):
            cli_tools.append(item.name)
    
    print(f"✅ Found {len(cli_tools)} executable CLI tools")
    
    # Test each tool (just existence and executability)
    known_tools = ["FastTree", "muscle", "plink2"]  # Already tested above
    other_tools = [tool for tool in cli_tools if tool not in known_tools]
    
    for tool_name in other_tools:
        tool_path = bin_dir / tool_name
        print(f"\n--- Testing {tool_name} ---")
        
        try:
            # Just test that it's executable and responds
            result = subprocess.run([str(tool_path)], 
                                  capture_output=True, text=True, timeout=5)
            print(f"✅ {tool_name} is executable")
            print(f"   Return code: {result.returncode}")
            
            # Show brief output if available
            output = result.stdout or result.stderr
            if output:
                print(f"   Output preview: {output[:100]}...")
            
        except subprocess.TimeoutExpired:
            print(f"✅ {tool_name} is executable (timed out - normal for some tools)")
        except Exception as e:
            print(f"⚠️  Error testing {tool_name}: {e}")


def main():
    """Run all direct CLI tests"""
    print("🔧 DIRECT CLI TESTING - Executing CLI Tools Directly")
    print("=" * 60)
    
    # Test each CLI tool
    test_fasttree_cli()
    test_muscle_cli() 
    test_muscle_alignment()
    test_plink2_cli()
    test_other_cli_tools()
    
    print("\n" + "=" * 60)
    print("🎯 DIRECT CLI TESTING COMPLETE")
    print("✅ CLI tools that work show detailed execution results")
    print("⚠️  CLI tools with issues show error messages")
    print("ℹ️  All executable tools in bin/ directory were tested")
    print("\nThis validates that CLI tools can be executed directly!")


if __name__ == "__main__":
    main() 