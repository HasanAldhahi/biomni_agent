#!/bin/bash

# List of models to test
models=(
    "qwen3-235b-a22b"
    "codestral-22b"
    "qwen2.5-coder-32b-instruct"
)

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for this run
timestamp=$(date +"%Y%m%d_%H%M%S")

echo "Starting model testing at $(date)"
echo "Results will be saved in logs/ directory"

# Test each model with both A1 and React agents
for model in "${models[@]}"; do
    echo "=================================="
    echo "Testing model: $model"
    echo "=================================="
    
    # Clean model name for filename (replace special characters)
    clean_model=$(echo "$model" | sed 's/[^a-zA-Z0-9]/_/g')
    
    # Set environment variable for the current model
    export CURRENT_TEST_MODEL="$model"
    
    echo "Testing A1 agent with $model..."
    python test_a1_dynamic.py > "logs/a1_${clean_model}_${timestamp}.txt" 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ A1 agent test completed successfully"
    else
        echo "✗ A1 agent test failed"
    fi
    
    # echo "Testing React agent with $model..."
    # python test_react_dynamic.py > "logs/react_${clean_model}_${timestamp}.log" 2>&1
    # if [ $? -eq 0 ]; then
    #     echo "✓ React agent test completed successfully"
    # else
    #     echo "✗ React agent test failed"
    # fi
    
    echo "Completed testing $model"
    echo ""
done

# Test Gemini separately
echo "=================================="
echo "Testing Gemini model separately..."
echo "=================================="
python test_gemini.py > "logs/gemini_${timestamp}.log" 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Gemini test completed successfully"
else
    echo "✗ Gemini test failed"
fi

echo "All tests completed at $(date)"
echo "Check logs/ directory for individual results" 