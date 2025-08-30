#!/bin/bash

# Comprehensive Model Testing Runner
# This script runs the comprehensive model test with proper environment setup

set -e  # Exit on error

echo "==============================================="
echo "COMPREHENSIVE MODEL TESTING SCRIPT"
echo "==============================================="
echo "Start time: $(date)"
echo ""

export CUSTOM_MODEL_BASE_URL="https://chat-ai.academiccloud.de/v1"
export CUSTOM_MODEL_API_KEY="YOUR_API_KEY_HERE"    

# Check if required environment variables are set
if [ -z "$CUSTOM_MODEL_BASE_URL" ] || [ -z "$CUSTOM_MODEL_API_KEY" ]; then
    echo "Warning: CUSTOM_MODEL_BASE_URL or CUSTOM_MODEL_API_KEY not set"
    echo "Custom models may fail without these environment variables"
    echo ""
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Clear old logs
echo "Clearing old log files..."
rm -f logs/*.txt
echo "Old logs cleared."
echo ""

# Set Python path to include project root  
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../.."

# Run the comprehensive test
echo "Starting comprehensive model test..."
echo "This will test all models on all prompts with proper rate limiting."
echo "Expected duration: 30-60 minutes depending on model response times."
echo ""

python3 comprehensive_model_test.py

echo ""
echo "==============================================="
echo "TEST COMPLETED"
echo "==============================================="
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - logs/test_summary.txt (overall summary)"
echo "  - logs/{model_name}_test_log.txt (detailed logs per model)"
echo ""
echo "Check the logs directory for detailed results." 