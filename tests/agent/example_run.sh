#!/bin/bash

# Example script showing how to set up environment and run comprehensive tests

echo "=== COMPREHENSIVE MODEL TESTING EXAMPLE ==="
echo ""

# 1. Set up environment variables (replace with your actual values)
echo "Setting up environment variables..."
export CUSTOM_MODEL_BASE_URL="https://your-custom-model-endpoint.com/v1"
export CUSTOM_MODEL_API_KEY="your-api-key-here"

echo "Environment configured:"
echo "  CUSTOM_MODEL_BASE_URL: $CUSTOM_MODEL_BASE_URL"
echo "  CUSTOM_MODEL_API_KEY: [HIDDEN]"
echo ""

# 2. Show what will be tested
echo "This test will run:"
echo "  - 4 models: Gemini 2.5 Pro, Qwen3-235B, Codestral-22B, Qwen2.5-Coder-32B"
echo "  - 17 prompts: 6 easy + 5 medium + 5 hard"
echo "  - Total: 68 individual tests (4 models × 17 prompts)"
echo "  - Duration: ~30-60 minutes"
echo ""

# 3. Run the actual test
echo "Starting comprehensive test..."
echo "Press Ctrl+C to cancel, or wait for automatic execution in 5 seconds..."
sleep 5

./run_comprehensive_test.sh

echo ""
echo "=== TEST COMPLETED ==="
echo "Check logs/ directory for results!" 