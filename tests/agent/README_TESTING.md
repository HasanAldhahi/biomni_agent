# Comprehensive Model Testing System

This testing system evaluates multiple AI models on bioinformatics prompts from three difficulty levels.

## Overview

The system tests 4 models:
- **Gemini 2.5 Pro** (with 5 calls/minute rate limiting)
- **Qwen3-235B-A22B** (custom model)
- **Codestral-22B** (custom model)  
- **Qwen2.5-Coder-32B-Instruct** (custom model)

On 17 prompts across 3 difficulty levels:
- **Easy** (6 prompts): Basic bioinformatics questions
- **Medium** (5 prompts): Complex analysis tasks
- **Hard** (5 prompts): Advanced multi-omics and computational biology

## Test Execution Strategy

The system executes tests **question by question** (not model by model):
1. Question 1 → All 4 models
2. Question 2 → All 4 models  
3. And so on...

This ensures fair comparison across models for each prompt type.

## Files

### Core Files
- `comprehensive_model_test.py` - Main testing script
- `run_comprehensive_test.sh` - Shell wrapper script
- `test_prompts_easy.txt` - Easy difficulty prompts
- `test_prompts_medium.txt` - Medium difficulty prompts  
- `test_prompts_hard.txt` - Hard difficulty prompts

### Legacy Files (for reference)
- `test_gemini.py` - Original Gemini-specific test
- `test_a1_dynamic.py` - Original dynamic model test

## Setup

### Environment Variables
Set these before running tests:
```bash
export CUSTOM_MODEL_BASE_URL="your_custom_model_endpoint"
export CUSTOM_MODEL_API_KEY="your_api_key"
```

### Dependencies
Ensure you have the `biomni` package installed and accessible:
```bash
pip install -r requirements.txt  # or your preferred installation method
```

## Running Tests

### Quick Start
```bash
cd tests/agent
./run_comprehensive_test.sh
```

### Manual Execution
```bash
cd tests/agent
python3 comprehensive_model_test.py
```

## Test Features

### Rate Limiting
- **Gemini**: Automatically respects 5 calls/minute limit with buffer
- **Custom Models**: No rate limiting (assumes sufficient quota)

### Logging
Each model gets its own detailed log file:
- `logs/gemini-2.5-pro_test_log.txt`
- `logs/qwen3-235b-a22b_test_log.txt`
- `logs/codestral-22b_test_log.txt`  
- `logs/qwen2.5-coder-32b-instruct_test_log.txt`
- `logs/test_summary.txt` - Overall statistics

### Error Handling
- Individual test failures don't stop the entire suite
- Detailed error logging with stack traces
- Execution time tracking for all tests

## Output

### Individual Log Format
Each model log contains:
```
================================================================================
Model: gemini-2.5-pro
Difficulty: easy
Question: 1
Timestamp: 2024-01-15T10:30:45
Success: true
Execution Time: 12.34s
Prompt: List the known protein-protein interactions for the human protein p53.
================================================================================
Response:
[Model response here]
================================================================================
```

### Summary Report
The summary includes:
- Overall success rates
- Per-model statistics (success rate, average execution time)
- Per-difficulty statistics
- Total test count and timing

## Expected Duration

- **Total Runtime**: 30-60 minutes
- **Per Question**: ~2-5 minutes (4 models × 30-75 seconds each)
- **Rate Limiting Impact**: Gemini adds ~12 seconds between calls

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**
   ```
   Warning: CUSTOM_MODEL_BASE_URL or CUSTOM_MODEL_API_KEY not set
   ```
   Solution: Set the required environment variables

2. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'biomni'
   ```
   Solution: Ensure biomni is installed and PYTHONPATH is set correctly

3. **Rate Limiting**
   ```
   Rate limiting: waiting 13.2s for gemini-2.5-pro
   ```
   This is normal - the system automatically handles Gemini's rate limits

### Monitoring Progress

The script provides real-time progress updates:
```
************************************************************
QUESTION 1 (easy): List the known protein-protein interactions...
************************************************************

============================================================
Testing gemini-2.5-pro on easy question 1
Prompt: List the known protein-protein interactions...
============================================================
Agent created successfully for gemini-2.5-pro
SUCCESS: gemini-2.5-pro completed in 15.23s
```

## Customization

### Adding Models
Edit `comprehensive_model_test.py` and add to the `models` list:
```python
{
    'name': 'your-model-name',
    'type': 'custom',  # or 'gemini'
    'rate_limit': None,  # or number for calls per minute
    'config': {
        'llm': 'custom',
        'custom_model_name': 'your-model-name',
        # ... other config
    }
}
```

### Adding Prompts
Add prompts to the appropriate difficulty file:
- `test_prompts_easy.txt`
- `test_prompts_medium.txt`
- `test_prompts_hard.txt`

One prompt per line, empty lines are ignored.

## Results Analysis

After testing, analyze results using:
1. **Summary file** for quick overview
2. **Individual logs** for detailed model responses
3. **Success rates** to identify model strengths/weaknesses
4. **Execution times** for performance comparison 