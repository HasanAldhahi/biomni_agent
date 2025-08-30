# Gemini API Key Rotation System

This system provides automatic API key rotation for Gemini Pro 2.5 with intelligent fallback handling for 429 rate limit errors.

## Features

- **Automatic Key Rotation**: Seamlessly switches between multiple API keys when rate limits are hit
- **Smart Error Detection**: Detects 429 errors and other rate limit indicators
- **Cooldown Management**: Implements configurable cooldown periods for rate-limited keys
- **Health Tracking**: Monitors success rates and error counts for each key
- **Retry Logic**: Automatic retries with exponential backoff
- **Detailed Logging**: Comprehensive logging and statistics tracking
- **Environment Integration**: Loads keys from `.env.local` file

## Quick Setup

1. **Copy the environment template**:
   ```bash
   cp env_local_template.txt .env.local
   ```

2. **Edit `.env.local`** with your actual API keys:
   ```bash
   # Gemini Pro 2.5 API Keys for Rotation
   GEMINI_API_KEY_1=your_first_api_key_here
   GEMINI_API_KEY_2=your_second_api_key_here
   # ... up to GEMINI_API_KEY_8
   ```

3. **Run the setup script**:
   ```bash
   python setup_gemini_rotation.py
   ```

4. **Install optional dependency** (recommended):
   ```bash
   pip install python-dotenv
   ```

## Usage

The rotation system is automatically integrated into `comprehensive_model_test.py`. Just run your tests normally:

```bash
python tests/agent/comprehensive_model_test.py
```

## API Keys Configuration

Add your Gemini API keys to `.env.local`:

```bash
# Primary key
GEMINI_API_KEY_1=XXXXXX
# Backup keys
GEMINI_API_KEY_2=XXXXXX
GEMINI_API_KEY_3=XXXXXX
# ... add up to 8 keys
```

## How It Works

1. **Normal Operation**: Uses the first available API key
2. **Rate Limit Detection**: When a 429 error occurs, the system:
   - Marks the current key for cooldown (default: 1 hour)
   - Rotates to the next available key
   - Retries the failed request
3. **Smart Recovery**: Keys automatically become available again after cooldown
4. **Load Balancing**: Distributes requests across healthy keys

## Configuration Options

The rotation manager can be configured with these parameters:

- `cooldown_minutes`: How long to wait before retrying a rate-limited key (default: 60)
- `max_retries_per_key`: Maximum retries before marking a key as inactive (default: 3)
- `retry_delay_seconds`: Delay between retries (default: 5)
- `log_level`: Logging verbosity (default: "INFO")

## Error Handling

The system handles these error types:
- **429 Rate Limit Exceeded**: Automatic key rotation
- **Quota Exceeded**: Automatic key rotation  
- **Resource Exhausted**: Automatic key rotation
- **Other API Errors**: Logs error and may rotate based on availability

## Monitoring and Statistics

The system tracks detailed statistics:
- Total requests and success rates
- Per-key usage and error counts
- Rotation events and cooldown status
- Real-time health monitoring

View statistics during execution or at the end of test runs.

## Files

- `gemini_api_rotation.py`: Core rotation manager
- `env_local_template.txt`: Environment template
- `setup_gemini_rotation.py`: Setup and testing script
- `comprehensive_model_test.py`: Updated test script with rotation

## Troubleshooting

### No API Keys Found
- Check that `.env.local` exists and contains `GEMINI_API_KEY_1` through `GEMINI_API_KEY_8`
- Ensure keys are valid and not quoted

### All Keys Rate Limited
- The system will reset cooldowns as an emergency measure
- Consider adding more API keys or increasing cooldown time

### Import Errors
- Make sure you're running from the project root directory
- Check that all files are in the correct locations

### Python-dotenv Not Found
- Install with: `pip install python-dotenv`
- Or set environment variables manually before running

## Example Output

```
🔄 Initialized Gemini API rotation with 8 keys
✅ Using API key #1: AIzaSyDfLl...
⚠️  Rate limit error detected for key #1
🔄 Rotating from key #1 to next available key
✅ Will retry with rotated API key...
✅ Using API key #2: AIzaSyDtia...

==============================================================
FINAL GEMINI API ROTATION STATISTICS
==============================================================
Uptime: 1847s
Total Keys: 8
Active Keys: 8
Keys in Cooldown: 2
Current Key: #3
Total Requests: 156
Success Rate: 94.2%
Rotations: 12
```

## Advanced Usage

For custom integration, you can use the rotation manager directly:

```python
from gemini_api_rotation import GeminiApiRotationManager

# Create manager
manager = GeminiApiRotationManager(cooldown_minutes=30)

# Get current key
api_key = manager.get_current_api_key()

# Handle errors
try:
    # Your API call here
    pass
except Exception as e:
    should_retry = manager.handle_api_error(e, response_code=429)
    if should_retry:
        # Retry with new key
        api_key = manager.get_current_api_key()

# Record success
manager.record_successful_request()

# Get statistics
stats = manager.get_statistics()
```

