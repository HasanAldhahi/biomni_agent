#!/usr/bin/env python3
"""
Gemini API Key Rotation Manager
Handles automatic rotation of Gemini API keys when encountering 429 rate limit errors.
"""

import os
import time
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import random

# Try to import dotenv to load .env.local file
try:
    from dotenv import load_dotenv
    load_dotenv('.env.local')
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables should be set manually.")


@dataclass
class ApiKeyStatus:
    """Track the status of an API key"""
    key: str
    is_active: bool = True
    last_error_time: Optional[datetime] = None
    error_count: int = 0
    cooldown_until: Optional[datetime] = None
    total_requests: int = 0
    successful_requests: int = 0


class GeminiApiRotationManager:
    """
    Manages rotation of Gemini API keys with automatic fallback on 429 errors.
    
    Features:
    - Automatic key rotation on 429 rate limit errors
    - Cooldown periods for rate-limited keys
    - Health tracking for each key
    - Configurable retry logic
    - Logging and monitoring
    """
    
    def __init__(self, 
                 cooldown_minutes: int = 60,
                 max_retries_per_key: int = 3,
                 retry_delay_seconds: int = 5,
                 log_level: str = "INFO"):
        """
        Initialize the rotation manager.
        
        Args:
            cooldown_minutes: Minutes to wait before retrying a rate-limited key
            max_retries_per_key: Maximum retries per key before marking as inactive
            retry_delay_seconds: Seconds to wait between retries
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.cooldown_minutes = cooldown_minutes
        self.max_retries_per_key = max_retries_per_key
        self.retry_delay_seconds = retry_delay_seconds
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load API keys from environment
        self.api_keys = self._load_api_keys()
        if not self.api_keys:
            raise ValueError("No Gemini API keys found in environment variables!")
        
        self.logger.info(f"Initialized with {len(self.api_keys)} API keys")
        
        # Track current key index and statistics
        self.current_key_index = 0
        self.total_requests = 0
        self.total_errors = 0
        self.rotation_count = 0
        self.start_time = datetime.now()
    
    def _load_api_keys(self) -> List[ApiKeyStatus]:
        """Load API keys from environment variables"""
        keys = []
        for i in range(1, 9):  # GEMINI_API_KEY_1 through GEMINI_API_KEY_8
            key_name = f"GEMINI_API_KEY_{i}"
            key_value = os.environ.get(key_name)
            if key_value:
                keys.append(ApiKeyStatus(key=key_value))
                self.logger.debug(f"Loaded {key_name}: {key_value[:10]}...")
        
        if not keys:
            self.logger.warning("No API keys found, checking for legacy GEMINI_API_KEY")
            legacy_key = os.environ.get("GEMINI_API_KEY")
            if legacy_key:
                keys.append(ApiKeyStatus(key=legacy_key))
                self.logger.info("Using legacy GEMINI_API_KEY")
        
        return keys
    
    def get_current_api_key(self) -> str:
        """Get the current active API key"""
        # Clean up expired cooldowns
        self._cleanup_cooldowns()
        
        # Find the next available key
        available_keys = [i for i, key_status in enumerate(self.api_keys) 
                         if key_status.is_active and 
                         (key_status.cooldown_until is None or 
                          key_status.cooldown_until <= datetime.now())]
        
        if not available_keys:
            self.logger.warning("No available API keys! Resetting all cooldowns...")
            self._reset_all_cooldowns()
            available_keys = [i for i, key_status in enumerate(self.api_keys) 
                             if key_status.is_active]
        
        if not available_keys:
            raise RuntimeError("All API keys are inactive!")
        
        # Use round-robin with some randomization to distribute load
        if len(available_keys) > 1:
            # Prefer keys with fewer errors
            available_keys.sort(key=lambda i: self.api_keys[i].error_count)
            # Add some randomization among the best keys
            best_error_count = self.api_keys[available_keys[0]].error_count
            best_keys = [i for i in available_keys 
                        if self.api_keys[i].error_count == best_error_count]
            self.current_key_index = random.choice(best_keys)
        else:
            self.current_key_index = available_keys[0]
        
        current_key = self.api_keys[self.current_key_index].key
        self.logger.debug(f"Using API key #{self.current_key_index + 1}: {current_key[:10]}...")
        return current_key
    
    def handle_api_error(self, error: Exception, response_code: Optional[int] = None) -> bool:
        """
        Handle API errors and decide whether to rotate keys.
        
        Args:
            error: The exception that occurred
            response_code: HTTP response code if available
            
        Returns:
            bool: True if key was rotated and retry should be attempted, False otherwise
        """
        self.total_errors += 1
        current_key_status = self.api_keys[self.current_key_index]
        current_key_status.error_count += 1
        current_key_status.last_error_time = datetime.now()
        
        error_str = str(error).lower()
        is_rate_limit_error = (
            response_code == 429 or
            "rate limit" in error_str or
            "quota exceeded" in error_str or
            "too many requests" in error_str or
            "resource has been exhausted" in error_str
        )
        
        if is_rate_limit_error:
            self.logger.warning(f"Rate limit error detected for key #{self.current_key_index + 1}")
            self._handle_rate_limit_error()
            return True
        else:
            self.logger.error(f"Non-rate-limit error for key #{self.current_key_index + 1}: {error}")
            # For non-rate-limit errors, still try rotating if we have other keys
            if len([k for k in self.api_keys if k.is_active]) > 1:
                self._rotate_to_next_key()
                return True
            return False
    
    def _handle_rate_limit_error(self):
        """Handle rate limit error by setting cooldown and rotating"""
        current_key_status = self.api_keys[self.current_key_index]
        
        # Set cooldown period
        current_key_status.cooldown_until = datetime.now() + timedelta(minutes=self.cooldown_minutes)
        
        # If this key has too many errors, mark it as inactive
        if current_key_status.error_count >= self.max_retries_per_key:
            current_key_status.is_active = False
            self.logger.warning(f"Deactivating key #{self.current_key_index + 1} due to repeated errors")
        
        self.logger.info(f"Key #{self.current_key_index + 1} in cooldown until {current_key_status.cooldown_until}")
        self._rotate_to_next_key()
    
    def _rotate_to_next_key(self):
        """Rotate to the next available key"""
        self.rotation_count += 1
        old_index = self.current_key_index
        
        # This will be handled in get_current_api_key()
        self.logger.info(f"Rotating from key #{old_index + 1} to next available key")
    
    def _cleanup_cooldowns(self):
        """Remove expired cooldowns"""
        now = datetime.now()
        for key_status in self.api_keys:
            if key_status.cooldown_until and key_status.cooldown_until <= now:
                key_status.cooldown_until = None
    
    def _reset_all_cooldowns(self):
        """Reset all cooldowns (emergency measure)"""
        for key_status in self.api_keys:
            key_status.cooldown_until = None
            if key_status.error_count < self.max_retries_per_key:
                key_status.is_active = True
        self.logger.warning("Reset all cooldowns - emergency measure activated")
    
    def record_successful_request(self):
        """Record a successful API request"""
        self.total_requests += 1
        self.api_keys[self.current_key_index].total_requests += 1
        self.api_keys[self.current_key_index].successful_requests += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about API key usage"""
        active_keys = sum(1 for k in self.api_keys if k.is_active)
        cooldown_keys = sum(1 for k in self.api_keys 
                           if k.cooldown_until and k.cooldown_until > datetime.now())
        
        uptime = datetime.now() - self.start_time
        success_rate = (self.total_requests - self.total_errors) / max(self.total_requests, 1) * 100
        
        key_stats = []
        for i, key_status in enumerate(self.api_keys):
            key_success_rate = (key_status.successful_requests / 
                              max(key_status.total_requests, 1) * 100)
            key_stats.append({
                'key_number': i + 1,
                'key_preview': key_status.key[:10] + "...",
                'is_active': key_status.is_active,
                'total_requests': key_status.total_requests,
                'successful_requests': key_status.successful_requests,
                'error_count': key_status.error_count,
                'success_rate': round(key_success_rate, 2),
                'in_cooldown': key_status.cooldown_until is not None and key_status.cooldown_until > datetime.now(),
                'cooldown_until': key_status.cooldown_until.isoformat() if key_status.cooldown_until else None,
                'last_error': key_status.last_error_time.isoformat() if key_status.last_error_time else None
            })
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'total_keys': len(self.api_keys),
            'active_keys': active_keys,
            'keys_in_cooldown': cooldown_keys,
            'current_key_index': self.current_key_index + 1,
            'total_requests': self.total_requests,
            'total_errors': self.total_errors,
            'success_rate_percent': round(success_rate, 2),
            'rotation_count': self.rotation_count,
            'key_statistics': key_stats
        }
    
    def print_status(self):
        """Print current status to console"""
        stats = self.get_statistics()
        print(f"\n{'='*60}")
        print(f"GEMINI API KEY ROTATION STATUS")
        print(f"{'='*60}")
        print(f"Uptime: {stats['uptime_seconds']:.0f}s")
        print(f"Total Keys: {stats['total_keys']}")
        print(f"Active Keys: {stats['active_keys']}")
        print(f"Keys in Cooldown: {stats['keys_in_cooldown']}")
        print(f"Current Key: #{stats['current_key_index']}")
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Success Rate: {stats['success_rate_percent']}%")
        print(f"Rotations: {stats['rotation_count']}")
        
        print(f"\nKEY DETAILS:")
        for key_stat in stats['key_statistics']:
            status_emoji = "✅" if key_stat['is_active'] else "❌"
            cooldown_emoji = "⏳" if key_stat['in_cooldown'] else "🟢"
            print(f"  Key #{key_stat['key_number']}: {status_emoji} {cooldown_emoji} "
                  f"Reqs: {key_stat['total_requests']} "
                  f"Success: {key_stat['success_rate']}% "
                  f"Errors: {key_stat['error_count']}")
        print(f"{'='*60}\n")


# Convenience function for easy integration
def create_gemini_rotation_manager(**kwargs) -> GeminiApiRotationManager:
    """Create and return a configured GeminiApiRotationManager"""
    return GeminiApiRotationManager(**kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test the rotation manager
    manager = create_gemini_rotation_manager(cooldown_minutes=1, log_level="DEBUG")
    
    print("Testing API key rotation...")
    for i in range(5):
        try:
            key = manager.get_current_api_key()
            print(f"Test {i+1}: Got key {key[:10]}...")
            
            # Simulate a rate limit error every other request
            if i % 2 == 1:
                manager.handle_api_error(Exception("Rate limit exceeded"), 429)
            else:
                manager.record_successful_request()
                
            time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
    
    manager.print_status()



