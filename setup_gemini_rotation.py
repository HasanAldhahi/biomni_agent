#!/usr/bin/env python3
"""
Setup script for Gemini API rotation system
"""

import os
import shutil
from pathlib import Path

def setup_environment():
    """Setup the environment file"""
    env_template = Path("env_local_template.txt")
    env_file = Path(".env.local")
    
    if env_template.exists() and not env_file.exists():
        try:
            shutil.copy(env_template, env_file)
            print(f"✅ Created .env.local from template")
            print(f"📝 Please edit .env.local to update your API keys and other settings")
        except Exception as e:
            print(f"❌ Could not create .env.local: {e}")
            print(f"📝 Please manually copy {env_template} to .env.local")
    elif env_file.exists():
        print(f"✅ .env.local already exists")
    else:
        print(f"❌ Template file {env_template} not found")
        return False
    
    return True

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import dotenv
        print("✅ python-dotenv is available")
    except ImportError:
        print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
        print("   (Optional but recommended for automatic .env.local loading)")
    
    return True

def test_rotation_system():
    """Test the rotation system"""
    try:
        from gemini_api_rotation import GeminiApiRotationManager
        
        # Try to create manager (will fail if no keys are set)
        print("\n🔄 Testing API rotation system...")
        
        # Set some dummy keys for testing
        os.environ["GEMINI_API_KEY_1"] = "test_key_1"
        os.environ["GEMINI_API_KEY_2"] = "test_key_2"
        
        manager = GeminiApiRotationManager(log_level="INFO")
        print(f"✅ Rotation manager created with {len(manager.api_keys)} keys")
        
        # Test key retrieval
        key = manager.get_current_api_key()
        print(f"✅ Current API key: {key[:10]}...")
        
        # Test error handling
        manager.handle_api_error(Exception("Rate limit exceeded"), 429)
        print("✅ Error handling works")
        
        # Clean up test environment variables
        del os.environ["GEMINI_API_KEY_1"]
        del os.environ["GEMINI_API_KEY_2"]
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing rotation system: {e}")
        return False

def main():
    print("🚀 Setting up Gemini API Rotation System")
    print("=" * 50)
    
    success = True
    
    # Setup environment
    if not setup_environment():
        success = False
    
    # Check dependencies
    if not check_dependencies():
        success = False
    
    # Test system
    if not test_rotation_system():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Edit .env.local with your actual Gemini API keys")
        print("2. Run your tests with: python tests/agent/comprehensive_model_test.py")
        print("3. The system will automatically rotate keys on 429 errors")
        print("\n💡 Features:")
        print("- Automatic key rotation on rate limits (429 errors)")
        print("- 1-hour cooldown for rate-limited keys")
        print("- Health tracking and statistics")
        print("- Detailed logging and monitoring")
    else:
        print("⚠️  Setup completed with warnings")
        print("Please check the messages above and fix any issues")

if __name__ == "__main__":
    main()



