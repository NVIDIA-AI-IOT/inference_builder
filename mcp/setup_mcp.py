#!/usr/bin/env python3
"""
Setup script for Inference Builder MCP Integration

This script helps install dependencies and configure the MCP integration.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print(f"✗ Command not found: {cmd[0]}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling MCP dependencies...")

    # Try to install MCP
    if not run_command([sys.executable, "-m", "pip", "install", "mcp"], "Installing MCP"):
        print("Failed to install MCP. Please install manually:")
        print("pip install mcp")
        return False

    # Try to install OmegaConf if not already installed
    try:
        import omegaconf
        print("✓ OmegaConf already installed")
    except ImportError:
        if not run_command([sys.executable, "-m", "pip", "install", "omegaconf"], "Installing OmegaConf"):
            print("Failed to install OmegaConf. Please install manually:")
            print("pip install omegaconf")
            return False

    return True

def test_integration():
    """Test the MCP integration"""
    print("\nTesting MCP integration...")

    if not run_command([sys.executable, "mcp/test_mcp_server.py"], "Running integration tests"):
        print("Integration tests failed. Please check the errors above.")
        return False

    return True

def create_cursor_config():
    """Create or update Cursor MCP configuration"""
    print("\nSetting up Cursor configuration...")

    cursor_dir = Path.home() / ".cursor"
    if not cursor_dir.exists():
        cursor_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created Cursor directory: {cursor_dir}")

    config_file = cursor_dir / "mcp.json"

    # Create the correct configuration with absolute paths
    correct_config = {
        "mcpServers": {
            "deepstream-inference-builder": {
                "command": str(Path(sys.executable)),
                "args": [str(Path.cwd() / "mcp"/ "mcp_server.py")],
                "cwd": str(Path.cwd()),
                "env": {
                    "PYTHONPATH": str(Path.cwd() / "mcp")
                }
            }
        }
    }

    import json

    # Check if existing config needs updating
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                existing_config = json.load(f)

            # Check if the config is already correct
            if (existing_config.get("mcpServers", {}).get("deepstream-inference-builder", {}).get("cwd") == str(Path.cwd()) and
                existing_config.get("mcpServers", {}).get("deepstream-inference-builder", {}).get("command") == str(Path(sys.executable))):
                print(f"✓ Cursor MCP config already exists and is correct: {config_file}")
                return True
            else:
                print(f"⚠️  Existing config found but needs updating: {config_file}")
                print("Updating with correct paths...")
        except Exception as e:
            print(f"⚠️  Existing config found but could not be read: {e}")
            print("Creating new configuration...")

    # Create or update the config
    try:
        with open(config_file, 'w') as f:
            json.dump(correct_config, f, indent=2)
        print(f"✓ Created/updated Cursor MCP config: {config_file}")
        print("Note: You may need to restart Cursor for the configuration to take effect.")
        return True
    except Exception as e:
        print(f"✗ Failed to create/update Cursor config: {e}")
        print("Please manually copy cursor-mcp-config.json to your Cursor configuration directory.")
        return False

def main():
    """Main setup function"""
    print("Inference Builder MCP Integration Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Python version incompatible")
        return False

    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed: Dependency installation failed")
        return False

    # Test integration
    if not test_integration():
        print("\n❌ Setup failed: Integration tests failed")
        return False

    # Setup Cursor config
    create_cursor_config()

    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Restart Cursor if you're using it")
    print("2. The MCP server should now be available in Cursor")
    print("3. Try using the tools:")
    print("   - 'Show me what sample configurations are available'")
    print("   - 'Generate a DeepStream object detection pipeline'")
    print("\nFor more information, see README-MCP.md")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

