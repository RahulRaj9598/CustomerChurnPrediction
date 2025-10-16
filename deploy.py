#!/usr/bin/env python3
"""
Deployment script for Churn Prediction Dashboard
This script helps set up and run the Streamlit dashboard
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements!")
        return False

def check_files():
    """Check if required files exist"""
    required_files = [
        'churn_dashboard.py',
        'tel_churn.csv',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    else:
        print("✅ All required files found!")
        return True

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Churn Prediction Dashboard...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "churn_dashboard.py"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

def main():
    """Main deployment function"""
    print("🎯 Churn Prediction Dashboard Deployment")
    print("=" * 50)
    
    # Check files
    if not check_files():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Run dashboard
    run_dashboard()

if __name__ == "__main__":
    main()
