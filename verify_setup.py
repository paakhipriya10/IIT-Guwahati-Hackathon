import sys
import os
import importlib

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.7+ required.")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'scipy'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            print(f"✅ {package} - Installed")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)

    return missing_packages

def check_project_files():
    """Check if all project files are present"""
    required_files = [
        'ndvi_land_cover_classifier.py',
        'example_usage.py',
        'data_validation.py',
        'visualization.py',
        'run_classification.py',
        'requirements.txt'
    ]

    missing_files = []

    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - Found")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)

    return missing_files

def check_data_files():
    """Check for dataset files"""
    data_files = ['hacktrain.csv', 'hacktest.csv', 'train.csv', 'test.csv']
    found_files = []

    for file in data_files:
        if os.path.exists(file):
            print(f"✅ {file} - Found")
            found_files.append(file)

    if not found_files:
        print("❌ No dataset files found")
        print("   Please add your dataset files:")
        print("   - hacktrain.csv (training data)")
        print("   - hacktest.csv (test data)")

    return found_files

def main():
    """Run complete setup verification"""
    print("NDVI Land Cover Classification - Setup Verification")
    print("=" * 60)

    all_good = True

    # Check Python version
    print("\n🐍 Python Version Check:")
    if not check_python_version():
        all_good = False

    # Check dependencies
    print("\n📦 Dependency Check:")
    missing_deps = check_dependencies()
    if missing_deps:
        all_good = False
        print(f"\n⚠️  Install missing packages with:")
        print(f"   pip install {' '.join(missing_deps)}")

    # Check project files
    print("\n📁 Project Files Check:")
    missing_files = check_project_files()
    if missing_files:
        all_good = False

    # Check data files
    print("\n📊 Dataset Files Check:")
    data_files = check_data_files()

    # Summary
    print("\n" + "=" * 60)
    if all_good and data_files:
        print("🎉 SETUP COMPLETE! Ready to run classification.")
        print("\n🚀 To start, run:")
        print("   python example_usage.py")
    elif all_good:
        print("⚠️  Setup mostly complete. Add your dataset files to proceed.")
    else:
        print("❌ Setup incomplete. Please address the issues above.")

    print("=" * 60)

if __name__ == "__main__":
    main()
