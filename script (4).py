# Create requirements.txt file
requirements_text = '''pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
'''

# Save requirements file
with open('requirements.txt', 'w') as f:
    f.write(requirements_text)

print("Created requirements.txt")

# Create command-line interface script
cli_code = '''import argparse
import sys
from ndvi_land_cover_classifier import NDVILandCoverClassifier
from data_validation import validate_data
from visualization import create_visualizations

def main():
    """Command-line interface for NDVI land cover classification"""
    parser = argparse.ArgumentParser(description='NDVI Land Cover Classification')
    parser.add_argument('--train', required=True, help='Training data CSV file')
    parser.add_argument('--test', required=True, help='Test data CSV file')
    parser.add_argument('--output', default='submission.csv', help='Output submission file')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation')
    
    args = parser.parse_args()
    
    print("NDVI Land Cover Classification - Command Line Interface")
    print("=" * 60)
    
    try:
        # Initialize classifier
        classifier = NDVILandCoverClassifier(random_state=42)
        
        # Load training data
        print(f"Loading training data: {args.train}")
        train_data = classifier.load_data(args.train)
        
        if train_data is None:
            print(f"Error: Could not load training data from {args.train}")
            sys.exit(1)
        
        # Validate data
        print("Validating data...")
        validation = validate_data(train_data)
        if not validation['is_valid']:
            print("Data validation failed:")
            for issue in validation['issues']:
                print(f"  - {issue}")
            sys.exit(1)
        
        # Preprocess and train
        print("Preprocessing data...")
        processed_train = classifier.clean_and_preprocess(train_data)
        
        print("Preparing features...")
        X, y = classifier.prepare_features_target(processed_train)
        
        print("Training model...")
        results = classifier.train_model(X, y)
        
        # Load test data and create submission
        print(f"Loading test data: {args.test}")
        test_data = classifier.load_data(args.test)
        
        if test_data is None:
            print(f"Error: Could not load test data from {args.test}")
            sys.exit(1)
        
        print("Creating submission...")
        submission = classifier.create_submission(test_data, args.output)
        
        # Create visualizations unless skipped
        if not args.no_viz:
            print("Creating visualizations...")
            try:
                create_visualizations(train_data, processed_train, classifier, results)
                print("Visualizations saved to 'plots/' directory")
            except Exception as e:
                print(f"Warning: Visualization creation failed: {e}")
        
        print("\\nPipeline completed successfully!")
        print(f"Submission saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

# Save CLI script
with open('run_classification.py', 'w') as f:
    f.write(cli_code)

print("Created run_classification.py")

# Create setup verification script
verify_code = '''import sys
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
    print("\\n🐍 Python Version Check:")
    if not check_python_version():
        all_good = False
    
    # Check dependencies
    print("\\n📦 Dependency Check:")
    missing_deps = check_dependencies()
    if missing_deps:
        all_good = False
        print(f"\\n⚠️  Install missing packages with:")
        print(f"   pip install {' '.join(missing_deps)}")
    
    # Check project files
    print("\\n📁 Project Files Check:")
    missing_files = check_project_files()
    if missing_files:
        all_good = False
    
    # Check data files
    print("\\n📊 Dataset Files Check:")
    data_files = check_data_files()
    
    # Summary
    print("\\n" + "=" * 60)
    if all_good and data_files:
        print("🎉 SETUP COMPLETE! Ready to run classification.")
        print("\\n🚀 To start, run:")
        print("   python example_usage.py")
    elif all_good:
        print("⚠️  Setup mostly complete. Add your dataset files to proceed.")
    else:
        print("❌ Setup incomplete. Please address the issues above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
'''

# Save verification script
with open('verify_setup.py', 'w') as f:
    f.write(verify_code)

print("Created verify_setup.py")