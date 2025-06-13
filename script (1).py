# Create the main example usage file that looks for the user's specific dataset names
example_usage_code = '''import os
import sys
from ndvi_land_cover_classifier import NDVILandCoverClassifier
from data_validation import validate_data
from visualization import create_visualizations

def check_data_files():
    """Check for required data files with user's specific filenames"""
    train_file = None
    test_file = None
    
    # Look for user's specific dataset names first
    if os.path.exists('hacktrain.csv'):
        train_file = 'hacktrain.csv'
        print("✓ Found training data: hacktrain.csv")
    elif os.path.exists('train.csv'):
        train_file = 'train.csv'
        print("✓ Found training data: train.csv")
    else:
        print("❌ Training data not found! Please add 'hacktrain.csv' or 'train.csv' to the project folder.")
        return None, None
        
    if os.path.exists('hacktest.csv'):
        test_file = 'hacktest.csv'
        print("✓ Found test data: hacktest.csv")
    elif os.path.exists('test.csv'):
        test_file = 'test.csv'
        print("✓ Found test data: test.csv")
    else:
        print("❌ Test data not found! Please add 'hacktest.csv' or 'test.csv' to the project folder.")
        return train_file, None
        
    return train_file, test_file

def run_classification_example():
    """Complete NDVI land cover classification pipeline"""
    print("="*60)
    print("🌱 NDVI Land Cover Classification Pipeline")
    print("="*60)
    
    # Check for data files
    train_file, test_file = check_data_files()
    
    if not train_file or not test_file:
        print("\\n❌ Cannot proceed without both training and test data files.")
        print("Please ensure you have:")
        print("   - hacktrain.csv (training data)")
        print("   - hacktest.csv (test data)")
        print("in the same folder as this script.")
        return None, None
    
    try:
        # Initialize classifier
        print("\\n🔧 Initializing classifier...")
        classifier = NDVILandCoverClassifier(random_state=42)
        
        # Load and validate training data
        print(f"\\n📂 Loading training data from {train_file}...")
        train_data = classifier.load_data(train_file)
        
        if train_data is None:
            print("❌ Failed to load training data")
            return None, None
            
        # Validate data structure
        print("\\n🔍 Validating data structure...")
        validation_results = validate_data(train_data)
        
        if not validation_results['is_valid']:
            print("❌ Data validation failed:")
            for issue in validation_results['issues']:
                print(f"   - {issue}")
            return None, None
        
        print("✅ Data validation passed!")
        
        # Preprocess training data
        print("\\n🧹 Preprocessing training data...")
        processed_train = classifier.clean_and_preprocess(train_data)
        
        # Prepare features and target
        print("\\n🎯 Preparing features and target...")
        X, y = classifier.prepare_features_target(processed_train)
        
        print(f"   - Feature matrix shape: {X.shape}")
        print(f"   - Target classes: {len(set(y))} unique classes")
        
        # Train model
        print("\\n🚀 Training logistic regression model...")
        results = classifier.train_model(X, y)
        
        # Load and preprocess test data
        print(f"\\n📊 Loading test data from {test_file}...")
        test_data = classifier.load_data(test_file)
        
        if test_data is None:
            print("❌ Failed to load test data")
            return classifier, results
            
        # Create submission
        print("\\n📝 Creating submission file...")
        submission = classifier.create_submission(test_data, 'submission.csv')
        
        # Create visualizations
        print("\\n📈 Creating visualizations...")
        try:
            create_visualizations(train_data, processed_train, classifier, results)
            print("✅ Visualizations saved to 'plots/' directory")
        except Exception as e:
            print(f"⚠️  Visualization creation failed: {e}")
        
        # Summary
        print("\\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📊 Model Performance Summary:")
        print(f"   - Training Accuracy: {results['train_accuracy']:.1%}")
        print(f"   - Validation Accuracy: {results['val_accuracy']:.1%}")
        print(f"   - Cross-Validation: {results['cv_mean']:.1%} (±{results['cv_std']:.1%})")
        print(f"\\n📁 Generated Files:")
        print(f"   - submission.csv (competition submission)")
        print(f"   - plots/ directory (analysis visualizations)")
        print(f"\\n🎯 Prediction Summary:")
        print(f"   - Total predictions: {len(submission)}")
        print(f"   - Submission file ready for competition!")
        
        return classifier, results
        
    except Exception as e:
        print(f"\\n❌ Error during classification pipeline: {e}")
        print("\\nTroubleshooting tips:")
        print("   1. Ensure your CSV files have the correct format")
        print("   2. Check that NDVI columns end with '_N'")
        print("   3. Verify that 'class' column exists in training data")
        print("   4. Make sure 'ID' column exists in both files")
        return None, None

if __name__ == "__main__":
    print("Starting NDVI Land Cover Classification...")
    print("Looking for your dataset files: hacktrain.csv and hacktest.csv")
    
    classifier, results = run_classification_example()
    
    if classifier and results:
        print("\\n🌟 Ready to submit your predictions!")
        print("Upload 'submission.csv' to the competition platform.")
    else:
        print("\\n❌ Pipeline failed. Please check the error messages above.")
'''

# Save the example usage file
with open('example_usage.py', 'w') as f:
    f.write(example_usage_code)

print("Created example_usage.py")