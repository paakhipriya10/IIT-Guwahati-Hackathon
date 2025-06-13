import argparse
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

        print("\nPipeline completed successfully!")
        print(f"Submission saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
