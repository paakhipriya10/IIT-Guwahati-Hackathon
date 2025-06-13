import pandas as pd
import numpy as np

def validate_data(data):
    """
    Comprehensive data validation for NDVI land cover dataset
    Returns dict with validation results and issues found
    """
    issues = []
    is_valid = True

    # Check basic structure
    if data is None or data.empty:
        issues.append("Data is None or empty")
        return {"is_valid": False, "issues": issues}

    print(f"Dataset shape: {data.shape}")

    # Check for required columns
    required_cols = ['ID']
    missing_required = [col for col in required_cols if col not in data.columns]
    if missing_required:
        issues.append(f"Missing required columns: {missing_required}")
        is_valid = False

    # Check for class column (should exist in training data)
    has_class = 'class' in data.columns
    if has_class:
        print("✓ Training data detected (has 'class' column)")
        unique_classes = data['class'].unique()
        print(f"Classes found: {sorted(unique_classes)}")

        # Check for expected classes
        expected_classes = {'Water', 'Impervious', 'Farm', 'Forest', 'Grass', 'Orchard'}
        actual_classes = set(unique_classes)

        missing_classes = expected_classes - actual_classes
        extra_classes = actual_classes - expected_classes

        if missing_classes:
            issues.append(f"Missing expected classes: {missing_classes}")
        if extra_classes:
            issues.append(f"Unexpected classes found: {extra_classes}")

    else:
        print("✓ Test data detected (no 'class' column)")

    # Check for NDVI columns
    ndvi_cols = [col for col in data.columns if col.endswith('_N')]
    ndvi_cols.sort()

    if not ndvi_cols:
        issues.append("No NDVI columns found (should end with '_N')")
        is_valid = False
    else:
        print(f"✓ Found {len(ndvi_cols)} NDVI time-series columns")

        # Check NDVI value ranges
        ndvi_data = data[ndvi_cols]
        min_val = ndvi_data.min().min()
        max_val = ndvi_data.max().max()

        print(f"NDVI value range: {min_val:.3f} to {max_val:.3f}")

        # NDVI should typically be between -1 and 1
        if min_val < -1.5 or max_val > 1.5:
            issues.append(f"NDVI values outside expected range [-1, 1]: [{min_val:.3f}, {max_val:.3f}]")

        # Check for missing values
        missing_count = ndvi_data.isnull().sum().sum()
        missing_pct = (missing_count / (ndvi_data.shape[0] * ndvi_data.shape[1])) * 100

        print(f"Missing NDVI values: {missing_count} ({missing_pct:.1f}%)")

        if missing_pct > 50:
            issues.append(f"High percentage of missing NDVI values: {missing_pct:.1f}%")

    # Check for duplicate IDs
    if 'ID' in data.columns:
        duplicate_ids = data['ID'].duplicated().sum()
        if duplicate_ids > 0:
            issues.append(f"Found {duplicate_ids} duplicate IDs")
            is_valid = False
        else:
            print("✓ All IDs are unique")

    # Check data types
    numeric_cols = ndvi_cols + ['ID']
    for col in numeric_cols:
        if col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                issues.append(f"Column '{col}' should be numeric but is {data[col].dtype}")

    # Summary
    if is_valid and not issues:
        print("✅ All validation checks passed!")
    elif issues:
        is_valid = False
        print(f"⚠️  Found {len(issues)} validation issues")

    return {
        "is_valid": is_valid,
        "issues": issues,
        "has_class": has_class,
        "ndvi_columns": ndvi_cols,
        "n_samples": len(data),
        "n_features": len(ndvi_cols)
    }

def quick_data_summary(data):
    """Print quick summary of dataset"""
    print("\nQuick Data Summary:")
    print("-" * 40)
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)[:5]}..." if len(data.columns) > 5 else f"Columns: {list(data.columns)}")

    if 'class' in data.columns:
        print(f"Class distribution:")
        class_counts = data['class'].value_counts()
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")

    # Show NDVI columns info
    ndvi_cols = [col for col in data.columns if col.endswith('_N')]
    if ndvi_cols:
        print(f"NDVI time series: {len(ndvi_cols)} time points")
        print(f"Time range: {ndvi_cols[0]} to {ndvi_cols[-1]}")

    print("-" * 40)

if __name__ == "__main__":
    # Test validation with sample data
    print("Data validation module loaded successfully!")
    print("Use validate_data(your_dataframe) to validate your dataset.")
