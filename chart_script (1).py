import plotly.graph_objects as go
import pandas as pd

# Data for the table with full names and descriptions
data = {
    'File Name': [
        'ndvi_land_cover_classifier.py',
        'example_usage.py', 
        'data_validation.py',
        'visualization.py',
        'run_classification.py',
        'verify_setup.py',
        'requirements.txt',
        'README.md'
    ],
    'Purpose': [
        'Core classification engine with preprocessing and ML pipeline',
        'Primary VSCode execution script with automatic file detection',
        'Data quality validation and structure checking', 
        'Plotting and analysis visualizations',
        'Command-line interface with argument parsing',
        'Environment and dependency verification',
        'Python package dependencies list',
        'Complete project documentation and setup guide'
    ],
    'Type': [
        'Core Engine',
        'Main Script',
        'Utility',
        'Utility', 
        'Utility',
        'Utility',
        'Documentation',
        'Documentation'
    ]
}

# Create the table with better formatting
fig = go.Figure(data=[go.Table(
    columnwidth=[200, 400, 120],  # Adjust column widths
    header=dict(
        values=['<b>File Name</b>', '<b>Purpose</b>', '<b>Type</b>'],
        fill_color='#1FB8CD',
        font=dict(color='white', size=14),
        align='left',
        height=40
    ),
    cells=dict(
        values=[data['File Name'], data['Purpose'], data['Type']],
        fill_color=[['#ECEBD5', '#FFC185'] * 4],  # Alternating row colors
        font=dict(color='black', size=11),
        align='left',
        height=45
    )
)])

# Update layout
fig.update_layout(
    title='NDVI Land Cover Classification Files'
)

# Save the chart
fig.write_image('ndvi_project_files_table.png')