import plotly.graph_objects as go
import plotly.io as pio

# Parse the data with shortened labels to meet 15 char limit
data = {
  "nodes": [
    {"id": "datasets", "label": "📊 User Data<br>hacktrain.csv<br>hacktest.csv", "type": "input"},
    {"id": "main", "label": "🚀 Main Script<br>example_usage.py", "type": "main"},
    {"id": "validation", "label": "🔍 Data Valid<br>data_valid.py", "type": "processing"},
    {"id": "classifier", "label": "🤖 ML Pipeline<br>ndvi_classifier", "type": "processing"},
    {"id": "visualization", "label": "📈 Visualize<br>visualization.py", "type": "processing"},
    {"id": "preprocessing", "label": "🧹 Preprocess<br>Clean & Prep", "type": "step"},
    {"id": "features", "label": "⚙️ Features<br>Time Patterns", "type": "step"},
    {"id": "training", "label": "🎯 Training<br>Log Regression", "type": "step"},
    {"id": "prediction", "label": "🔮 Predict<br>Land Classes", "type": "step"},
    {"id": "submission", "label": "📝 Output<br>submission.csv", "type": "output"},
    {"id": "plots", "label": "📊 Plots<br>plots/ dir", "type": "output"},
    {"id": "metrics", "label": "📋 Metrics<br>Console Out", "type": "output"}
  ],
  "edges": [
    {"from": "datasets", "to": "main"},
    {"from": "main", "to": "validation"},
    {"from": "main", "to": "classifier"},
    {"from": "main", "to": "visualization"},
    {"from": "classifier", "to": "preprocessing"},
    {"from": "preprocessing", "to": "features"},
    {"from": "features", "to": "training"},
    {"from": "training", "to": "prediction"},
    {"from": "prediction", "to": "submission"},
    {"from": "visualization", "to": "plots"},
    {"from": "training", "to": "metrics"}
  ]
}

# Better spaced positions
positions = {
    "datasets": (0, 8),
    "main": (0, 6.5),
    "validation": (-3.5, 5),
    "classifier": (0, 5),
    "visualization": (3.5, 5),
    "preprocessing": (0, 3.5),
    "features": (0, 2.5),
    "training": (0, 1.5),
    "prediction": (0, 0.5),
    "submission": (-2, -1),
    "plots": (3.5, 3.5),
    "metrics": (2, -1)
}

# High contrast colors as requested
colors = {
    "input": "#1FB8CD",     # Blue for input data
    "main": "#ECEBD5",      # Light green for main scripts  
    "processing": "#FFC185", # Orange for processing modules
    "step": "#FFC185",      # Orange for processing steps
    "output": "#B4413C"     # Red for outputs
}

# Create figure
fig = go.Figure()

# Draw edges first
for edge in data["edges"]:
    from_pos = positions[edge["from"]]
    to_pos = positions[edge["to"]]
    
    fig.add_trace(go.Scatter(
        x=[from_pos[0], to_pos[0]], 
        y=[from_pos[1], to_pos[1]],
        line=dict(width=3, color='#5D878F'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))
    
    # Add arrowheads
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    length = (dx**2 + dy**2)**0.5
    
    if length > 0:
        fig.add_annotation(
            x=to_pos[0], y=to_pos[1],
            ax=to_pos[0] - 0.4 * dx/length, 
            ay=to_pos[1] - 0.4 * dy/length,
            arrowhead=2, arrowsize=2, arrowwidth=3,
            arrowcolor='#5D878F',
            showarrow=True
        )

# Draw nodes as rectangles (boxes) for all nodes
for node in data["nodes"]:
    pos = positions[node["id"]]
    color = colors[node["type"]]
    
    # Create box shape
    fig.add_shape(
        type="rect",
        x0=pos[0]-1, y0=pos[1]-0.5,
        x1=pos[0]+1, y1=pos[1]+0.5,
        fillcolor=color,
        line=dict(color="#13343B", width=3)
    )
    
    # Add text with better formatting and larger font
    fig.add_annotation(
        x=pos[0], y=pos[1],
        text=node["label"],
        showarrow=False,
        font=dict(size=12, color='black'),
        align="center",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0)",
        borderwidth=0
    )

# Update layout with better spacing
fig.update_layout(
    title="NDVI Land Cover Workflow",
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 5]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 9]),
    plot_bgcolor='white'
)

# Save the chart
fig.write_image("ndvi_workflow_chart.png")