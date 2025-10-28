## Overview
This project implements the K-Means clustering algorithm from scratch and applies it to the MNIST handwritten digits dataset. The implementation includes clustering visualization and performance evaluation using unsupervised clustering accuracy metrics.

## Features
- **Custom K-Means Implementation**: Pure Python/NumPy implementation without external libraries
- **MNIST Data Processing**: Handles MNIST dataset loading and preprocessing
- **PCA Dimensionality Reduction**: Projects high-dimensional image data to 2D for visualization
- **Performance Evaluation**: Implements unsupervised clustering accuracy metric
- **Multiple Initializations**: Runs K-Means with different random seeds for robust results
- **Visualization**: Plots clustering results with color-coded labels

## Dataset
The project uses a subset of the MNIST dataset containing digits 0, 1, 2, and 3 with 1000 total samples. Images are normalized and projected to 2D using Principal Component Analysis (PCA).

## Implementation Details

### K-Means Algorithm
The custom K-Means implementation includes:
- Random centroid initialization
- Euclidean distance calculation
- Iterative centroid updates
- Convergence checking
- Multiple random initializations for better results

### Key Functions
- `kmeans()`: Main K-Means clustering implementation
- `load_mnist()`: MNIST data loading and preprocessing
- `accuracy_score()`: Clustering accuracy evaluation
- `plot_with_colors()`: Visualization of clustering results

## Results
The implementation achieves clustering accuracy of approximately 74.5% on the MNIST subset across multiple random initializations, demonstrating the effectiveness of the custom K-Means algorithm.

## Usage

### Running the Code
```python
# Load and preprocess data
X, y = load_mnist()

# Apply K-Means clustering
labels = kmeans(X2d, k=4, max_iter=500, random_state=0)

# Evaluate clustering accuracy
accuracy = accuracy_score(y, labels)
print(f"Clustering Accuracy: {accuracy:.4f}")
