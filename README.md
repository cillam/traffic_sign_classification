# Traffic Sign Classification
Dataset: https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification

## Project Overview
This project implements a deep learning model for classifying traffic signs, developed as part of an Applied Machine Learning course. The system uses computer vision and neural networks to identify and classify different types of traffic signs from images.

## Technologies Used
- **Python** - Primary programming language
- **TensorFlow/Keras** - Deep learning framework for model development
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations and array operations
- **Matplotlib** - Data visualization
- **Scikit-learn** - Machine learning utilities and model evaluation

## Dataset Overview
### Class Aggregation
To improve model performance and create more meaningful categories, the 58 original classes were consolidated into four main categories:
1. Speed Limit Signs
2. Prohibitive Signs
3. Indicative Signs
4. Warning Signs

This aggregation strategy:
- Groups functionally similar signs together
- Creates more balanced class distribution
- Reflects real-world sign categorization

### Exploratory Analysis
The data exploration included:
- Visual examination of sample images from each class
- Analysis of original and aggregated class distributions
- Verification of image quality and characteristics
- Assessment of potential challenges in classification

## Data Preprocessing
### Image Processing Pipeline
1. **Image Standardization**
   - Resizing all images to 128x128 pixels
   - Aspect ratio preservation during resizing
   - Pixel value normalization (0-1 scaling)

2. **Data Augmentation**
   - Brightness adjustment
   - Contrast enhancement
   - Preservation of original samples alongside augmented versions

3. **Dataset Splitting**
   - 60% Training set
   - 40% Validation set
   - Separate held-out test set

### Advanced Preprocessing
- Implementation of class weights to address class imbalance
- One-hot encoding of categorical labels
- Random shuffling of training data
- Early stopping implementation to prevent overfitting

### Final Dataset Dimensions
- Training set: 5,004 images
- Validation set: 1,668 images
- Test set: 1,994 images
- Image dimensions: 128x128x3 (RGB)

## Model Development
### Baseline Model
Initial approach using a simple CNN architecture:

**Architecture:**
- Initial convolutional layer with 12 filters (4x4 kernel)
- MaxPooling layer for dimensionality reduction
- Dropout layer (30%) for regularization
- Dense output layer with softmax activation

**Training Configuration:**
- Optimizer: Adam (learning rate: 0.01)
- Loss: Categorical Cross-Entropy
- Early stopping monitoring validation accuracy
- 20 maximum epochs

**Baseline Performance:**
- Training Accuracy: 35.09%
- Validation Accuracy: 32.01%
- Early stopping activated after epoch 5

### Enhanced Model (Model 1)
Following the baseline results, a deeper architecture was implemented:

**Architecture Improvements:**
- Two convolutional layers (128 and 64 filters)
- Two max pooling layers for hierarchical feature extraction
- Increased model capacity with 512-unit dense layer
- Maintained dropout (30%) for regularization

**Training Enhancements:**
- Implementation of class weights to address imbalance
- Early stopping with baseline validation threshold
- Same learning rate and optimizer configuration

**Model 1 Performance:**
- Training Accuracy: 23.22%
- Validation Accuracy: 27.04%
- Test Accuracy: 17.95%
- Early stopping activated after epoch 4

**Key Observations:**
- Performance degradation compared to baseline model
- Significant gap between validation and test accuracy
- Possible underfitting despite increased model capacity
- Potential issues with learning rate or optimization

**Architecture Comparison:**
- Parameters: increased from 197K to 33.7M
- Deeper feature extraction capability
- More complex feature hierarchy
- Added intermediate dense layer for better feature combination

### Deep Architecture with Optimized Learning (Model 4)
Combining the deeper architecture with optimized learning rate:

**Architectural Design:**
- Four convolutional layers (128, 64, 64, 32 filters)
- 3x3 kernels throughout
- Four max pooling layers
- Dropout regularization (30%)
- Dense layer with 512 units
- Significantly reduced parameter count (1.18M)

**Training Configuration:**
- Reduced learning rate (0.001)
- Class weight balancing
- Early stopping mechanism
- Best model weights from epoch 6

**Performance:**
- Training Accuracy: 99.84%
- Validation Accuracy: 100.00%
- Test Accuracy: 96.99%

## Results and Discussion

### Model Comparison

| Model     | Parameters | Learning Rate | Train Acc | Val Acc | Test Acc |
|-----------|------------|---------------|-----------|----------|-----------|
| Baseline  | 197K       | 0.01          | 35.09%    | 32.01%   | -         |
| Model 1   | 33.7M      | 0.01          | 23.22%    | 27.04%   | 17.95%    |
| Model 2   | 33.7M      | 0.001         | 99.42%    | 99.46%   | 95.99%    |
| Model 3   | 1.18M      | 0.01          | 23.74%    | 24.34%   | 18.25%    |
| Model 4   | 1.18M      | 0.001         | 99.84%    | 100.00%  | 96.99%    |

### Key Findings
1. **Architectural Insights**
   - Deeper networks with proper learning rates achieve superior performance
   - Smaller kernels (3x3) with more layers outperform larger kernels
   - Parameter efficiency: Model 4 achieved best results with fewer parameters

2. **Learning Rate Impact**
   - 0.001 learning rate crucial for model convergence
   - Higher learning rates (0.01) consistently led to poor performance
   - Learning rate more critical than network architecture

3. **Model Evolution**
   - Baseline established minimum performance threshold
   - Model 2 demonstrated importance of learning rate
   - Model 3 showed potential of deeper architecture
   - Model 4 combined insights for optimal performance

4. **Training Dynamics**
   - Fast convergence in early epochs
   - Excellent generalization despite high accuracy
   - Effective regularization through dropout
   - Class weight balancing crucial for performance
