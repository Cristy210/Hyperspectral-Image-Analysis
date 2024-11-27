# Clustering Algorithms Implementation on Pavia Dataset
This folder contains the implementation of different clustering algorithms for hyperspectral image analysis, using the widely known **Pavia Dataset**.

## About the Dataset
These are two scenes acquired by the Reflective Optics System Imaging Spectrometer (ROSIS) sensor during a flight campaign over Pavia, nothern Italy. 

### Pavia Center:
- **Spatial Dimensions**: 1096 x 715 pixels
- **Number of Spectral Bands**: 102

### Pavia University:
- **Spatial Dimensions**: 610 x 340 pixes
- **Number of Spectral Bands**: 103

## Groundtruth Classes for the Pavia Centre Scene

| #   | Class                  | Samples |
|-----|------------------------|---------|
| 1   | Water                 | 824     |
| 2   | Trees                 | 820     |
| 3   | Asphalt               | 816     |
| 4   | Self-Blocking Bricks  | 808     |
| 5   | Bitumen               | 808     |
| 6   | Tiles                 | 1260    |
| 7   | Shadows               | 476     |
| 8   | Meadows               | 824     |
| 9   | Bare Soil             | 820     |

## Groundtruth Classes for the Pavia Univeristy Scene

| #   | Class                  | Samples |
|-----|------------------------|---------|
| 1   | Asphalt               | 6631    |
| 2   | Meadows               | 18649   |
| 3   | Gravel                | 2099    |
| 4   | Trees                 | 3064    |
| 5   | Painted metal sheets  | 1345    |
| 6   | Bare Soil             | 5029    |
| 7   | Bitumen               | 1330    |
| 8   | Self-Blocking Bricks  | 3682    |
| 9   | Shadows               | 947     |

## Clustering Algorithms

This project explores unsupervised learning techniques to segment and cluster hyperspectral data from the Pavia University Dataset. The implemented algorithms include:
- **K-means Clustering**
- **Spectral Clustering**
- **K-Subspaces Clustering**

## Data Credits
This Dataset is downloaded from [Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

---

## Clustering Results

#### Spectral Clustering Results:
Below Image shows the clustering results obtained from Spectral Clustering algorithm -- Plotted against the Ground Truth

![Ground Truth Vs Clustering Results](/Clustering%20Results/Pavia/Pavia_GT_Res.png)

#### Confusion Matrix:
Below Confusion Matrix helps evaluate clustering algorithm's  accuracy against ground truth labels
![Confusion Matrix](/Clustering%20Results/Pavia/Confusion_Matrix.png)
