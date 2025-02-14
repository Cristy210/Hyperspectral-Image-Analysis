# Clustering Algorithms Implementation on Pavia Dataset
This folder contains the implementation of different clustering algorithms for hyperspectral image analysis, using the widely known **Salinas Dataset**.

## About the Dataset
The Salinas Dataset was collected using the 224-band AVIRIS sensor over Salinas Valley, California. This dataset is characterized by its high spatial resolution (3.7-meter pixels) and covers an area of 512 rows by 217 columns. After removing 20 water absorption bands, the dataset comprises 204 spectral bands. It is widely used for agricultural analysis as it includes various crops, bare soils, and vineyard fields. The ground truth data contains 16 distinct classes, representing different types of vegetation and soil.

### Salinas Scene:
- **Spatial Dimensions**: 512 x 217 pixels
- **Number of Spectral Bands**: 204

## Groundtruth Classes for the Salinas Dataset

| #   | Class                       | Samples |
|-----|-----------------------------|---------|
| 1   | Broccoli_green_weeds_1      | 2009    |
| 2   | Broccoli_green_weeds_2      | 3726    |
| 3   | Fallow                      | 1976    |
| 4   | Fallow_rough_plow           | 1394    |
| 5   | Fallow_smooth               | 2678    |
| 6   | Stubble                     | 3959    |
| 7   | Celery                      | 3579    |
| 8   | Grapes_untrained            | 11271   |
| 9   | Soil_vineyard_develop       | 6203    |
| 10  | Corn_senesced_green_weeds   | 3278    |
| 11  | Lettuce_romaine_4wk         | 1068    |
| 12  | Lettuce_romaine_5wk         | 1927    |
| 13  | Lettuce_romaine_6wk         | 916     |
| 14  | Lettuce_romaine_7wk         | 1070    |
| 15  | Vineyard_untrained          | 7268    |
| 16  | Vineyard_vertical_trellis   | 1807    |


## Clustering Algorithms

This project explores unsupervised learning techniques to segment and cluster hyperspectral data from the Pavia University Dataset. The implemented algorithms include:
- **K-means Clustering**
- **Spectral Clustering**
- **K-Subspaces Clustering**


## Data Credits
This Dataset is downloaded from [Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

## Clustering Results
#### Spectral Clustering Results:
Below Image shows the clustering results obtained from Spectral Clustering algorithm -- Plotted against the Ground Truth

![Ground Truth Vs Clustering Results](/Clustering%20Results/Salinas/Salinas_GT_CluRes_800.png)

#### Confusion Matrix:
Below Confusion Matrix helps evaluate clustering algorithm's  accuracy against ground truth labels
![Confusion Matrix](/Clustering%20Results/Salinas/Salinas_Conf_Mat.png)
