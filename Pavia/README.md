<div align="center">
<h1> Clustering Algorithms Implementation on Pavia Dataset </h1>
</div>

This folder contains the implementation of different clustering algorithms for hyperspectral image analysis, using the widely known **Pavia Dataset**.

## About the Dataset
These are two scenes acquired by the Reflective Optics System Imaging Spectrometer (ROSIS) sensor during a flight campaign over Pavia, nothern Italy. 

### Pavia Center:
- **Spatial Dimensions**: 1096 x 715 pixels
- **Number of Spectral Bands**: 102

### Pavia University:
- **Spatial Dimensions**: 610 x 340 pixes
- **Number of Spectral Bands**: 103

<div align="center">
<h2> Groundtruth Classes for the Pavia Centre Scene </h2>
</div>

<table align="center">
    <thead>
        <tr>
            <th style="text-align:center">#</th>
            <th style="text-align:center">Class</th>
            <th style="text-align:center">Samples</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">1</td>
            <td align="center">Water</td>
            <td align="center">824</td>
        </tr>
        <tr>
            <td align="center">2</td>
            <td align="center">Trees</td>
            <td align="center">820</td>
        </tr>
        <tr>
            <td align="center">3</td>
            <td align="center">Asphalt</td>
            <td align="center">816</td>
        </tr>
         <tr>
            <td align="center">4</td>
            <td align="center">Self-Blocking Bricks</td>
            <td align="center">808</td>
        </tr>
       <tr>
            <td align="center">5</td>
            <td align="center">Bitumen</td>
            <td align="center">808</td>
        </tr>
       <tr>
            <td align="center">6</td>
            <td align="center">Tiles</td>
            <td align="center">1260</td>
        </tr>
       <tr>
            <td align="center">7</td>
            <td align="center">Shadows</td>
            <td align="center">476</td>
        </tr>
       <tr>
            <td align="center">8</td>
            <td align="center">Meadows</td>
            <td align="center">824</td>
        </tr>
       <tr>
            <td align="center">9</td>
            <td align="center">Bare Soil</td>
            <td align="center">820</td>
        </tr>
    </tbody>
</table>

<div align="center">
<h2> Groundtruth Classes for the Pavia Univeristy Scene </h2>
</div>

<table align="center">
    <thead>
        <tr>
            <th style="text-align:center">#</th>
            <th style="text-align:center">Class</th>
            <th style="text-align:center">Samples</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">1</td>
            <td align="center">Asphalt</td>
            <td align="center">6631</td>
        </tr>
        <tr>
            <td align="center">2</td>
            <td align="center">Meadows</td>
            <td align="center">18649</td>
        </tr>
        <tr>
            <td align="center">3</td>
            <td align="center">Gravel</td>
            <td align="center">2099</td>
        </tr>
         <tr>
            <td align="center">4</td>
            <td align="center">Trees</td>
            <td align="center">3064</td>
        </tr>
       <tr>
            <td align="center">5</td>
            <td align="center">Painted metal sheets</td>
            <td align="center">1345</td>
        </tr>
       <tr>
            <td align="center">6</td>
            <td align="center">Bare Soil</td>
            <td align="center">5029</td>
        </tr>
       <tr>
            <td align="center">7</td>
            <td align="center">Bitumen</td>
            <td align="center">1330</td>
        </tr>
       <tr>
            <td align="center">8</td>
            <td align="center">Self-Blocking Bricks</td>
            <td align="center">3682</td>
        </tr>
       <tr>
            <td align="center">9</td>
            <td align="center">Shadows</td>
            <td align="center">947</td>
        </tr>
    </tbody>
</table>


## Clustering Algorithms

This project explores unsupervised learning techniques to segment and cluster hyperspectral data from the Pavia University Dataset. The implemented algorithms include:
- **K-means Clustering**
- **Spectral Clustering**
- **K-Subspaces Clustering**

## Data Credits
This Dataset is downloaded from [Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

---

## Clustering Results

## Below Image shows the clustering results obtained from Spectral Clustering algorithm -- Plotted against the Ground Truth

![Ground Truth Vs Clustering Results](/Clustering%20Results/Pavia/GT_CluRes_Pavia.png)

## Below Confusion Matrix helps evaluate clustering algorithm's  accuracy against ground truth labels
![Confusion Matrix](/Clustering%20Results/Pavia/Conf_Mat_Pavia.png)
