<div align="center">
<h1> Hyperspectral Image Analysis </h1>
</div>

**Hyperspectral Imaging** is a powerful technique that captures and analyzes a wide spectrum of light across hundreds of narrow bands. This technology allows for detailed material identification, pattern recognition, and classification across diverse fields, including environmental monitoring, medical imaging, agriculture, art conservation, and more.

This repository demonstrates the application of machine learning algorithms, particularly unsupervised learning techniques, to analyze and interpret hyperspectral datasets. The methods presented provide insights into spectral patterns, material properties, and class segmentation.


📌 **1. Project Introduction** 

This repository implements **unsupervised machine learning algorithms** on hyperspectral datasets (Pavia, Salinas, and Onera Satellite Datasets) to perform clustering and segmentation:
- **K-Means Clustering**
- **K-Subspaces Clustering**
- **Spectral Clustering**

📊 **2. Implemented Algorithms** 

#### K-Means Clustering:
K-Means clustering is one of the widely used clustering algorithms due to its simplicity. K-Means clustering groups data points into k clusters by minimizing the within cluster variance. Each cluster is represented by a centroid, and data points are assigned to the cluster with the nearest centroid. The algorithm works iteratively to assign each data point to the cluster with the nearest centroid (based on **Euclidean distance**) and then updates the centroids as the mean of the points in each cluster. This process repeats until the centroids stabilize or a stopping criterion is met.

<div align="center">
<h4> The objective of K-Means is to minimize the within-cluster sum of sqaured errors. </h4>
</div>
 
$$J = \sum_{j=1}^k \sum_{\mathbf{y}_i \in C_j} \left\| \mathbf{y}_i - \boldsymbol{\mu}_j \right\|_2^2$$ 

**Where:**
- $$k$$: Number of clusters.
- $$y_i$$: Data point $$i$$ in the dataset. 
- $$\mu_j$$: Centroid of cluster $$j$$. 
- $$C_j$$: Set of points assigned to cluster $$j$$.
- $$\| \cdot \|_2$$: Euclidean norm (distance).


#### K-Subspaces Clustering:
K-Subspaces (KSS) clustering extends K-Means by addressing the challenges of clustering high-dimensional data. Instead of relying solely on Euclidean distance, KSS assumes the data lies in a union of low-dimensional subspaces. Each cluster is modeled by a distinct subspace, and points are assigned to the subspace that minimizes the residual from the projections. 

<div align="center">
<h4> The objective of K-Subspaces clustering is to minimize the total reconstruction error. </h4>
</div>

$$J = \sum_{k=1}^{K} \sum_{\mathbf{y}_i \in \mathcal{C}_k} \|\mathbf{y}_i - \mathbf{U}_k \mathbf{y}_i\|_2^2$$

**Where:**
- $$k$$: Number of clusters.
- $$y_i$$: Data point $$i$$ in the dataset. 
- $$\mu_j$$: Centroid of cluster $$j$$. 
- $$C_j$$: Set of points assigned to cluster $$j$$.
- $$U_k$$: Subspace Basis for Cluster $$k$$. 
- $$\| \cdot \|_2$$: Euclidean norm (distance).
