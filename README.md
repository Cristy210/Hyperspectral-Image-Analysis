<div align="center">
<h1> Hyperspectral Image Analysis </h1>
</div>

**Hyperspectral Imaging** is a powerful technique that captures and analyzes a wide spectrum of light across hundreds of narrow bands. This technology allows for detailed material identification, pattern recognition, and classification across diverse fields, including environmental monitoring, medical imaging, agriculture, art conservation, and more.

This repository demonstrates the application of machine learning algorithms, particularly unsupervised learning techniques, to analyze and interpret hyperspectral datasets. The methods presented provide insights into spectral patterns, material properties, and class segmentation.


📌 **1. Project Introduction** 

This repository implements both **unsupervised and supervised machine learning algorithms** on hyperspectral datasets (Pavia, Salinas, and Onera Satellite Datasets) to perform clustering and segmentation:
- **K-Means Clustering**
- **K-Subspaces Clustering**
- **Spectral Clustering**

📊 **2. Implemented Algorithms** 

#### K-Means Clustering:
K-Means clustering is one of the widely used clustering algorithms due to its simplicity. K-Means clustering groups data points into k clusters by minimizing the within cluster variance. Each cluster is represented by a centroid, and data points are assigned to the cluster with the nearest centroid. The algorithm works iteratively to assign each data point to the cluster with the nearest centroid (based on **Euclidean distance**) and then updates the centroids as the mean of the points in each cluster. This process repeats until the centroids stabilize or a stopping criterion is met.

<div align="center">
<h4> The objective of K-Means is to minimize the within-cluster sum of sqaured errors. </h4>
</div>
 
$$J = \sum_{k=1}^K \sum_{\mathbf{y}_i \in C_j} \left\| \mathbf{y}_i - \boldsymbol{\mu}_j \right\|_2^2$$ 

**Where:**
- $K$: Number of clusters.
- $y_i$: Data point at index $i$ in the dataset. 
- $\mu_k$: Centroid of cluster $k$. 
- $C_k$: Set of points assigned to cluster $k$.
- $\| \cdot \|_2$: Euclidean norm (distance).


#### K-Subspaces Clustering:
K-Subspaces (KSS) clustering extends K-Means by addressing the challenges of clustering high-dimensional data. Instead of relying solely on Euclidean distance, KSS assumes the data lies in a union of low-dimensional subspaces. Clustering is accomplished through a iterative process where subspace space basis are updated and data points are assigned based on their projection onto the closest subspace.   

<div align="center">
<h4> The objective of K-Subspaces clustering is to minimize the total projection error. </h4>
</div>

$$J = \sum_{k=1}^{K} \sum_{\mathbf{y}_i \in \mathcal{C}_k} \|\mathbf{y}_i - \mathbf{U}_k \mathbf{U}_k^{\top} \mathbf{y}_i\|_2^2$$

**Where:**
- $K$: Number of clusters.
- $y_i$: Data point at index $i$ in the dataset.  
- $C_k$: Set of points assigned to cluster $k$.
- $U_k$: Subspace Basis for Cluster $k$. 
- $\| \cdot \|_2$: Euclidean norm (distance).

#### K-Affine spaces Clustering:
K-Affine spaces (KAS) clustering. Just like KSS, KAS assumes that the data lies in a union of low-dimensional affine spaces formed from a set of bases and a bias vector rather than a set of linear subspaces. Clustering is accomplished through a iterative process where the basis and the bias vector for each affine space are updated, and the data points are assigned based on their projection onto the closest affine space.  

<div align="center">
<h4> The objective of K-Affine spaces clustering is to minimize the total projection error. </h4>
</div>

$$J = \sum_{k=1}^{K} \sum_{\mathbf{y}_i \in \mathcal{C}_k} \|\mathbf{y}_i - [\mathbf{U}_k \mathbf{U}_k^{\top} (\mathbf{y}_i - \boldsymbol{\mu}_k) + \boldsymbol{\mu}_k]\|_2^2$$

**Where:**
- $K$: Number of clusters.
- $y_i$: Data point at index $i$ in the dataset. 
- $\mu_k$: Centroid of cluster $k$. 
- $C_k$: Set of points assigned to cluster $k$.
- $U_k$: Affine space Basis for Cluster $k$. 
- $\| \cdot \|_2$: Euclidean norm (distance).

#### Thresholded subspace Clustering:
In Thresholded Subspace Clustering (TSC) algorithm, data points are treated as nodes in a graph, which are then clustered using techniques from spectral graph theory. TSC algorithm is made up of three important matrices.
Adjaceny Matrix ($A$) defines similarity between any two nodes in the dataset; Degree Matrix ($D$) represents the sum of the weights of all edges connected to a node; Laplacian Matrix($L$) captures the structure of the graph by combining information from the above two matrices. K-Means clustering is applied on the eigenvectors corresponding to the $K$ smallest eigenvalues of $L_{sym}$. 


$$L_{sym} = I - D^{-1/2}AD^{-1/2}$$

**where:**
- $I$: Identity matrix of size $MN \times MN$
- $MN$: Total number of data points
- $A$: $A = Z + Z^{\top}$; Z = thresholded version of $C$
$$C_{ij} = \exp \left[ -2 \cdot \arccos \left( \frac{\mathbf{y}_i^{\top} \mathbf{y}_j}{\|\mathbf{y}_i\|_2 \cdot \|\mathbf{y}_j\|_2} \right) \right], \quad \text{for } i,j = 1, \dots, MN.$$
- $D$ = $\text{diag}(d), d_i = \sum_{j=1}^{MN}A_{ij} \quad \text{for } i = 1, \dots, MN$

#### Mean-based Classification:
Mean-based classification classifies data points based on the nearest centroid, where each centroid ($\boldsymbol{\mu}_k$) is the mean of the data points in the corresponding class ($k$).

$$ \boldsymbol{\mu}_k = \frac{1}{N_k} y_n^{(k)} \in \mathbb{R}^{L}, \quad \text{for } k = 1, \dots, K$$

**where:**
- $K$: Number of Classes
- $N_k$: Number of data points in class $k$. 
-  $y_n^{(k)}$: $n^{\text{th}}$ training data point in class $k$. 

$$\text{classify}(y) = \text{argmin}_{k = \{1, \dots, K\}} \| y - \boldsymbol{\mu}_k\|_2^2$$


#### Subspace-based Classification:
Subspace-based classification classifies data points based on the closest projection onto the defined low-dimensional subspace, where each subspace corresponds to a class. subspace basis ($\mathbf{U}$) are formed by the number of left singular vectors from training data specified by the dimensions ($dim$) intitially chosen for each class ($k$). 

$$\mathbf{U}_k^{dim_k} = \mathbf{\hat{U}}[:, 1:dim_k] \quad \text{where } \mathcal{Y}_k = \mathbf{\hat{U}}\mathbf{\hat{\Sigma}}\mathbf{\hat{V}}^{\top} \text{ is an SVD, } \quad \text{for } k = 1, \dots, K$$

**where:**
- $\mathbf{U}_k^{dim_k}:$ Subspace basis corresponding to the class $k$ with dimensions $dim_k$.
- $\mathcal{Y}_k \in \mathbb{R}^{L \times N_k}:$  data matrix for the selected $N_k$ training data points in class $k$; $L$ refers to the original feature space dimensions data points are in. 

$$\text{classify}(\mathbf{y}) = \text{argmin}_{k \in \{1,\dots,K\}}\; \|\mathbf{y}\|_2^2 - \|(\textbf{U}^{\text{dim}_k}_k)^{\top} \mathbf{y}\|_2^2$$

#### Affine space-based Classification:
Affine space-based classification classifies data points based on the closest projection onto the defined low-dimensional closest affine space, where each class is represented in an affine space formed from a set of basis ($\mathbf{U_k}$) and a centroid ($\boldsymbol{\mu}_k$) which acts as the center for the affine space.

$$\boldsymbol{\mu}_k = \frac{1}{N_k} y_n^{(k)} \in \mathbb{R}^{L}$$
$$\mathbf{U}_k^{dim_k} = \mathbf{\hat{U}}[:, 1:dim_k] \quad \text{where } \mathcal{Y}_k - \boldsymbol{\mu}_k1_{N_k}^{\top} = \mathbf{\hat{U}}\mathbf{\hat{\Sigma}}\mathbf{\hat{V}}^{\top} \text{ is an SVD, } \quad \text{for } k = 1, \dots, K$$

**where:**
- $\mathbf{U}_k^{dim_k}:$ Affine space basis corresponding to the class $k$ with dimensions $dim_k$.
- $\boldsymbol{\mu}_k:$ Center of the affine space aka the mean vector formed by the mean of the data points in the corresponding class $k$. 


$$\text{classify}(\mathbf{y}) = \text{argmin}_{k \in \{1,\dots,K\}} \; \|\mathbf{y} - [(\textbf{U}_k^{dim_k})(\textbf{U}^{\text{dim}_k}_k)^{\top} (\mathbf{y} - \boldsymbol{\mu}_k) + \boldsymbol{\mu}_k]\|_2^2$$





