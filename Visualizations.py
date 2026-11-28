import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from plotly.subplots import make_subplots

from sklearn.manifold import MDS

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram

from scipy.spatial.distance import pdist



# Correlation Heatmap
def corr_heatmap(correlation):
    """
    This function generates and displays a heatmap of the Spearman's correlation matrix.

    Parameters:
        correlation(pandas.DataFrame): A DataFrame containing the correlation matrix to be visualized.
        
    Returns:
        None: The function displays a heatmap of the correlation matrix and does not return any value.
    """
    plt.figure(figsize = (15, 15))
    ax = sns.heatmap(data = correlation, annot = True, mask = np.triu(np.ones_like(correlation)), cmap = 'RdPu', fmt = '.1')
    ax.set_title("Spearman's Correlation Heatmap", fontdict = {'fontsize': 15})
    plt.show()


# Histogram
def histogram(data, variable_to_plot, title):
    """
    This function generates and displays a histogram for a specified variable.

    Parameters:
        data(pandas.DataFrame): The dataset containing multiple rows and columns.
        variable_to_plot(str): The name of the column for which the histogram will be created.
        title(str): The title for the histogram.
        
    Returns:
        None: The function displays a histogram and does not return any value.
    """
    fig = px.histogram(data_frame = data, x = variable_to_plot, title = title, barmode = 'group')
    fig.update_layout(title = title)
    fig.show()



def plot_pca_explained_variance(pca, threshold=0.85):
    """
    Plot individual and cumulative explained variance.

    Parameters
    ----------
    pca : fitted sklearn PCA
        Fitted PCA model.
    threshold : float
        Target cumulative explained variance.
    """

    individual = pca.explained_variance_ratio_
    cumulative = np.cumsum(individual)

    n_components = np.searchsorted(cumulative, threshold) + 1

    fig, ax1 = plt.subplots(figsize=(12, 5))

    pcs = np.arange(1, len(individual) + 1)

    ax1.bar(pcs, individual * 100, alpha=0.65,
            label="Individual variance")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance (%)")
    ax1.set_xticks(pcs)

    ax2 = ax1.twinx()
    ax2.plot(pcs, cumulative * 100, marker="o",
             label="Cumulative variance")
    ax2.axhline(threshold * 100, linestyle="--",
                color="grey", label=f"{threshold:.0%} threshold")
    ax2.axvline(n_components, linestyle=":", color="red",
                label=f"{n_components} components")

    ax2.set_ylabel("Cumulative explained variance (%)")
    ax2.set_ylim(0, 105)

    ax1.set_title("PCA – Explained variance")
    ax1.legend(loc="upper left")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    print(f"Components needed for {threshold:.0%} variance: {n_components}")
    print(f"Variance explained: {cumulative[n_components-1]:.2%}")

    return n_components


def plot_pca_contributions(pca, feature_names, n_pcs=5, top_n=8):
    """
    Plot the top contributing original features for each PC.

    Contributions are calculated as squared PCA coefficients,
    expressed as percentages of each component's total contribution.

    Parameters
    ----------
    pca : fitted sklearn PCA
        Fitted PCA model.
    feature_names : list
        Original feature names, in PCA input order.
    n_pcs : int
        Number of principal components to display.
    top_n : int
        Number of top features to display per component.

    Returns
    -------
    pandas.DataFrame
        Full contribution matrix, in percentages.
    """

    contributions = pd.DataFrame(
        pca.components_[:n_pcs].T ** 2 * 100,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(n_pcs)]
    )

    fig, axes = plt.subplots(
        n_pcs, 1,
        figsize=(11, 3 * n_pcs)
    )

    if n_pcs == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        pc = f"PC{i+1}"

        top = contributions[pc].nlargest(top_n).sort_values()

        top.plot.barh(ax=ax)
        ax.set_title(
            f"{pc} – Top {top_n} features "
            f"({pca.explained_variance_ratio_[i]:.1%} of total variance)"
        )
        ax.set_xlabel("Contribution to component (%)")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.show()

    return contributions




# Inertia and Silhoutte 
def plot_inertia_and_silhouette(data, k_min = 2, k_max = 15):
    """
    This function generates and displays plots for inertia and silhouette scores over a range of cluster numbers for K-means clustering.

    Parameters:
        data(pandas.DataFrame): The dataset to be clustered.
        k_min(int, optional): The minimum number of clusters to evaluate. Defaults to 2.
        k_max(int, optional): The maximum number of clusters to evaluate. Defaults to 15.

    Returns:
        None: The function displays plots for inertia and silhouette scores and does not return any value
    """
    inertia_values = []
    silhouette_values = []

    k_clusters = range(k_min, k_max + 1)

    for k in k_clusters:
        kmeans = KMeans(n_clusters = k, random_state = 0).fit(data)
        inertia_values.append(kmeans.inertia_)  
        kmeans.predict(data)
        silhouette_values.append(silhouette_score(data, kmeans.labels_, metric = 'euclidean'))  

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(k_clusters, inertia_values, marker = 'o')  
    ax1.set_title('Inertia Plot')
    ax2.plot(k_clusters, silhouette_values, marker = 'o')  
    ax2.set_title('Silhouette Score Plot')
    plt.show()

# Silhouette
def silhoette(data, cluster_labels) -> None:
    
    number_clusters = len(np.unique(cluster_labels))

    silhouette_average = silhouette_score(data, cluster_labels)
    sample_silhouette_values = silhouette_samples(data, cluster_labels)

    fig, ax = plt.subplots()
    fig.set_size_inches(7, 4)

    y_lower = 10
    for i in range(number_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.get_cmap("Spectral")(float(i) / number_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor = color, edgecolor = color, alpha = 0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    ax.axvline(x = silhouette_average, color = "black", linestyle = "--")

    ax.set_yticks([])
    ax.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_title("Silhouette plot for {} clusters".format(number_clusters))

    plt.show()

    print("Silhouette score for {} clusters: {:.4f}".format(number_clusters, silhouette_average))


# elbow plot for DBSCAN epsilon
def kdist_plot(X, dim=None):
    dim = dim or X.shape[1]
    min_samples = 2 * dim
    d, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)   
    plt.plot(np.sort(d[:, -1])[::-1])                                       
    plt.xlabel("points, sorted"); plt.ylabel(f"distance to neighbour k={min_samples} (incl. self)")
    plt.show()
    return min_samples

# Dendrogram
def create_dendrogram(data, method, threshold = None):
    """
    This function generates and displays a dendrogram for hierarchical clustering using the specified linkage method.

    Parameters:
        data(pandas.DataFrame): The dataset to be clustered.
        method(str): The linkage criterion to use. This can be 'ward', 'complete', 'average', or 'single'.
        threshold(float, optional): The threshold to draw a horizontal line in the dendrogram for visualizing the clustering 
                                    cut-off. Default is None.

    Returns:
        None: The function displays a dendrogram and does not return any value.
    """
    clustering_model = AgglomerativeClustering(linkage = method, distance_threshold = 0, n_clusters = None).fit(data)

    fig, ax = plt.subplots()
    plt.title('Hierarchical Clustering Dendrogram')

    sample_counts = np.zeros(clustering_model.children_.shape[0])
    total_samples = len(clustering_model.labels_)

    for index, nodes in enumerate(clustering_model.children_):
        count = 0
        for node in nodes:
            if node < total_samples:
                count += 1 
            else:
                count += sample_counts[node - total_samples]
        sample_counts[index] = count

    linkage_matrix = np.column_stack([clustering_model.children_, clustering_model.distances_, sample_counts]).astype(float)

    dendrogram(linkage_matrix, truncate_mode = 'level', p = 50)

    if threshold is not None:
        plt.axhline(y = threshold, color = 'black', linestyle = '--')

    plt.show()


def mds_embedding(X, n_samples=3000, seed=42):
    """2-D metric MDS of (a random subsample of) X.
    Returns the embedding, the row positions used, and Kruskal stress-1."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), size=min(n_samples, len(X)), replace=False))
    Xs = X.iloc[idx] if hasattr(X, "iloc") else X[idx]

    mds = MDS(n_components=2, n_init=1, max_iter=300, random_state=seed)
    emb = mds.fit_transform(Xs)

    # Kruskal stress-1: sqrt( sum (d_orig - d_2D)^2 / sum d_orig^2 )
    d_orig = pdist(np.asarray(Xs))
    d_2d = pdist(emb)
    stress1 = np.sqrt(((d_orig - d_2d) ** 2).sum() / (d_orig ** 2).sum())
    return emb, idx, stress1

def mds_scatter(embedding, labels, title="MDS", stress=None, cmap_name="Spectral"):
    """Scatter plot of a 2-D MDS embedding coloured by cluster label."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    cmap = plt.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, c in enumerate(classes):
        mask = labels == c
        name = "Noise" if c == -1 else f"Cluster {c}"
        colour = "lightgrey" if c == -1 else cmap(i / max(len(classes) - 1, 1))
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=12, alpha=0.6,
                   color=colour, label=f"{name} (n={mask.sum()})")

    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(title + (f"  |  normalized stress = {stress:.3f}" if stress is not None else ""))
    ax.legend(title="Clusters", markerscale=2)
    plt.tight_layout()
    plt.show()

