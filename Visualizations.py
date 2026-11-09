import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram

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