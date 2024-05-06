import matplotlib.pyplot as plt

from dataset_loader import DatasetLoader
from kmeans_clustering import KMeansClustering
from scaler import Scaler
from spark_session_provider import SparkSessionProvider
from vectorizer import Vectorizer


def plot_silhouette_scores(scores, k_search_range):
    plt.plot(k_search_range, scores)
    plt.xlabel('k')
    plt.ylabel('silhouette score')
    plt.title('Silhouette Score')
    plt.show()


def main():
    spark = SparkSessionProvider().provide_session()
    dataset_loader = DatasetLoader(spark_session=spark)
    vectorizer = Vectorizer()
    scaler = Scaler()
    clusterizer = KMeansClustering()

    dataset = dataset_loader.load_dataset()
    vectorized_dataset = vectorizer.vectorize(dataset)
    scaled_dataset = scaler.scale(vectorized_dataset)

    scores = clusterizer.clusterize(scaled_dataset)
    plot_silhouette_scores(scores, clusterizer.k_search_range)

    spark.stop()


if __name__ == '__main__':
    main()
