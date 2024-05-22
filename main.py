import os
import sys

import matplotlib.pyplot as plt

from logger import Logger

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from dataset_loader import DatasetLoader  # noqa: E402
from kmeans_clustering import KMeansClustering  # noqa: E402
from scaler import Scaler  # noqa: E402
from spark_session_provider import SparkSessionProvider  # noqa: E402
from vectorizer import Vectorizer  # noqa: E402


def plot_silhouette_scores(scores, k_search_range):
    plt.plot(k_search_range, scores)
    plt.xlabel('k')
    plt.ylabel('silhouette score')
    plt.title('Silhouette Score')
    plt.show()


def main():
    logger = Logger(show=True)
    log = logger.get_logger(__name__)

    log.info('Spark setup...')
    spark = SparkSessionProvider().provide_session()
    dataset_loader = DatasetLoader(spark_session=spark)
    vectorizer = Vectorizer()
    scaler = Scaler()
    clusterizer = KMeansClustering()

    log.info('Loading dataset...')
    dataset = dataset_loader.load_dataset()
    log.info('Vectorizing dataset...')
    vectorized_dataset = vectorizer.vectorize(dataset)
    log.info('Scaling dataset...')
    scaled_dataset = scaler.scale(vectorized_dataset)
    scaled_dataset.collect()

    _ = clusterizer.clusterize(scaled_dataset)
    # plot_silhouette_scores(scores, clusterizer.k_search_range)

    spark.stop()


if __name__ == '__main__':
    main()
