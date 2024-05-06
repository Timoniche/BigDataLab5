import pathlib
import matplotlib.pyplot as plt

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql import SparkSession

# TODO:
#  1) добавить параметры (спарк by examples)
#  2) Вынести все параметры в отдельный конфиг
#  3) ООП

spark = SparkSession.builder \
    .master("local") \
    .appName("KMean") \
    .getOrCreate()


def load_dataset(dataset_path):
    dataset = spark.read.csv(
        dataset_path,
        header=True,
        inferSchema=True,
        sep='\t',
    )
    dataset.fillna(value=0)

    output_col = 'features'
    vector_assembler = VectorAssembler(
        inputCols=[
            'product_quantity',
            'completeness',
            'energy-kcal_100g',
            'sugars_100g',
            'energy_100g',
            'fat_100g',
            'saturated-fat_100g',
            'carbohydrates_100g',
        ],
        outputCol=output_col,
        handleInvalid='skip',
    )

    assembled_data = vector_assembler.transform(dataset)

    print('Assembled data count: ', assembled_data.count())
    show_n = 5
    print(f'First {show_n} vectorized: ')
    assembled_data.select(output_col).show(show_n)

    return assembled_data


def scale_assembled_dataset(assembled_data):
    scaler = StandardScaler(
        inputCol='features',
        outputCol='scaled_features',
        withStd=True,
        withMean=False,
    )
    scaler_model = scaler.fit(assembled_data)
    scaled_data = scaler_model.transform(assembled_data)

    scaled_data.select('scaled_features').show(5)

    return scaled_data


k_range = range(2, 7)


def kmeans_clustering(scaled_data):
    evaluator = ClusteringEvaluator(
        predictionCol='prediction',
        featuresCol='scaled_features',
        metricName='silhouette',
        distanceMeasure='squaredEuclidean'
    )

    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(featuresCol='scaled_features', k=k)
        model = kmeans.fit(scaled_data)
        predictions = model.transform(scaled_data)
        score = evaluator.evaluate(predictions)

        silhouette_scores.append(score)
        print(f'Silhouette Score for k = {k} is {score}')

    return silhouette_scores


def plot_silhouette_scores(scores):
    plt.plot(k_range, scores)
    plt.xlabel('k')
    plt.ylabel('silhouette score')
    plt.title('Silhouette Score')
    plt.show()


def main():
    curdir = str(pathlib.Path(__file__).parent.resolve())
    dataset_filename = 'dataset.csv'
    dataset_path = curdir + '/' + dataset_filename

    assembled_data = load_dataset(dataset_path)
    scaled_data = scale_assembled_dataset(assembled_data)
    scores = kmeans_clustering(scaled_data)
    plot_silhouette_scores(scores)

    spark.stop()


if __name__ == '__main__':
    main()
