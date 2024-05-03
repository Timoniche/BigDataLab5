import pathlib

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

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
            "product_quantity",
            "completeness",
            "energy-kcal_100g",
            "sugars_100g",
        ],
        outputCol=output_col,
        handleInvalid='skip',
    )

    assembled_data = vector_assembler.transform(dataset)

    print('Assembled data count: ', assembled_data.count())
    show_n = 5
    print(f'First {show_n} vectorized: ')
    assembled_data.select(output_col).show(show_n)


def main():
    curdir = str(pathlib.Path(__file__).parent.resolve())
    dataset_filename = 'small.csv'
    dataset_path = curdir + '/' + dataset_filename

    load_dataset(dataset_path)


if __name__ == '__main__':
    main()
