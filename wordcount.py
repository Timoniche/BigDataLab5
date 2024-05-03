import pathlib

import findspark

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local") \
    .appName("WordCount") \
    .getOrCreate()
sc = spark.sparkContext


def word_count(filepath):
    text_file = sc.textFile(filepath)
    counts = text_file.flatMap(lambda line: line.split(" ")) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda x, y: x + y) \
        .collect()

    return counts


def main():
    findspark.init()

    curdir = str(pathlib.Path(__file__).parent.resolve())
    filename = 'input.txt'
    filepath = curdir + '/' + filename

    word_counts = word_count(filepath)
    for word, count in word_counts:
        print(word, count, sep=": ")

    sc.stop()
    spark.stop()


if __name__ == '__main__':
    main()
