import configparser

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame


class Vectorizer:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        self.output_col = self.config['vectorizer']['vectorizedColumnName']
        col_names = self.config['dataset']['featureColumns']
        self.input_cols = col_names.split(',')

        print('input columns for vectorization: ', self.input_cols)

        self.vector_assembler = VectorAssembler(
            inputCols=self.input_cols,
            outputCol=self.output_col,
            handleInvalid='skip',
        )

    def vectorize(
            self,
            dataset: DataFrame,
    ):
        assembled_data = self.vector_assembler.transform(dataset)

        print('Assembled data count: ', assembled_data.count())
        show_n = 5
        print(f'First {show_n} vectorized: ')
        assembled_data.select(self.output_col).show(show_n)

        return assembled_data
