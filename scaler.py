import configparser

from pyspark.ml.feature import StandardScaler
from pyspark.sql import DataFrame


class Scaler:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        input_col = self.config['vectorizer']['vectorizedColumnName']
        self.output_col = self.config['scaler']['scaledColumnName']

        self.scaler = StandardScaler(
            inputCol=input_col,
            outputCol=self.output_col,
            withStd=True,
            withMean=False,
        )

    def scale(
            self,
            dataset: DataFrame,
    ):
        scaler_model = self.scaler.fit(dataset)
        scaled_data = scaler_model.transform(dataset)

        scaled_data.select(self.output_col).show(5)

        return scaled_data
