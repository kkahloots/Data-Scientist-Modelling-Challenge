# Copyright (c) 2023 Kal Kahloot
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


from pyspark.ml import Transformer
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import DataFrame

class SparkFramedPipeline(Pipeline):
    """
    Custom Pipeline that extends the PySpark Pipeline to store transformed column names.
    """

    def __init__(self, stages=None):
        """
        Initialize the SparkFramedPipeline with the given stages.

        :param stages: List of stages that make up the pipeline.
        """
        super(SparkFramedPipeline, self).__init__()
        self.setStages(stages or [])
        self.transformed_columns = None

    def fit(self, df: DataFrame):
        model = super(SparkFramedPipeline, self).fit(df)
        return model

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Apply transforms to the data, and store the transformed column names.

        :param df: Input data to be transformed.
        :return: Transformed data.
        """
        transformed_df = df
        for stage in self.getStages():
            transformed_df = stage.transform(transformed_df)

        self.transformed_columns = transformed_df.columns
        return transformed_df

    def get_feature_names_out(self):
        """
        Get the feature names of the transformed output.

        :return: Transformed column names.
        """
        return self.transformed_columns


class SparkFramer(Transformer):
    """
    Custom transformer that wraps another transformer, applies it to the input data,
    and returns a DataFrame with the transformed data and updated column names.
    """

    def __init__(self, transformer, passthrough=False):
        """
        Initialize the SparkFramer with the given transformer and passthrough option.

        :param transformer: Transformer instance to be wrapped.
        :param passthrough: Boolean, if True, original columns will be included in the output DataFrame.
        """
        super(SparkFramer, self).__init__()
        self.transformer = transformer
        self.passthrough = passthrough

    def _transform(self, df: DataFrame) -> DataFrame:
        """
        Transform the input data using the wrapped transformer and return a DataFrame
        with the transformed data and updated column names.

        :param df: Input data to be transformed.
        :return: Transformed data as a DataFrame.
        """
        transformed_df = self.transformer.transform(df)

        # Drop original columns if passthrough is set to False
        if not self.passthrough:
            for column in df.columns:
                transformed_df = transformed_df.drop(column)

        return transformed_df
