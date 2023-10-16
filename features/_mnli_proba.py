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

from transformers import pipeline
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

class MNLI_Probability(Transformer, HasInputCol, HasOutputCol):
    """
    Custom transformer to calculate probability based on zero-shot-classification using BERT.

    It can be used in a PySpark ML Pipeline.
    """

    def __init__(
      self, 
      inputCol,
      outputCol,      
      topic,
      model_name="facebook/bart-large-mnli",
      **kwargs
      ):
        """
        Initialize the MNLI_Probability with the given topic.

        :param topic: Topic used for classification.
        :param inputCol: The name of the input column.
        :param outputCol: The name of the output column.
        """
        super().__init__(**kwargs)
        self._set(inputCol=inputCol, outputCol=outputCol)
        self.topic = topic

        # Create the classifier
        self.classifier = pipeline("zero-shot-classification", model=model_name)

    @staticmethod
    def calculate_probability(text: str, classifier, topic) -> float:
        """
        Method to calculate probability using BERT zero-shot-classification.

        :param text: Input text to be processed.
        :param classifier: Classifier used for zero-shot-classification.
        :param topic: Topic for classification.
        :return: Probability.
        """
        candidate_labels = [topic]
        result = classifier(text, candidate_labels)
        probability = result['scores'][0]
        return probability
   
    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset by applying the calculate_probability function.

        :param dataset: Input DataFrame.
        :return: Transformed DataFrame.
        """

        # Create a user-defined function (UDF) to apply the text probability calculation.
        calculate_probability_udf = udf(lambda text: MNLI_Probability.calculate_probability(text, self.classifier, self.topic), DoubleType())

        # Apply the UDF to the specified input column and create a new DataFrame with the output column.
        return dataset.withColumn(self.getOutputCol(), calculate_probability_udf(dataset[self.getInputCol()]))

      