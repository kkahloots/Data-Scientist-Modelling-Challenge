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
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

class LabelEncoder(Transformer, HasInputCol, HasOutputCol):
    """
    Transformer to encode values in the specified input column using a provided dictionary.

    Parameters:
    - inputCol: The name of the input column containing the values to be encoded.
    - outputCol: The name of the output column containing the encoded values.
    - cat_dict: Dictionary of keys and values used for encoding.
    """

    def __init__(self, cat_dict, inputCol, outputCol, **kwargs):
        super().__init__(**kwargs)
        self._set(inputCol=inputCol, outputCol=outputCol)
        self.cat_dict = cat_dict

    def _transform(self, dataset):
        """
        Transforms the input dataset by applying the value encoding.

        :param dataset: Input DataFrame.
        :return: Transformed DataFrame.
        """

        # Create a user-defined function (UDF) to apply the value encoding.
        def encode_value(value):
            return self.cat_dict.get(value, value)  # Return original value if key not found

        encode_udf = udf(encode_value, IntegerType())

        # Apply the UDF to the specified input column and create a new DataFrame with the output column.
        return dataset.withColumn(self.getOutputCol(), encode_udf(self.getInputCol()))
