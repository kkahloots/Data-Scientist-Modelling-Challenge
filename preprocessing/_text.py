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

from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from text_unidecode import unidecode
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.sql.types import StringType

import re
import codecs
import emoji


def replace_encoding_with_utf8(error: UnicodeError):
    # replace unencodable characters with '?'
    return ('?' * len(error.object[error.start : error.end]), error.end)

def replace_decoding_with_cp1252(error : UnicodeError):
    # attempt to decode bytes using cp1252
    return (error.object[error.start : error.end].decode('cp1252', 'replace'), error.end)

class TextNormalizer(Transformer, HasInputCol, HasOutputCol):
    """
    Custom transformer that resolves encoding issues and normalizes text.

    It can be used in a PySpark ML Pipeline.
    """

    def __init__(self, inputCol, outputCol, **kwargs):
        """
        Initialize the TextNormalizer with given passthrough columns and transformation column.

        - inputCol: The name of the input column containing the text.
        - outputCol: The name of the output column containing the transformed data.
        """
        super().__init__(**kwargs)
        self._set(inputCol=inputCol, outputCol=outputCol)
        
    @staticmethod
    def resolve_normalize(text: str) -> str:
        """
        Method to resolve different encodings and normalize the text.
        
        :param text: Input text to be processed.
        :return: Normalized text.
        """
        # Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
        codecs.register_error('replace_encoding_with_utf8', replace_encoding_with_utf8)
        codecs.register_error('replace_decoding_with_cp1252', replace_decoding_with_cp1252)

        text = (
            text.encode('raw_unicode_escape').
            decode('utf-8', errors='replace_decoding_with_cp1252').
            encode().
            decode('cp1252', errors='replace_encoding_with_utf8').
            encode().
            decode('utf-8', errors='replace_decoding_with_cp1252')
        )
        text = unidecode(text)

        # Remove "RT" indicator
        text = re.sub(r'\bRT\b', '', text)
        
        # Remove twitter handles (@user)
        text = re.sub(r'@\w+:', '', text)  # updated to remove ':'
        
        # Remove urls
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove '[URL]'
        text = re.sub(r'\[URL\]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Replace handles with a generic token
        text = re.sub(r'@\w+', '@USER', text)

        # Transform text with emojis replaced by their names.
        text = emoji.demojize(text, delimiters=("", ""))

        # Remove the '#' symbol from hashtags but keep the text
        text = text.replace('#', '')  

        return text
   
    def _transform(self, dataset: DataFrame) -> DataFrame:
        """
        Transforms the input dataset by applying the resolve_normalize function.

        :param dataset: Input DataFrame.
        :return: Transformed DataFrame.
        """

        # Create a user-defined function (UDF) to apply the text normalization.
        resolve_udf = udf(TextNormalizer.resolve_normalize, StringType())

        # Apply the UDF to the specified input column and create a new DataFrame with the output column.
        return dataset.withColumn(self.getOutputCol(), resolve_udf(dataset[self.getInputCol()]))
    