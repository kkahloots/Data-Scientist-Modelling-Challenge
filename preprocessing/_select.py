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
from pyspark.sql import DataFrame
from pyspark.sql.functions import col


class ColumnSelectTransformer(Transformer):
    """
    A custom transformer that selects specified columns from the input data.

    Parameters
    ----------
    column_selector : list
        List of column names to be selected.
    """

    def __init__(self, column_selector):
        super(ColumnSelectTransformer, self).__init__()
        self.column_selector = column_selector

    def _transform(self, df: DataFrame) -> DataFrame:
        """
        Transforms the input data by selecting the specified columns.

        Parameters
        ----------
        df : DataFrame
            The input data to be transformed.

        Returns
        -------
        DataFrame
            The transformed data with only the selected columns.
        """
        return df.select(self.column_selector)
