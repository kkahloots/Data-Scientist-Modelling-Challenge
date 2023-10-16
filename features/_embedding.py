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

from sentence_transformers import SentenceTransformer
import hdbscan
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import lit

import spacy
from sklearn.decomposition import TruncatedSVD

class UmapEmbeddingExtractor(Transformer, HasInputCol, HasOutputCol):
    """
    Transformer to extract embeddings using SentenceTransformer and reduce dimensionality using UMAP.

    Parameters:
    - inputCol: The name of the input column containing the text.
    - outputCol: The name of the output column containing the transformed data.
    - model_name: The name of the SentenceTransformer model (default 'all-MiniLM-L6-v2').
    - n_components: Number of components for UMAP reduction (default 5).
    """
  
    def __init__(
      self,
      inputCol,
      umap_model,
      clusterer,
      model_name='all-MiniLM-L6-v2',
      n_components=5,
      **kwargs
      ):
        super().__init__(**kwargs)
        self._set(inputCol=inputCol)

        self.model_name = model_name
        self.n_components = n_components
        self.sentence_transformer = SentenceTransformer(self.model_name)
        self.umap_model = umap_model
        self.clusterer = clusterer

    def _transform(self, dataset):
      # Apply UMAP transform
      texts = [row[self.getInputCol()] for row in dataset.collect()]
      embeddings = self.sentence_transformer.encode(texts, convert_to_tensor=True)
      umap_embeddings = self.umap_model.transform(embeddings)
      clusters, _ = hdbscan.approximate_predict(self.clusterer, umap_embeddings)

      # Create a DataFrame with UMAP-transformed embeddings
      umap_df = dataset.sparkSession.createDataFrame(
          [(Vectors.dense(row),) for row in umap_embeddings],
          ["umap_features"]
      )

      # Define a UDF to extract a specific element from a dense vector
      def extract_vector_element(vector, index):
          return float(vector[index])

      extract_vector_element_udf = udf(extract_vector_element, DoubleType())

      # Separate the UMAP features into individual columns
      for i in range(self.n_components):
          umap_df = umap_df.withColumn(
              f'umap{i + 1:02}',
              extract_vector_element_udf("umap_features", lit(i))
          )

      # Drop the "umap_features" column
      umap_df = umap_df.drop("umap_features")

      # Add the cluster labels
      cluster_df = dataset.sparkSession.createDataFrame(
          [(int(label),) for label in clusters],
          ["cluster_label"]
      )

      # Create a DataFrame with unique IDs to facilitate joining
      original_df_with_id = dataset.withColumn("id", monotonically_increasing_id())
      umap_df_with_id = umap_df.withColumn("id", monotonically_increasing_id())
      cluster_df_with_id = cluster_df.withColumn("id", monotonically_increasing_id())

      # Concatenate the UMAP features and cluster labels with the original dataset
      return (
        original_df_with_id
        .join(umap_df_with_id, "id", "inner")
        .join(cluster_df_with_id, "id", "inner")
        .drop("id")
        )


class SpacySvdEmbeddingExtractor(Transformer, HasInputCol, HasOutputCol):
    """
    Transformer to extract embeddings using SpaCy and reduce dimensionality using SVD.

    Parameters:
    - inputCol: The name of the input column containing the text.
    - outputCol: The name of the output column containing the transformed data.
    - model_name: The name of the SpaCy model (default 'en_core_web_lg').
    - n_components: Number of components for SVD reduction (default 5).
    """

    def __init__(
        self,
        inputCol,
        n_components=5,
        model_name='en_core_web_lg',
        **kwargs
    ):
        super().__init__(**kwargs)
        self._set(inputCol=inputCol)

        self.model_name = model_name
        self.n_components = n_components
        self.spacy_model = spacy.load(self.model_name)
        self.svd_model = TruncatedSVD(n_components=self.n_components)

    def _transform(self, dataset):
        # Extract embeddings using SpaCy
        texts = [row[self.getInputCol()] for row in dataset.collect()]
        embeddings = [self.spacy_model(text).vector for text in texts]

        # Apply SVD transform
        svd_embeddings = self.svd_model.fit_transform(embeddings)

        # Create a DataFrame with SVD-transformed embeddings
        svd_df = dataset.sparkSession.createDataFrame(
            [(Vectors.dense(row),) for row in svd_embeddings],
            ["svd_features"]
        )

        # Define a UDF to extract a specific element from a dense vector
        def extract_vector_element(vector, index):
            return float(vector[index])

        extract_vector_element_udf = udf(extract_vector_element, DoubleType())

        # Separate the SVD features into individual columns
        for i in range(self.n_components):
            svd_df = svd_df.withColumn(
                f'svd{i + 1:02}',
                extract_vector_element_udf("svd_features", lit(i))
            )

        # Drop the "svd_features" column
        svd_df = svd_df.drop("svd_features")

        # Create a DataFrame with unique IDs to facilitate joining
        original_df_with_id = dataset.withColumn("id", monotonically_increasing_id())
        svd_df_with_id = svd_df.withColumn("id", monotonically_increasing_id())

        # Concatenate the SVD features with the original dataset
        return (
            original_df_with_id
            .join(svd_df_with_id, "id", "inner")
            .drop("id")
        )
