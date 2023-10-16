# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC #### RELEVANCY MODELLING CHALLENGE
# MAGIC Black Swan Data creates dashboards for clients which enable them to make more informed brand and product decisions. These dashboards provide clients such as PepsiCo, Coty and P&G with a landscape view of market social trends. A crucial part of our success as a business rests on our ability to correctly identify when a user is talking about a particular topic so that clients can have faith in our assesments of social trends. Relevancy models provide one way of assessing whether a topic mentioned within a social media post is relevant or not. Your goal is to build one of these relevancy models. 
# MAGIC
# MAGIC #####Context classification challenge 
# MAGIC When analyzing social media text content, the first step is to gather the relevant raw text data needed for the purposes of the analysis. This can be accomplished by querying the overall social media database with a set of keywords in order to create a smaller dataset relevant to each market category. At Black Swan Data, we work with thousands of query words to achieve this. The result is millions of tweets, blogs, and other social media documents grouped into each thematic database each month. However, as countless words have multiple meanings, these databases are inevitably noisy. 
# MAGIC
# MAGIC Take the word `chips` for example. If we search for this on social media, we get all kinds of hits, e.g. 
# MAGIC >  *<font size="1">I ate chips with majo. </font>* 
# MAGIC >
# MAGIC >  *<font size="1">Intel chips are better than AMD ones. </font>* 
# MAGIC >
# MAGIC >  *<font size="1">Cheap professional poker chips (11.5g) with case!  </font>*  
# MAGIC
# MAGIC Depending on which category we are analyzing, only one of the three examples will be relevant, and other two will be noise! If the category is 'food & snacking' then only the first message will be relevant. If, however, the category is 'technology' then the second message will be relevant and so on.
# MAGIC
# MAGIC Your challenge is to use the learning database linked below to develop a machine learning system which can predict whether a query word hit is relevant to a particular market category or not given the context provided by the social media document.
# MAGIC
# MAGIC **Hints:**
# MAGIC * We understand you are doing this in your personal time and will not be able to do everything you would like. Your aim should therefore be to demonstrate you understand concepts, technologies and issues related to the problem. 
# MAGIC * Be prepared to explain your methodology and to talk us through your code during the interview.
# MAGIC * Do not worry if you find incorrect annotations.
# MAGIC * You have permissions to connect to and manage a cluster called `Interview Cluster - <name>` which can be selected from the drop down menu in the top right corner.
# MAGIC * Cluster libraries can be modified by opening the cluster and navigating to the `Libraries` tab. For a list of pre-installed libraries, please refer to the following page: https://docs.databricks.com/release-notes/runtime/13.2ml.html
# MAGIC * You will also have access to the mounted S3 bucket at `/dbfs/mnt/bs-databricks-workspaces-interview/data-science-test/` which you may use to store any downloaded models or intermediary files which you choose to use/create. This can be browsed by clicking `Data` on the left hand panel and then selecting `Browse DBFS`.
# MAGIC * Get in touch if you have problems using the cluster, or accessing the data as this is not meant to be part of the test.
# MAGIC
# MAGIC #####Data 
# MAGIC
# MAGIC There are three data resources available: 
# MAGIC
# MAGIC - Training database JSON: `/dbfs/mnt/bs-databricks-workspaces-interview/data-science-test/us_snacking_relevance_competition_train`
# MAGIC - Testing database JSON: `/dbfs/mnt/bs-databricks-workspaces-interview/data-science-test/us_snacking_relevance_competition_test`
# MAGIC - Scalable evaluation database: `/dbfs/mnt/bs-databricks-workspaces-interview/data-science-test/us_snacking_june21_live_rnd_30m`
# MAGIC
# MAGIC The data content of the files is the same, but while the training and testing database is tagged (is_relevant field), this information is missing from the scalable evaluation database. Use this dataset to evaluate how scalable your model(s) are on large datasets (30M documents).
# MAGIC
# MAGIC The files are in UTF-8 encoded .json format and contain the following fields: 
# MAGIC * *_unit_id*: unique identifier of documents
# MAGIC * *category*: the market (Beverages or Snacking)
# MAGIC * *message*: the text document itself
# MAGIC * *topic*: the name of the ingredient in question
# MAGIC * *is_relevant*: the target variable / class label (relevant or irrelevant)
# MAGIC * *inclusion*: how the topic was mentioned within the text
# MAGIC * *date*: the date the document was created / downloaded
# MAGIC * *taxonomies_lenses*: a python-formatted list containing the categories of the mentions found in the message
# MAGIC * *taxonomies*: an array of structs listing all the topic mentions which exist within the message and the index locations of each mention
# MAGIC
# MAGIC Naturally, it is not mandatory to use all fields, only those that effectively help with model building. 
# MAGIC

# COMMAND ----------

# Data
df_train = spark.read.json('/dbfs/mnt/bs-databricks-workspaces-interview/data-science-test/us_snacking_relevance_competition_train')

df_test = spark.read.json('/dbfs/mnt/bs-databricks-workspaces-interview/data-science-test/us_snacking_relevance_competition_test')

df_scale = spark.read.parquet('/dbfs/mnt/bs-databricks-workspaces-interview/data-science-test/us_snacking_june21_live_rnd_30m')


# COMMAND ----------

display(df_train.select(["_unit_id", "message", "is_relevant"]).show(5))

# COMMAND ----------

display(df_test.select(["_unit_id", "message", "is_relevant"]).show(5))

# COMMAND ----------

display(df_scale.select(["source_id", "message"]).show(5))

# COMMAND ----------

df_scale = df_scale.withColumnRenamed("source_id", "_unit_id")

# COMMAND ----------

display(df_scale.select(["_unit_id", "message"]).show(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Here is some assumtion that I will work with:
# MAGIC * I am going to assume that `message` is the most informative column, other columns can be revised later.
# MAGIC * I am going to replace the emojis replaced by their names. 
# MAGIC * `message` need to be cleaned up and formated.
# MAGIC * I'm not going to investigate class imbalance.
# MAGIC * Im not going to use advance model.
# MAGIC * Im not going to investigate model performance.
# MAGIC * I will use pretrained models to extract some embedding. (Many methods and many models can be used instead) 
# MAGIC
# MAGIC

# COMMAND ----------

!pip install text-unidecode
!pip installbertopic
!pip install emoji
!pip install umap-learn==0.5.3
!pip install hdbscan 
!pip install sentence_transformers


# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC
# MAGIC import logging
# MAGIC
# MAGIC logger = logging.getLogger(__name__)
# MAGIC previous_level = logger.getEffectiveLevel()
# MAGIC
# MAGIC # Set logging level to CRITICAL to only display error messages
# MAGIC logger.setLevel(logging.CRITICAL)

# COMMAND ----------

from sentence_transformers import SentenceTransformer
import umap
import hdbscan

from preprocessing import ColumnSelectTransformer, SparkFramedPipeline, TextNormalizer

# COMMAND ----------

feature_col = "message"
feature_norm_col = f"{feature_col}_norm"
SENT_MODEL = "facebook/bart-large-mnli" # 'all-MiniLM-L6-v2'
N_COM = 5
N_NEIG = 10
MIN_DIST = 0.1
MIN_SMP = 5
MIN_CLUS = 5
TOPIC ="snacking"

# COMMAND ----------

normalizer = SparkFramedPipeline(
  [
    ColumnSelectTransformer(column_selector=["_unit_id", "is_relevant", feature_col]),
    TextNormalizer(inputCol=feature_col, outputCol=feature_norm_col)
    ]
  )

sentence_transformer = SentenceTransformer(SENT_MODEL)

# UMAP for dimensionality reduction
umap_model = umap.UMAP(n_neighbors=N_NEIG, min_dist=MIN_DIST, n_components=N_COM, metric='cosine')

# HDBSCAN for clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUS, min_samples=MIN_SMP, metric='euclidean', cluster_selection_method='eom', prediction_data=True)



# COMMAND ----------

processed_df = normalizer.transform(df_train)
texts = [row[feature_norm_col] for row in processed_df.collect()]
embeddings = sentence_transformer.encode(texts, convert_to_tensor=True)
umap_embeddings = umap_model.fit_transform(embeddings)
clusterer.fit(umap_embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check the normalization in here /preprocessing/_text.py

# COMMAND ----------

display(processed_df.select(["_unit_id", "message", "message_norm", "is_relevant"]).limit(10))

# COMMAND ----------

from pyspark.sql import functions as F
count_df = processed_df.groupBy("is_relevant").agg(F.count("is_relevant").alias("count"))
count_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Transformers Approach

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check the SentenceTransformer /features/_embedding.py
# MAGIC ### Check the zero-shot-classification /features/_mnli_proba.py

# COMMAND ----------

from features import UmapEmbeddingExtractor, MNLI_Probability, LabelEncoder

feature_extractor = SparkFramedPipeline(
  [
    normalizer,
    UmapEmbeddingExtractor(
      inputCol=feature_norm_col,
      umap_model=umap_model,
      clusterer=clusterer,
      n_components=N_COM,
      ),
    MNLI_Probability(
      inputCol=feature_norm_col,
      outputCol="mnli_proba",
      topic=TOPIC,
    ),
    ]
  )


# COMMAND ----------

model_cols = []
for i in range(N_COM):
  model_cols += [f'umap{i + 1:02}',]
model_cols +=  ["mnli_proba", "cluster_label",]

label_col = "is_relevant_label"
pred_col = "is_relevant_pred"

# COMMAND ----------

training_elt = SparkFramedPipeline(
  [
    feature_extractor,
    LabelEncoder(cat_dict={"irrelevant": 0, "relevant": 1}, inputCol="is_relevant", outputCol=label_col),
    ColumnSelectTransformer(column_selector= model_cols + [label_col]),
  ]
)


# COMMAND ----------

train_data, test_data = training_elt.transform(df_train), training_elt.transform(df_test)

# COMMAND ----------

display(train_data.select(["umap01", "umap02", "umap03", "cluster_label", "mnli_proba", "is_relevant_label"]).show(25))

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier


# COMMAND ----------


# Preprocess data and assembling features
assembler = VectorAssembler(inputCols=model_cols, outputCol="features")

# Create GBTClassifier
gbt_classifier = GBTClassifier(
  featuresCol="features",
  labelCol=label_col,
  predictionCol=pred_col,
  maxIter=100,
  subsamplingRate=0.75,
  )

# Create pipeline with GBTClassifier
training_pipeline = Pipeline(stages=[assembler, gbt_classifier])

# Setup the parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(gbt_classifier.maxDepth, [6, 10]) \
    .build()

evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=pred_col, metricName="areaUnderROC")

# # Create a cross-validator
# crossval = CrossValidator(estimator=training_pipeline,
#                           estimatorParamMaps=paramGrid,
#                           evaluator=evaluator,
#                           numFolds=3)



# COMMAND ----------

# # Fit the model
# cvModel = crossval.fit(train_data)

# # Retrieve best Model
# best_model = cvModel.bestModel

# COMMAND ----------

# Fit the model
gbt_pipeline = training_pipeline.fit(train_data)

# COMMAND ----------


# Save the best model to a specified path
path = "/Users/kkahloots@gmail.com/models/"
gbt_pipeline.write().overwrite().save(path)

# COMMAND ----------

# Evaluate the model on the test data
predictions = gbt_pipeline.transform(test_data)
auc = evaluator.evaluate(predictions)

print("Training Dataset Area Under ROC: ", auc)

# COMMAND ----------

# Evaluate the model on the test data
predictions = gbt_pipeline.transform(test_data)
auc = evaluator.evaluate(predictions)

print("Testing Dataset Area Under ROC: ", auc)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Production Pipleline

# COMMAND ----------

inference_elt = SparkFramedPipeline(
  [
    ColumnSelectTransformer(column_selector=["_unit_id", feature_col]),
    TextNormalizer(inputCol=feature_col, outputCol=feature_norm_col),
    UmapEmbeddingExtractor(
      inputCol=feature_norm_col,
      umap_model=umap_model,
      clusterer=clusterer,
      n_components=N_COM,
      ),
    MNLI_Probability(
      inputCol=feature_norm_col,
      outputCol="mnli_proba",
      topic=TOPIC,
    ),
    gbt_pipeline,
  ]
)


# COMMAND ----------

display(inference_elt.transform(df_scale.limit(10)))

# COMMAND ----------

# MAGIC %md
# MAGIC # SpaCy Approach

# COMMAND ----------

!pip install spacy
#!python -m spacy download en_core_web_lg
!python -m spacy download en_core_web_sm

# COMMAND ----------

from features import SpacySvdEmbeddingExtractor

feature_extractor = SparkFramedPipeline(
  [
    normalizer,
    SpacySvdEmbeddingExtractor(
      inputCol=feature_norm_col,
      n_components=N_COM,
      model_name="en_core_web_sm",
      ),

    ]
  )



# COMMAND ----------

model_cols = []
for i in range(N_COM):
  model_cols += [f'svd{i + 1:02}',]

label_col = "is_relevant_label"
pred_col = "is_relevant_pred"

# COMMAND ----------

training_elt = SparkFramedPipeline(
  [
    feature_extractor,
    LabelEncoder(cat_dict={"irrelevant": 0, "relevant": 1}, inputCol="is_relevant", outputCol=label_col),
    ColumnSelectTransformer(column_selector= model_cols + [label_col]),
  ]
)


# COMMAND ----------

train_data, test_data = training_elt.transform(df_train), training_elt.transform(df_test)

# COMMAND ----------

display(train_data.select(["svd01", "svd02", "svd03", "is_relevant_label"]).show(25))

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

# COMMAND ----------


# Preprocess data and assembling features
assembler = VectorAssembler(inputCols=model_cols, outputCol="features")

rf_classifier = RandomForestClassifier(         
                            featuresCol="features",
                            labelCol=label_col,
                            predictionCol=pred_col,
                            numTrees=100,
                            subsamplingRate=0.75,
                            )

# Create pipeline with GBTClassifier
training_pipeline = Pipeline(stages=[assembler, rf_classifier])

evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=pred_col, metricName="areaUnderROC")




# COMMAND ----------

# Fit the model
rf_pipeline = training_pipeline.fit(train_data)

# COMMAND ----------

# Evaluate the model on the test data
predictions = rf_pipeline.transform(test_data)
auc = evaluator.evaluate(predictions)

print("Training Dataset Area Under ROC: ", auc)

# COMMAND ----------

# Evaluate the model on the test data
predictions = rf_pipeline.transform(test_data)
auc = evaluator.evaluate(predictions)

print("Testing Dataset Area Under ROC: ", auc)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Production Pipleline
# MAGIC

# COMMAND ----------

inference_elt = SparkFramedPipeline(
  [
    ColumnSelectTransformer(column_selector=["_unit_id", feature_col]),
    TextNormalizer(inputCol=feature_col, outputCol=feature_norm_col),
    SpacySvdEmbeddingExtractor(
      inputCol=feature_norm_col,
      n_components=N_COM,
      model_name="en_core_web_sm",
      ),
    rf_pipeline,
  ]
)


# COMMAND ----------

display(inference_elt.transform(df_scale.limit(10)))

# COMMAND ----------



# COMMAND ----------

