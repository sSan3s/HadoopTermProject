from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, col, when
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create SparkSession
spark = SparkSession.builder \
    .appName("Spam Classification") \
    .getOrCreate()

# Load the input CSV file from HDFS
original_data = spark.read.csv("hdfs://localhost:9000/spam_project/input/spam.csv", header=True, inferSchema=True)

### Data Preprocessing
# Combine "_c2", "_c3", and "_c4" into "v2" column and drop unnecessary columns
data = original_data.withColumn("v2", concat_ws("", col("v2"), col("_c2"), col("_c3"), col("_c4")))
data = data.drop("_c2", "_c3", "_c4")

# Select only relevant columns and rename them
data = data.select(col("v1").alias("label"), col("v2").alias("message"))

# Transform "label" column: Set "spam" to 1 and others (e.g., "ham") to 0
data = data.withColumn("label", when(col("label") == "spam", 1).otherwise(0))

### Feature Engineering
# Tokenize the "message" column into individual words
tokenizer = Tokenizer(inputCol="message", outputCol="words")
data = tokenizer.transform(data)

# Remove stopwords (e.g., specific characters like commas) from tokenized words
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words", stopWords=[","])
data = remover.transform(data)

# Convert "filtered_words" into term frequency (TF) features
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=1000)
data = hashingTF.transform(data)

# Compute Inverse Document Frequency (IDF) from term frequency for better feature scaling
idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(data)
data = idf_model.transform(data)

# Show the processed data with extracted features
data.select("features", "label").show(5)

### Data Splitting
# Split the dataset into training (80%) and test (20%) sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

print(f"Training Data Count: {train_data.count()}")
print(f"Test Data Count: {test_data.count()}")

### Model Training
# Create a Naive Bayes model with features and label columns
nb = NaiveBayes(featuresCol="features", labelCol="label")
model = nb.fit(train_data)

### Model Prediction
# Predict labels on the test data
predictions = model.transform(test_data)

# Show the prediction results (actual vs predicted labels)
predictions.select("label", "prediction").show(5)

### Model Evaluation
# Evaluate the model's accuracy using a multiclass evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy:.2f}")

### Save Results
# Save the prediction results (message, actual label, predicted label) to HDFS
predictions.select("message", "label", "prediction") \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv("hdfs://localhost:9000/spam_project/output/spam_classification_results.csv", header=True)
