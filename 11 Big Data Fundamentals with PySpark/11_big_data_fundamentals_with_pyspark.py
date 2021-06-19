# Course 11
# Big Data Fundamentals with PySpark

#Chapter 1
#Introduction to Big Data analysis with Spark

#Understanding SparkContext
''''
Print the version of SparkContext in the PySpark shell.
Print the Python version of SparkContext in the PySpark shell.
What is the master of SparkContext in the PySpark shell?
'''
# Print the version of SparkContext
print("The version of Spark Context in the PySpark shell is", sc.version)

# Print the Python version of SparkContext
print("The Python version of Spark Context in the PySpark shell is", sc.pythonVer)

# Print the master of SparkContext
print("The master of Spark Context in the PySpark shell is", sc.master)

#Interactive Use of PySpark
''''
Create a python list named numb containing the numbers 1 to 100.
Load the list into Spark using Spark Context's parallelize method and assign it to a variable spark_data.
'''
# Create a python list of numbers from 1 to 100 
numb = range(1, 100)

# Load the list into PySpark  
spark_data = sc.parallelize(numb)

#Loading data in PySpark shell
#Load a local text file README.md in PySpark shell.

# Load a local file into PySpark shell
lines = sc.textFile(file_path)

#Use of lambda() with map()
''''
Print my_list which is available in your environment.
Square each item in my_list using map() and lambda().
Print the result of map function.
'''
# Print my_list in the console
print("Input list is", my_list)

# Square all numbers in my_list
squared_list_lambda = list(map(lambda x: x ** 2, my_list))

# Print the result of the map function
print("The squared numbers are", squared_list_lambda)

#Use of lambda() with filter()
''''
Print my_list2 which is available in your environment.
Filter the numbers divisible by 10 from my_list2 using filter() and lambda().
Print the numbers divisible by 10 from my_list2.
'''
# Print my_list2 in the console
print("Input list is:", my_list2)

# Filter numbers divisible by 10
filtered_list = list(filter(lambda x: (x%10 == 0), my_list2))

# Print the numbers divisible by 10
print("Numbers divisible by 10 are:", filtered_list)

#Chapter 2
#Programming in PySpark RDD’s

#RDDs from Parallelized collections
''''
Create an RDD named RDD from a list of words.
Confirm the object created is RDD.
'''
# Create an RDD from a list of words
RDD = sc.parallelize(["Spark", "is", "a", "framework", "for", "Big Data processing"])

# Print out the type of the created object
print("The type of RDD is", type(RDD))

#RDDs from External Datasets
# Print the file_path
print("The file_path is", file_path)

# Create a fileRDD from file_path
fileRDD = sc.textFile(file_path)

# Check the type of fileRDD
print("The file type of fileRDD is", type(fileRDD))

#Partitions in your data
''''
Find the number of partitions that support fileRDD RDD.
Create an RDD named fileRDD_part from the file path but create 5 partitions.
Confirm the number of partitions in the new fileRDD_part RDD.
'''
# Check the number of partitions in fileRDD
print("Number of partitions in fileRDD is", fileRDD.getNumPartitions())

# Create a fileRDD_part from file_path with 5 partitions
fileRDD_part = sc.textFile(file_path, minPartitions = 5)

# Check the number of partitions in fileRDD_part
print("Number of partitions in fileRDD_part is", fileRDD_part.getNumPartitions())

#Map and Collect
''''
Create map() transformation that cubes all of the numbers in numbRDD.
Collect the results in a numbers_all variable.
Print the output from numbers_all variable.
'''
# Create map() transformation to cube numbers
cubedRDD = numbRDD.map(lambda x: x ** 3)

# Collect the results
numbers_all = cubedRDD.collect()

# Print the numbers from numbers_all
for numb in numbers_all:
	print(numb)

#Filter and Count
''''
Create filter() transformation to select the lines containing the keyword Spark.
How many lines in fileRDD_filter contains the keyword Spark?
Print the first four lines of the resulting RDD.
'''
# Filter the fileRDD to select lines with Spark keyword
fileRDD_filter = fileRDD.filter(lambda line: 'Spark' in line)

# How many lines are there in fileRDD?
print("The total number of lines with the keyword Spark is", fileRDD_filter.count())

# Print the first four lines of fileRDD
for line in fileRDD_filter.take(4): 
	print(line)

#ReduceBykey and Collect
''''
Create a pair RDD named Rdd with tuples (1,2),(3,4),(3,6),(4,5).
Transform the Rdd with reduceByKey() into a pair RDD Rdd_Reduced by adding the values with the same key.
Collect the contents of pair RDD Rdd_Reduced and iterate to print the output.
'''
# Create PairRDD Rdd with key value pairs
Rdd = sc.parallelize([(1,2), (3,4), (3,6), (4,5)])

# Apply reduceByKey() operation on Rdd
Rdd_Reduced = Rdd.reduceByKey(lambda x, y: x + y)

# Iterate over the result and print the output
for num in Rdd_Reduced.collect(): 
  print("Key {} has {} Counts".format(num[0], num[1]))

#SortByKey and Collect
''''
Sort the Rdd_Reduced RDD using the key in descending order.
Collect the contents and iterate to print the output.
'''
# Sort the reduced RDD with the key by descending order
Rdd_Reduced_Sort = Rdd_Reduced.sortByKey(ascending=False)

# Iterate over the result and print the output
for num in Rdd_Reduced_Sort.collect():
  print("Key {} has {} Counts".format(num[0], num[1]))

#CountingBykeys
''''
Use the countByKey() action on the Rdd to count the unique keys and assign the result to a variable total.
What is the type of total?
Iterate over the total and print the keys and their counts.
'''
# Transform the rdd with countByKey()
total = Rdd.countByKey()

# What is the type of total?
print("The type of total is", type(total))

# Iterate over the total and print the output
for k, v in total.items(): 
  print("key", k, "has", v, "counts")

#Create a base RDD and transform it
# Create a baseRDD from the file path
baseRDD = sc.textFile(file_path)

# Split the lines of baseRDD into words
splitRDD = baseRDD.flatMap(lambda x: x.split())

# Count the total number of words
print("Total number of words in splitRDD:", splitRDD.count())

#Remove stop words and reduce the dataset
# Convert the words in lower case and remove stop words from stop_words
splitRDD_no_stop = splitRDD.filter(lambda x: x.lower() not in stop_words)

# Create a tuple of the word and 1 
splitRDD_no_stop_words = splitRDD_no_stop.map(lambda w: (w, 1))

# Count of the number of occurences of each word 
resultRDD = splitRDD_no_stop_words.reduceByKey(lambda x, y: x + y)

#Print word frequencies
''''
Print the first 10 words and their frequencies from the resultRDD.
Swap the keys and values in the resultRDD.
Sort the keys according to descending order.
Print the top 10 most frequent words and their frequencies.
'''
# Display the first 10 words and their frequencies
for word in resultRDD.take(10):
	print(word)

# Swap the keys and values
resultRDD_swap = resultRDD.map(lambda x: (x[1], x[0]))

# Sort the keys in descending order
resultRDD_swap_sort = resultRDD_swap.sortByKey(ascending=False)

# Show the top 10 most frequent words and their frequencies
for word in resultRDD_swap_sort.take(10):
	print("{} has {} counts". format(word[1], word[0]))

#Chapter 3
#PySpark SQL & DataFrames
# Create a list of tuples
sample_list = [('Mona',20), ('Jennifer',34), ('John',20), ('Jim',26)]

# Create a RDD from the list
rdd = sc.parallelize(sample_list)

# Create a PySpark DataFrame
names_df = spark.createDataFrame(rdd, schema=['Name', 'Age'])

# Check the type of names_df
print("The type of names_df is", type(names_df))

#Loading CSV into DataFrame
''''
Create a DataFrame from file_path variable which is the path to the people.csv file.
Confirm the output as PySpark DataFrame.
'''
# Create an DataFrame from file_path
people_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Check the type of people_df
print("The type of people_df is", type(people_df))

#Inspecting data in PySpark DataFrame
''''
Print the first 10 observations in the people_df DataFrame.
Count the number of rows in the people_df DataFrame.
How many columns does people_df DataFrame have and what are their names?
'''
# Print the first 10 observations 
people_df.show(10)

# Count the number of rows 
print("There are {} rows in the people_df DataFrame.".format(people_df.count()))

# Count the number of columns and their names
print("There are {} columns in the people_df DataFrame and their names are {}".format(len(people_df.columns), people_df.columns))

#PySpark DataFrame subsetting and cleaning
''''
Select 'name', 'sex' and 'date of birth' columns from people_df and create people_df_sub DataFrame.
Print the first 10 observations in the people_df DataFrame.
Remove duplicate entries from people_df_sub DataFrame and create people_df_sub_nodup DataFrame.
'''
# Select name, sex and date of birth columns
people_df_sub = people_df.select('name', 'sex', 'date of birth')

# Print the first 10 observations from people_df_sub
people_df_sub.show(10)

# Remove duplicate entries from people_df_sub
people_df_sub_nodup = people_df_sub.dropDuplicates()

# Count the number of rows
print("There were {} rows before removing duplicates, and {} rows after removing duplicates".format(people_df_sub.count(), people_df_sub_nodup.count()))

#Filtering your DataFrame
''''
Filter the people_df DataFrame to select all rows where sex is female into people_df_female DataFrame.
Filter the people_df DataFrame to select all rows where sex is male into people_df_male DataFrame.
Count the number of rows in people_df_female and people_df_male DataFrames.
'''
# Filter people_df to select females 
people_df_female = people_df.filter(people_df.sex == "female")

# Filter people_df to select males
people_df_male = people_df.filter(people_df.sex == "male")

# Count the number of rows 
print("There are {} rows in the people_df_female DataFrame and {} rows in the people_df_male DataFrame".format(people_df_female.count(), people_df_male.count()))

#Running SQL Queries Programmatically
''''
Create a temporary table people that's a pointer to the people_df DataFrame.
Construct a query to select the names of the people from the temporary table people.
Assign the result of Spark's query to a new DataFrame - people_df_names.
Print the top 10 names of the people from people_df_names DataFrame.
'''
# Create a temporary table "people"
people_df.createOrReplaceTempView("people")

# Construct a query to select the names of the people from the temporary table "people"
query = '''SELECT name FROM people'''

# Assign the result of Spark's query to people_df_names
people_df_names = spark.sql(query)

# Print the top 10 names of the people
people_df_names.show(10)

#SQL queries for filtering Table
''''
Filter the people table to select all rows where sex is female into people_female_df DataFrame.
Filter the people table to select all rows where sex is male into people_male_df DataFrame.
Count the number of rows in both people_female and people_male DataFrames.
'''
# Filter the people table to select female sex 
people_female_df = spark.sql('SELECT * FROM people WHERE sex=="female"')

# Filter the people table DataFrame to select male sex
people_male_df = spark.sql('SELECT * FROM people WHERE sex=="male"')

# Count the number of rows in both DataFrames
print("There are {} rows in the people_female_df and {} rows in the people_male_df DataFrames".format(people_female_df.count(), people_male_df.count()))

#PySpark DataFrame visualization
''''
Print the names of the columns in names_df DataFrame.
Convert names_df DataFrame to df_pandas Pandas DataFrame.
Use matplotlib's plot() method to create a horizontal bar plot with 'Name' on x-axis and 'Age' on y-axis.
'''
# Check the column names of names_df
print("The column names of names_df are", names_df.columns)

# Convert to Pandas DataFrame  
df_pandas = names_df.toPandas()

# Create a horizontal bar plot
df_pandas.plot(kind='barh', x='Name', y='Age', colormap='winter_r')
plt.show()

#Part 1: Create a DataFrame from CSV file
# Load the Dataframe
fifa_df = spark.read.csv('/usr/local/share/datasets/Fifa2018_dataset.csv', header=True, inferSchema=True)

# Check the schema of columns
fifa_df.printSchema()

# Show the first 10 observations
fifa_df.show(10)

# Print the total number of rows
print("There are {} rows in the fifa_df DataFrame".format(fifa_df.count()))

#Part 2: SQL Queries on DataFrame
''''
Create temporary table fifa_df from fifa_df_table DataFrame.
Construct a "query" to extract the "Age" column from Germany players.
Apply the SQL "query" to the temporary view table and create a new DataFrame.
Computes basic statistics of the created DataFrame.
'''
# Create a temporary view of fifa_df
fifa_df.createOrReplaceTempView('fifa_df_table')

# Construct the "query"
query = '''SELECT Age FROM fifa_df_table WHERE Nationality == "Germany"'''

# Apply the SQL "query"
fifa_df_germany_age = spark.sql(query)

# Generate basic statistics
fifa_df_germany_age.describe().show()

#Part 3: Data visualization
''''
Convert fifa_df_germany_age to fifa_df_germany_age_pandas Pandas DataFrame.
Generate a density plot of the 'Age' column from the fifa_df_germany_age_pandas Pandas DataFrame.
'''
# Convert fifa_df to fifa_df_germany_age_pandas DataFrame
fifa_df_germany_age_pandas = fifa_df_germany_age.toPandas()

# Plot the 'Age' density of Germany Players
fifa_df_germany_age_pandas.plot(kind='density')
plt.show()

#Chapter 4
#Machine Learning with PySpark MLlib
''''
Import pyspark.mllib recommendation submodule and Alternating Least Squares class.
Import pyspark.mllib classification submodule and Logistic Regression with LBFGS class.
Import pyspark.mllib clustering submodule and kmeans class.
'''
# Import the library for ALS
from pyspark.mllib.recommendation import ALS

# Import the library for Logistic Regression
from pyspark.mllib.classification import LogisticRegressionWithLBFGS

# Import the library for Kmeans
from pyspark.mllib.clustering import KMeans

#Loading Movie Lens dataset into RDDs
''''
Load the ratings.csv dataset into an RDD.
Split the RDD using , as a delimiter.
For each line of the RDD, using Rating() class create a tuple of userID, productID, rating.
Randomly split the data into training data and test data (0.8 and 0.2).
'''
# Load the data into RDD
data = sc.textFile(file_path)

# Split the RDD 
ratings = data.map(lambda l: l.split(','))

# Transform the ratings RDD 
ratings_final = ratings.map(lambda line: Rating(int(line[0]), int(line[1]), float(line[2])))

# Split the data into training and test
training_data, test_data = ratings_final.randomSplit([0.8, 0.2])

#Model training and predictions
''''
Train ALS algorithm with training data and configured parameters (rank = 10 and iterations = 10).
Drop the rating column in the test data.
Test the model by predicting the rating from the test data.
Print the first two rows of the predicted ratings.
'''
# Create the ALS model on the training data
model = ALS.train(training_data, rank=10, iterations=10)

# Drop the ratings column 
testdata_no_rating = test_data.map(lambda p: (p[0], p[1]))

# Predict the model  
predictions = model.predictAll(testdata_no_rating)

# Print the first rows of the RDD
predictions.take(2)

#Model evaluation using MSE
''''
Organize ratings RDD to make ((user, product), rating).
Organize predictions RDD to make ((user, product), rating).
Join the prediction RDD with the ratings RDD.
Evaluate the model using MSE between original rating and predicted rating and print it.
'''
# Prepare ratings data
rates = ratings_final.map(lambda r: ((r[0], r[1]), r[2]))

# Prepare predictions data
preds = predictions.map(lambda r: ((r[0], r[1]), r[2]))

# Join the ratings data with predictions data
rates_and_preds = rates.join(preds)

# Calculate and print MSE
MSE = rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error of the model for the test data = {:.2f}".format(MSE))

#Loading spam and non-spam data
''''
Create two RDDS, one for 'spam' and one for 'non-spam (ham)'.
Split each email in 'spam' and 'non-spam' RDDs into words.
Print the first element in the split RDD of both 'spam' and 'non-spam'.
'''
# Load the datasets into RDDs
spam_rdd = sc.textFile(file_path_spam)
non_spam_rdd = sc.textFile(file_path_non_spam)

# Split the email messages into words
spam_words = spam_rdd.map(lambda email: email.split(' '))
non_spam_words = non_spam_rdd.map(lambda email: email.split(' '))

# Print the first element in the split RDD
print("The first element in spam_words is", spam_words.first())
print("The first element in non_spam_words is", non_spam_words.first())

#Loading spam and non-spam data
''''
Create two RDDS, one for 'spam' and one for 'non-spam (ham)'.
Split each email in 'spam' and 'non-spam' RDDs into words.
Print the first element in the split RDD of both 'spam' and 'non-spam'.
'''
# Load the datasets into RDDs
spam_rdd = sc.textFile(file_path_spam)
non_spam_rdd = sc.textFile(file_path_non_spam)

# Split the email messages into words
spam_words = spam_rdd.flatMap(lambda email: email.split(' '))
non_spam_words = non_spam_rdd.flatMap(lambda email: email.split(' '))

# Print the first element in the split RDD
print("The first element in spam_words is", spam_words.first())
print("The first element in non_spam_words is", non_spam_words.first())

#Feature hashing and LabelPoint
''''
Create a HashingTF() instance to map email text to vectors of 200 features.
Each message in 'spam' and 'non-spam' datasets are split into words, and each word is mapped to one feature.
Label the features: 1 for spam, 0 for non-spam.
Combine both the spam and non-spam samples into a single dataset.
'''
# Create a HashingTf instance with 200 features
tf = HashingTF(numFeatures=200)

# Map each word to one feature
spam_features = tf.transform(spam_words)
non_spam_features = tf.transform(non_spam_words)

# Label the features: 1 for spam, 0 for non-spam
spam_samples = spam_features.map(lambda features:LabeledPoint(1, features))
non_spam_samples = non_spam_features.map(lambda features:LabeledPoint(0, features))

# Combine the two datasets
samples = spam_samples.join(non_spam_samples)

#Logistic Regression model training
''''
Split the combined data into training and test sets (80/20).
Train the Logistic Regression (LBFGS variant) model with the training dataset.
Create a prediction label from the trained model on the test dataset.
Combine the labels in the test dataset with the labels in the prediction dataset.
Calculate the accuracy of the trained model using original and predicted labels on the labels_and_preds.
'''
# Split the data into training and testing
train_samples, test_samples = samples.randomSplit([0.8, 0.2])

# Train the model
model = LogisticRegressionWithLBFGS.train(train_samples)

# Create a prediction label from the test data
predictions = model.predict(test_samples.map(lambda x: x.features))

# Combine original labels with the predicted labels
labels_and_preds = test_samples.map(lambda x: x.label).zip(predictions)

# Check the accuracy of the model on the test data
accuracy = labels_and_preds.filter(lambda x: x[0] == x[1]).count() / float(test_samples.count())
print("Model accuracy : {:.2f}".format(accuracy))

#Loading and parsing the 5000 points data
''''
Load the 5000_points dataset into a RDD named clusterRDD.
Transform the clusterRDD by splitting the lines based on the tab ("\t").
Transform the split RDD to create a list of integers for the two columns.
Confirm that there are 5000 rows in the dataset.
'''
# Load the dataset into a RDD
clusterRDD = sc.textFile(file_path)

# Split the RDD based on tab
rdd_split = clusterRDD.map(lambda x: x.split("\t"))

# Transform the split RDD by creating a list of integers
rdd_split_int = rdd_split.map(lambda x: [int(x[0]), int(x[1])])

# Count the number of rows in RDD 
print("There are {} rows in the rdd_split_int dataset".format(rdd_split_int.count()))

#K-means training
''''
Train the KMeans model with clusters from 13 to 16 and print the WSSSE for each cluster.
Train the KMeans model again with the best cluster (k=15).
Get the Cluster Centers (centroids) of KMeans model trained with k=15.
'''
# Train the model with clusters from 13 to 16 and compute WSSSE 
for clst in range(13, 17):
    model = KMeans.train(rdd_split_int, clst, seed=1)
    WSSSE = rdd_split_int.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("The cluster {} has Within Set Sum of Squared Error {}".format(clst, WSSSE))

# Train the model again with the best k 
model = KMeans.train(rdd_split_int, k=15, seed=1)

# Get cluster centers
cluster_centers = model.clusterCenters

#Visualizing clusters
''''
Convert rdd_split_int RDD into a Spark DataFrame.
Convert Spark DataFrame into a Pandas DataFrame.
Create a Pandas DataFrame from cluster_centers list.
Create a scatter plot of the raw data and an overlaid scatter plot with centroids for k = 15.
'''
# Convert rdd_split_int RDD into Spark DataFrame
rdd_split_int_df = spark.createDataFrame(rdd_split_int, schema=["col1", "col2"])

# Convert Spark DataFrame into Pandas DataFrame
rdd_split_int_df_pandas = rdd_split_int_df.toPandas()

# Convert cluster_centers into Panda DataFrame
cluster_centers_pandas = pd.DataFrame(cluster_centers, columns=["col1", "col2"])

# Create an overlaid scatter plot
plt.scatter(rdd_split_int_df_pandas["col1"], rdd_split_int_df_pandas["col2"])
plt.scatter(cluster_centers_pandas["col1"], cluster_centers_pandas["col2"], color="red", marker="x")
plt.show()