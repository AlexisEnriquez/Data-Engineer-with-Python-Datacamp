# Course 6
# Introduction to Pyspark

#Chapter 1
#Getting to know PySpark

#Examining The SparkContext
''''
Call print() on sc to verify there's a SparkContext in your environment.
print() sc.version to see what version of Spark is running on your cluster.
'''
# Verify SparkContext
print(sc)

# Print Spark version
print(sc.version)

#Creating a SparkSession
''''
Import SparkSession from pyspark.sql.
Make a new SparkSession called my_spark using SparkSession.builder.getOrCreate().
Print my_spark to the console to verify it's a SparkSession.
'''
# Import SparkSession from pyspark.sql
from pyspark.sql import SparkSession

# Create my_spark
my_spark = SparkSession.builder.getOrCreate()

# Print my_spark
print(my_spark)

#Viewing tables
#See what tables are in your cluster by calling spark.catalog.listTables() and printing the result!
# Print the tables in the catalog
print(spark.catalog.listTables())

#Are you query-ious?
''''
Use the .sql() method to get the first 10 rows of the flights table and save the result to flights10. The variable query contains the appropriate SQL query.
Use the DataFrame method .show() to print flights10.
'''
# Don't change this query
query = "FROM flights SELECT * LIMIT 10"

# Get the first 10 rows of flights
flights10 = spark.sql(query)

# Show the results
flights10.show()

#Pandafy a Spark DataFrame
''''
Run the query using the .sql() method. Save the result in flight_counts.
Use the .toPandas() method on flight_counts to create a pandas DataFrame called pd_counts.
Print the .head() of pd_counts to the console.
'''
# Don't change this query
query = "SELECT origin, dest, COUNT(*) as N FROM flights GROUP BY origin, dest"

# Run the query
flight_counts = spark.sql(query)

# Convert the results to a pandas DataFrame
pd_counts = flight_counts.toPandas()

# Print the head of pd_counts
print(pd_counts.head())

#Put some Spark in your data
''''
The code to create a pandas DataFrame of random numbers has already been provided and saved under pd_temp.
Create a Spark DataFrame called spark_temp by calling the .createDataFrame() method with pd_temp as the argument.
Examine the list of tables in your Spark cluster and verify that the new DataFrame is not present. Remember you can use spark.catalog.listTables() to do so.
Register spark_temp as a temporary table named "temp" using the .createOrReplaceTempView() method. Remember that the table name is set including it as the only argument!
Examine the list of tables again!
'''
# Create pd_temp
pd_temp = pd.DataFrame(np.random.random(10))

# Create spark_temp from pd_temp
spark_temp = spark.createDataFrame(pd_temp)

# Examine the tables in the catalog
print(spark.catalog.listTables())

# Add spark_temp to the catalog
spark_temp.createOrReplaceTempView("temp")

# Examine the tables in the catalog again
print(spark.catalog.listTables())

#Dropping the middle man
''''
Use the .read.csv() method to create a Spark DataFrame called airports
The first argument is file_path
Pass the argument header=True so that Spark knows to take the column names from the first line of the file.
Print out this DataFrame by calling .show().
'''
# Don't change this file path
file_path = "/usr/local/share/datasets/airports.csv"

# Read in the airports data
airports = spark.read.csv(file_path, header=True)

# Show the data
airports.show()

#Creating columns
''''
Use the spark.table() method with the argument "flights" to create a DataFrame containing the values of the flights table in the .catalog. Save it as flights.
Show the head of flights using flights.show(). The column air_time contains the duration of the flight in minutes.
Update flights to include a new column called duration_hrs, that contains the duration of each flight in hours.
'''
# Create the DataFrame flights
flights = spark.table("flights")

# Show the head
flights.show()

# Add duration_hrs
flights = flights.withColumn("duration_hrs", flights.air_time/60)

#Filtering Data
# Filter flights by passing a string
long_flights1 = flights.filter("distance > 1000")

# Filter flights by passing a column of boolean values
long_flights2 = flights.filter(flights.distance > 1000)

# Print the data to check they're equal
long_flights1.show()
long_flights2.show()

#Selecting
''''
Select the columns tailnum, origin, and dest from flights by passing the column names as strings. Save this as selected1.
Select the columns origin, dest, and carrier using the df.colName syntax and then filter the result using both of the filters already defined for you (filterA and filterB) to only keep flights from SEA to PDX. Save this as selected2.
'''
# Select the first set of columns
selected1 = flights.select("tailnum", "origin", "dest")

# Select the second set of columns
temp = flights.select(flights.origin, flights.dest, flights.carrier)

# Define first filter
filterA = flights.origin == "SEA"

# Define second filter
filterB = flights.dest == "PDX"

# Filter the data, first by filterA then by filterB
selected2 = temp.filter(filterA).filter(filterB)

#Selecting II
''''
Calculate average speed by dividing the distance by the air_time (converted to hours). Use the .alias() method name this column "avg_speed". Save the output as the variable avg_speed.
Select the columns "origin", "dest", "tailnum", and avg_speed (without quotes!). Save this as speed1.
Create the same table using .selectExpr() and a string containing a SQL expression. Save this as speed2.
'''
# Define avg_speed
avg_speed = (flights.distance/(flights.air_time/60)).alias("avg_speed")

# Select the correct columns
speed1 = flights.select("origin", "dest", "tailnum", avg_speed)

# Create the same table using a SQL expression
speed2 = flights.selectExpr("origin", "dest", "tailnum", "distance/(air_time/60) as avg_speed")

#Aggregating
''''
Find the length of the shortest (in terms of distance) flight that left PDX by first .filter()ing and using the .min() method. Perform the filtering by referencing the column directly, not passing a SQL string.
Find the length of the longest (in terms of time) flight that left SEA by filter()ing and using the .max() method. Perform the filtering by referencing the column directly, not passing a SQL string.
'''
# Find the shortest flight from PDX in terms of distance
flights.filter(flights.origin == "PDX").groupBy().min("distance").show()

# Find the longest flight from SEA in terms of air time
flights.filter(flights.origin == "SEA").groupBy().max("air_time").show()

#Aggregating II
''''
Use the .avg() method to get the average air time of Delta Airlines flights (where the carrier column has the value "DL") that left SEA. The place of departure is stored in the column origin. show() the result.
Use the .sum() method to get the total number of hours all planes in this dataset spent in the air by creating a column called duration_hrs from the column air_time. show() the result.
'''
# Average duration of Delta flights
flights.filter(flights.carrier == "DL").filter(flights.origin == "SEA").groupBy().avg("air_time").show()

# Total hours in the air
flights.withColumn("duration_hrs", flights.air_time/60).groupBy().sum("duration_hrs").show()

#Grouping and Aggregating I
''''
Create a DataFrame called by_plane that is grouped by the column tailnum.
Use the .count() method with no arguments to count the number of flights each plane made.
Create a DataFrame called by_origin that is grouped by the column origin.
Find the .avg() of the air_time column to find average duration of flights from PDX and SEA.
'''
# Group by tailnum
by_plane = flights.groupBy("tailnum")

# Number of flights each plane made
by_plane.count().show()

# Group by origin
by_origin = flights.groupBy("origin")

# Average duration of flights from PDX and SEA
by_origin.avg("air_time").show()

#Grouping and Aggregating II
''''
Import the submodule pyspark.sql.functions as F.
Create a GroupedData table called by_month_dest that's grouped by both the month and dest columns. Refer to the two columns by passing both strings as separate arguments.
Use the .avg() method on the by_month_dest DataFrame to get the average dep_delay in each month for each destination.
Find the standard deviation of dep_delay by using the .agg() method with the function F.stddev().
'''
# Import pyspark.sql.functions as F
import pyspark.sql.functions as F

# Group by month and dest
by_month_dest = flights.groupBy("month", "dest")

# Average departure delay by month and destination
by_month_dest.avg("dep_delay").show()

# Standard deviation of departure delay
by_month_dest.agg(F.stddev("dep_delay")).show()

#Joining II
''''
Examine the airports DataFrame by calling .show(). Note which key column will let you join airports to the flights table.
Rename the faa column in airports to dest by re-assigning the result of airports.withColumnRenamed("faa", "dest") to airports.
Join the flights with the airports DataFrame on the dest column by calling the .join() method on flights. Save the result as flights_with_airports.
The first argument should be the other DataFrame, airports.
The argument on should be the key column.
The argument how should be "leftouter".
Call .show() on flights_with_airports to examine the data again. Note the new information that has been added.
'''
# Examine the data
airports.show()

# Rename the faa column
airports = airports.withColumnRenamed("faa", "dest")

# Join the DataFrames
flights_with_airports = flights.join(airports, on="dest", how="leftouter")

# Examine the new DataFrame
flights_with_airports.show()

#Chapter 3
#Getting started with machine learning pipelines
''''
First, rename the year column of planes to plane_year to avoid duplicate column names.
Create a new DataFrame called model_data by joining the flights table with planes using the tailnum column as the key.
'''
# Rename year column
planes = planes.withColumnRenamed("year", "plane_year")

# Join the DataFrames
model_data = flights.join(planes, on="tailnum", how="leftouter")

#String to integer
''''
Use the method .withColumn() to .cast() the following columns to type "integer". Access the columns using the df.col notation:
model_data.arr_delay
model_data.air_time
model_data.month
model_data.plane_year
'''
# Cast the columns to integers
model_data = model_data.withColumn("arr_delay", model_data.arr_delay.cast("integer"))
model_data = model_data.withColumn("air_time", model_data.air_time.cast("integer"))
model_data = model_data.withColumn("month", model_data.month.cast("integer"))
model_data = model_data.withColumn("plane_year", model_data.plane_year.cast("integer"))

#Create a new column
''''
Create the column plane_age using the .withColumn() method and subtracting the year of manufacture (column plane_year) from the year (column year) of the flight.
'''
# Create the column plane_age
model_data = model_data.withColumn("plane_age", model_data.year - model_data.plane_year)

#Making a Boolean
'''
Use the .withColumn() method to create the column is_late. This column is equal to model_data.arr_delay > 0.
Convert this column to an integer column so that you can use it in your model and name it label (this is the default name for the response variable in Spark's machine learning routines).
Filter out missing values (this has been done for you).
'''
# Create is_late
model_data = model_data.withColumn("is_late", model_data.arr_delay > 0)

# Convert to an integer
model_data = model_data.withColumn("label",  model_data.is_late.cast("integer"))

# Remove missing values
model_data = model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")

#Carrier
''''
Create a StringIndexer called carr_indexer by calling StringIndexer() with inputCol="carrier" and outputCol="carrier_index".
Create a OneHotEncoder called carr_encoder by calling OneHotEncoder() with inputCol="carrier_index" and outputCol="carrier_fact".
'''
# Create a StringIndexer
carr_indexer = StringIndexer(inputCol="carrier", outputCol="carrier_index")

# Create a OneHotEncoder
carr_encoder = OneHotEncoder(inputCol="carrier_index",outputCol="carrier_fact")

#Destination
''''
Create a StringIndexer called dest_indexer by calling StringIndexer() with inputCol="dest" and outputCol="dest_index".
Create a OneHotEncoder called dest_encoder by calling OneHotEncoder() with inputCol="dest_index" and outputCol="dest_fact".
'''
# Create a StringIndexer
dest_indexer = StringIndexer(inputCol="dest", outputCol="dest_index")

# Create a OneHotEncoder
dest_encoder = OneHotEncoder(inputCol="dest_index",outputCol="dest_fact")

#Assemble a vector
''''
Create a VectorAssembler by calling VectorAssembler() with the inputCols names as a list and the outputCol name "features".
The list of columns should be ["month", "air_time", "carrier_fact", "dest_fact", "plane_age"].
'''
# Make a VectorAssembler
vec_assembler = VectorAssembler(inputCols=["month", "air_time", "carrier_fact", "dest_fact", "plane_age"], outputCol="features")

#Create the pipeline
''''
Import Pipeline from pyspark.ml.
Call the Pipeline() constructor with the keyword argument stages to create a Pipeline called flights_pipe.
stages should be a list holding all the stages you want your data to go through in the pipeline. Here this is just: [dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler]
'''
# Import Pipeline
from pyspark.ml import Pipeline

# Make the pipeline
flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])

#Transform the data
''''
Create the DataFrame piped_data by calling the Pipeline methods .fit() and .transform() in a chain. Both of these methods take model_data as their only argument.
'''
# Fit and transform the data
piped_data = flights_pipe.fit(model_data).transform(model_data)

#Split the data
''''
Use the DataFrame method .randomSplit() to split piped_data into two pieces, training with 60% of the data, and test with 40% of the data by passing the list [.6, .4] to the .randomSplit() method.
'''
# Split the data into training and test sets
training, test = piped_data.randomSplit([.6, .4])

#Chapter 4
#Model Tuning and Selection

#Create the modeler
''''
Import the LogisticRegression class from pyspark.ml.classification.
Create a LogisticRegression called lr by calling LogisticRegression() with no arguments.
'''
# Import LogisticRegression
from pyspark.ml.classification import LogisticRegression

# Create a LogisticRegression Estimator
lr = LogisticRegression()

#Create the evaluator
''''
Import the submodule pyspark.ml.evaluation as evals.
Create evaluator by calling evals.BinaryClassificationEvaluator() with the argument metricName="areaUnderROC".
'''
# Import the evaluation submodule
import pyspark.ml.evaluation as evals

# Create a BinaryClassificationEvaluator
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")

#Make a grid
''''
Import the submodule pyspark.ml.tuning under the alias tune.
Call the class constructor ParamGridBuilder() with no arguments. Save this as grid.
Call the .addGrid() method on grid with lr.regParam as the first argument and np.arange(0, .1, .01) as the second argument. This second call is a function from the numpy module (imported as np) that creates a list of numbers from 0 to .1, incrementing by .01. Overwrite grid with the result.
Update grid again by calling the .addGrid() method a second time create a grid for lr.elasticNetParam that includes only the values [0, 1].
Call the .build() method on grid and overwrite it with the output.
'''
# Import the tuning submodule
import pyspark.ml.tuning as tune

# Create the parameter grid
grid = tune.ParamGridBuilder()

# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])

# Build the grid
grid = grid.build()

#Make the validator
''''
Create a CrossValidator by calling tune.CrossValidator() with the arguments:
estimator=lr
estimatorParamMaps=grid
evaluator=evaluator
Name this object cv.
'''

# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr,
               estimatorParamMaps=grid,
               evaluator=evaluator
               )

#Fit the model(s)
''''
Create best_lr by calling lr.fit() on the training data.
Print best_lr to verify that it's an object of the LogisticRegressionModel class.
'''
# Call lr.fit()
best_lr = lr.fit(training)

# Print best_lr
print(best_lr)

#Evaluate the model
''''
Use your model to generate predictions by applying best_lr.transform() to the test data. Save this as test_results.
Call evaluator.evaluate() on test_results to compute the AUC. Print the output.
'''
# Use the model to predict the test set
test_results = best_lr.transform(test)

# Evaluate the predictions
print(evaluator.evaluate(test_results))