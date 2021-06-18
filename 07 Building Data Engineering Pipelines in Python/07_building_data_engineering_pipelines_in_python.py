# Course 7
# Building Data Engineering Pipelines in Python

#Chapter 1
#Ingesting Data

#Working with JSON
''''
Import the Python module we need to deal with JSON.
Open a file with the name database_config.json for writing (and only for writing).
Serialize the database_address dictionary as JSON and write it to the open file handle. If you’re unsure which arguments the function takes, type ?json.dump to get more information on what obj and fp expect.
'''
# Import json
import json

database_address = {
  "host": "10.0.0.5",
  "port": 8456
}

# Open the configuration file in writable mode
with open("database_config.json", "w") as fh:
  # Serialize the object in this file handle
  json.dump(obj=database_address, fp=fh)

#Specifying the schema of the data
''''
Infer from the example above the name and the data type of each component of the store’s items. Complete the JSON schema object with this information.
Write that schema, using the write_schema() function, to the "products" stream using the Singer API.
'''
# Complete the JSON schema
schema = {'properties': {
    'brand': {'type': 'string'},
    'model': {'type': 'string'},
    'price': {'type': 'number'},
    'currency': {'type': 'string'},
    'quantity': {'type': 'number', 'minimum': 1},  
    'date': {'type': 'string', 'format': 'date'},
    'countrycode': {'type': 'string', 'pattern': "^[A-Z]{2}$"}, 
    'store_name': {'type': 'string'}}}

# Write the schema
singer.write_schema(stream_name="products", schema=schema, key_properties=[])

#Communicating with an API
''''
Fill in the correct API key.
Create the URL of the web API by completing the template URL above. You need to pass the endpoint first and then the API key.
Use that URL in the call to requests.get() so that you may see what more the API can tell you about itself.
'''
endpoint = "http://localhost:5000"

# Fill in the correct API key
api_key = "scientist007"

# Create the web API’s URL
authenticated_endpoint = "{}/{}".format(endpoint, api_key)

# Get the web API’s reply to the endpoint
api_response = requests.get(authenticated_endpoint).json()
pprint.pprint(api_response)

''''
Take a look at the output in the console from the previous step. Notice that it is a list of endpoints, each containing a description of the content found at the endpoint and the template for the URL to access it. The template can be filled in, like you did in the previous step.

Complete the URL that should give you back a list of all shops that were scraped by the marketing team.
'''
# Create the API’s endpoint for the shops
shops_endpoint = "{}/{}/{}/{}".format(endpoint, api_key, "diaper/api/v1.0", "shops")
shops = requests.get(shops_endpoint).json()
print(shops)

''''
Take a look at the output in the console from the previous step. The shops variable contains the list of all shops known by the web API.

From the shops variable, find the one that starts with the letter “D”. Use it in the second (templated) url that was shown by the call to pprint.pprint(api_response), to list the items of this specific shop. You must use the appropriate url endpoint, combined with the http://localhost:5000, similar to how you completed the previous step.
'''
# Create the API’s endpoint for items of the shop starting with a "D"
items_of_specific_shop_URL = "{}/{}/{}/{}/{}".format(endpoint, api_key, "diaper/api/v1.0", "items", "DM")
products_of_shop = requests.get(items_of_specific_shop_URL).json()
pprint.pprint(products_of_shop)

#Streaming records

#Retrieve the products of the shop called Tesco.
# Use the convenience function to query the API
tesco_items = retrieve_products("Tesco")

singer.write_schema(stream_name="products", schema=schema,
                    key_properties=[])

''''
Based on the output of the previous step, use the function write_record() to write one of these products to the products stream, which is where you also wrote the schema to. Make sure to add to the product a key-value pair that is mentioned in the schema, but is missing from the product, so that the record you write complies with the schema.
'''
# Write a single record to the stream, that adheres to the schema
singer.write_record(stream_name="products", 
                    record={**tesco_items[0], "store_name": "Tesco"})

''''
Now use the more appropriate function write_records() to write all items for all shops exposed by the API. As you don’t know a priori how big the dataset will be, you will be using a generator expression in which you enrich the items with the store_name one at a time.
'''
# Use the convenience function to query the API
tesco_items = retrieve_products("Tesco")

singer.write_schema(stream_name="products", schema=schema,
                    key_properties=[])

# Write a single record to the stream, that adheres to the schema
singer.write_record(stream_name="products", 
                    record={**tesco_items[0], "store_name": "Tesco"})

for shop in requests.get(SHOPS_URL).json()["shops"]:
    # Write all of the records that you retrieve from the API
    singer.write_records(
      stream_name="products", # Use the same stream name that you used in the schema
      records=({**item, "store_name": shop}
               for item in retrieve_products(shop))
    )

#Chapter 2
#Creating a data transformation pipeline with PySpark

#Reading a CSV file
''''
Create a DataFrameReader object using the spark.read property.
Make the reader object use the header of the CSV file to name the columns automatically, by passing in the correct keyword arguments to the reader’s .options() method.
'''
# Read a csv file and set the headers
df = (spark.read
      .options(header="true")
      .csv("/home/repl/workspace/mnt/data_lake/landing/ratings.csv"))

df.show()

#Defining a schema
''''
Define the schema for the spreadsheet that has the columns “brand”, “model”, “absorption_rate” and “comfort”, in that order.
Pass the predefined schema while loading the CSV file using the .schema() method.
'''
# Define the schema
schema = StructType([
  StructField("brand", StringType(), nullable=False),
  StructField("model", StringType(), nullable=False),
  StructField("absorption_rate", ByteType(), nullable=True),
  StructField("comfort", ByteType(), nullable=True)
])

better_df = (spark
             .read
             .options(header="true")
             # Pass the predefined schema to the Reader
             .schema(schema)
             .csv("/home/repl/workspace/mnt/data_lake/landing/ratings.csv"))
print(better_df.dtypes)

#Removing invalid rows

#Remove any invalid rows by passing the correct keyword (and associated value) to the reader’s .options() method.
# Specify the option to drop invalid rows
ratings = (spark
           .read
           .options(header=True, mode="DROPMALFORMED")
           .csv("/home/repl/workspace/mnt/data_lake/landing/ratings_with_invalid_rows.csv"))
ratings.show()

#Filling unknown data
#Fill the incomplete rows, by supplying the default numeric value of 4 for the comfort column.
print("BEFORE")
ratings.show()

print("AFTER")
# Replace nulls with arbitrary value on column subset
ratings = ratings.fillna(4, subset=["comfort"])
ratings.show()

#Conditionally replacing values
''''
Use the .withColumn() method to relabel the column named “comfort”.
Use the when() function to replace values of the “comfort” column larger than 3 with the string "sufficient".
Use the .otherwise() method to replace remaining values with "insufficient".
'''
from pyspark.sql.functions import col, when

# Add/relabel the column
categorized_ratings = ratings.withColumn(
    "comfort",
    # Express the condition in terms of column operations
    when(col("comfort") > 3, "sufficient").otherwise("insufficient"))

categorized_ratings.show()

#Selecting and renaming columns
''''
Select the columns “brand”, “model” and “absorption_rate” from the ratings DataFrame.
Rename the “absorption_rate” column to “absorbency”.
Show only the distinct values of the resulting DataFrame.
'''
from pyspark.sql.functions import col

# Select the columns and rename the "absorption_rate" column
result = ratings.select([col("brand"),
                       col("model"),
                       col("absorption_rate").alias("absorbency")])

# Show only unique values
result.distinct().show()

#Grouping and aggregating data
''''
Use the .groupBy() method to group the data by the “Country” column.
In these groups, compute the average of the “Salary” column and name the resulting column “average_salary”.
Compute the standard deviation of the “Salary” column in each group in the same aggregation.
Retrieve the largest “Salary” in each group, in the same aggregation, and name the resulting column “highest_salary”.
'''
from pyspark.sql.functions import col, avg, stddev_samp, max as sfmax

aggregated = (purchased
              # Group rows by 'Country'
              .groupBy(col('Country'))
              .agg(
                # Calculate the average salary per group and rename
                avg('Salary').alias('average_salary'),
                # Calculate the standard deviation per group
                stddev_samp('Salary'),
                # Retain the highest salary per group and rename
                sfmax('Salary').alias('highest_salary')
              )
             )

aggregated.show()

#Chapter 3
#Testing your Data Pipeline

#Creating in-memory DataFrames
''''
Use the Record class, which has the 5 instance attributes given in the Row class constructor, to create a tuple of Row-like records, that will be assigned to the data variable.
Use the createDataFrame() function to create a Spark DataFrame.
'''
from datetime import date
from pyspark.sql import Row

Record = Row("country", "utm_campaign", "airtime_in_minutes", "start_date", "end_date")

# Create a tuple of records
data = (
  Record("USA", "DiapersFirst", 28, date(2017, 1, 20), date(2017, 1, 27)),
  Record("Germany", "WindelKind", 31, date(2017, 1, 25), None),
  Record("India", "CloseToCloth", 32, date(2017, 1, 25), date(2017, 2, 2))
)

# Create a DataFrame from these records
frame = spark.createDataFrame(data)
frame.show()

#Chapter 4
#Managing and orchestrating a workflow

#Specifying the DAG schedule
#Complete the cron expression by using the schedule_interval to represent every Monday, at 7 o’clock in the morning.
from datetime import datetime
from airflow import DAG

reporting_dag = DAG(
    dag_id="publish_EMEA_sales_report", 
    # Insert the cron expression
    schedule_interval="0 7 * * 1",
    start_date=datetime(2019, 11, 24),
    default_args={"owner": "sales"}
)

#Specifying operator dependencies
''''
Set prepare_crust to precede apply_tomato_sauce using the appropriate method.
Set apply_tomato_sauceto precede each of tasks in tasks_with_tomato_sauce_parent using the appropriate method.
Set the tasks_with_tomato_sauce_parent list to precede bake_pizza using either the bitshift operator >> or <<.
Set bake_pizza to succeed prepare_oven using the appropriate method.
'''
# Specify direction using verbose method
prepare_crust.set_downstream(apply_tomato_sauce)

tasks_with_tomato_sauce_parent = [add_cheese, add_ham, add_olives, add_mushroom]
for task in tasks_with_tomato_sauce_parent:
    # Specify direction using verbose method on relevant task
    apply_tomato_sauce.set_downstream(task)

# Specify direction using bitshift operator
tasks_with_tomato_sauce_parent >> bake_pizza

# Specify direction using verbose method
bake_pizza.set_upstream(prepare_oven)

#Preparing a DAG for daily pipelines
# Create a DAG object
dag = DAG(
  dag_id='optimize_diaper_purchases',
  default_args={
    # Don't email on failure
    'email_on_failure': False,
    # Specify when tasks should have started earliest
    'start_date': datetime(2019, 6, 25)
  },
  # Run the DAG daily
  schedule_interval='@daily')

#Scheduling bash scripts with Airflow
''''
Assign the task an id of ingest_data.
Pipe the output from tap-marketing-api to target-csv, using the bash_command argument of the BashOperator. Pass the reference to the data_lake.conf as the value to target-csv’s --config flag. The command you construct in this way should be equivalent to what you’ve executed in the last exercise of chapter 1.
'''
config = os.path.join(os.environ["AIRFLOW_HOME"], 
                      "scripts",
                      "configs", 
                      "data_lake.conf")

ingest = BashOperator(
  # Assign a descriptive id
  task_id="ingest_data", 
  # Complete the ingestion pipeline
  bash_command='tap-marketing-api | target-csv --config %s' % config,
  dag=dag)

#Scheduling Spark jobs with Airflow
''''
Import the SparkSubmitOperator.
Set the path for entry_point by joining the AIRFLOW_HOME environment variable and scripts/clean_ratings.py.
Set the path for dependency_path by joining the AIRFLOW_HOME environment variable and dependencies/pydiaper.zip.
Complete the clean_data task by passing a reference to the file that starts the Spark job and the additional files the job will use.
'''
# Import the operator
from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator

# Set the path for our files.
entry_point = os.path.join(os.environ["AIRFLOW_HOME"], "scripts", "clean_ratings.py")
dependency_path = os.path.join(os.environ["AIRFLOW_HOME"], "dependencies", "pydiaper.zip")

with DAG('data_pipeline', start_date=datetime(2019, 6, 25),
         schedule_interval='@daily') as dag:
  	# Define task clean, running a cleaning job.
    clean_data = SparkSubmitOperator(
        application=entry_point, 
        py_files=dependency_path,
        task_id='clean_data',
        conn_id='spark_default')

#Scheduling the full data pipeline with Airflow
''''
Use the correct operators for the ìngest (a bash task), clean (a Spark job) and insight (another Spark job) tasks.
Define the order in which the tasks should be run.
'''
spark_args = {"py_files": dependency_path,
              "conn_id": "spark_default"}
# Define ingest, clean and transform job.
with dag:
    ingest = BashOperator(task_id='Ingest_data', bash_command='tap-marketing-api | target-csv --config %s' % config)
    clean = SparkSubmitOperator(application=clean_path, task_id='clean_data', **spark_args)
    insight = SparkSubmitOperator(application=transform_path, task_id='show_report', **spark_args)
    
    # set triggering sequence
    ingest >> clean >> insight