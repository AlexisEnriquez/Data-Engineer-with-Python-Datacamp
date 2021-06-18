# Course 4
# Introduction to Bash Scripting

#Chapter 1
#From Command-Line to Bash Script

''''
Create a single-line script that concatenates the mentioned file.
Save your script and run from the console.
'''

#!/bin/bash

# Concatenate the file
cat server_log_with_todays_date.txt

# Now save and run!

#Shell pipelines to Bash scripts
''''
Create a single-line pipe to cat the file, cut out the relevant field and aggregate (sort & uniq -c will help!) based on winning team.
Save your script and run from the console.
'''
#!/bin/bash

# Create a single-line pipe
cat soccer_scores.csv | cut -d "," -f 2 | tail -n +2 | sort | uniq -c

# Now save and run!

#Extract and edit using Bash scripts
''''
Create a pipe using sed twice to change the team Cherno to Cherno City first, and then Arda to Arda United.
Pipe the output to a file called soccer_scores_edited.csv.
Save your script and run from the console. Try opening soccer_scores_edited.csv using shell commands to confirm it worked (the first line should be changed)!
'''
#!/bin/bash

# Create a sed pipe to a new file
cat soccer_scores.csv | sed 's/Cherno/Cherno City/g' | sed 's/Arda/Arda United/g' > soccer_scores_edited.csv

# Now save and run!

#Using arguments in Bash scripts
''''
Echo the first and second ARGV arguments.
Echo out the entire ARGV array in one command (not each element).
Echo out the size of ARGV (how many arguments fed in).
Save your script and run from the terminal pane using the arguments Bird Fish Rabbit. Don't use the ./script.sh method.
'''
# Echo the first and second ARGV arguments
echo $1 
echo $2

# Echo out the entire ARGV array
echo $@

# Echo out the size of ARGV
echo $#

#Using arguments with HR data

''''
Echo the first ARGV argument so you can confirm it is being read in.
cat all the files in the directory /hire_data and pipe to grep to filter using the city name (your first ARGV argument).
On the same line, pipe out the filtered data to a new CSV called cityname.csv where cityname is taken from the first ARGV argument.
Save your script and run from the console twice (do not use the ./script.sh method). Once with the argument Seoul. Then once with the argument Tallinn.
'''
# Echo the first ARGV argument
echo $1 

# Cat all the files
# Then pipe to grep using the first ARGV argument
# Then write out to a named csv using the first ARGV argument
cat hire_data/* | grep "$1" > "$1".csv

#Using variables in Bash
''''
Create a variable, yourname that contains the name of the user. Let's use the test name 'Sam' for this.
Fix the echo statement so it prints the variable and not the word yourname.
Run your script.
'''
# Create the required variable
yourname="Sam"

# Print out the assigned name (Help fix this error!)
echo "Hi there $yourname, welcome to the website!"

#Converting Fahrenheit to Celsius
''''
Create a variable temp_f from the first ARGV argument.
Call a shell-within-a-shell to subtract 32 from temp_f and assign to variable temp_f2.
Using the same method, multiply temp_f2 by 5 and divide by 9, assigning to a new variable temp_c then print out temp_c.
Save and run your script (in the terminal) using 108 Fahrenheit (the forecast temperature in Parramatta, Sydney this Saturday!).
'''
# Get first ARGV into variable
temp_f=$1

# Subtract 32
temp_f2=$(echo "scale=2; $temp_f - 32" | bc)

# Multiply by 5/9
temp_c=$(echo "scale=2; $temp_f2 * 5 / 9" | bc)

# Print the celsius temp
echo $temp_c

# Create three variables from the temp data files' contents

# Create three variables from the temp data files' contents
temp_a=$(cat temps/region_A)
temp_b=$(cat temps/region_B)
temp_c=$(cat temps/region_C)

# Print out the three variables
echo "The three temperatures were $temp_a, $temp_b, and $temp_c"

#Creating an array

''''
Create a normal array called capital_cities which contains the cities Sydney, New York and Paris. Do not use the declare method; fill the array as you create it. Be sure to put double quotation marks around each element!
'''
# Create a normal array with the mentioned elements
capital_cities=("Sydney" "New York" "Paris")

''''
Create a normal array called capital_cities. However, use the declare method to create in this exercise.
Below, add each city, appending to the array. The cities were Sydney, New York, and Paris. Remember to use double quotation marks.
'''
# Create a normal array with the mentioned elements using the declare method
declare -a capital_cities

# Add (append) the elements
capital_cities+=("Sydney")
capital_cities+=("New York")
capital_cities+=("Paris")

''''
Now you have the array created, print out the entire array using a special array property.
Then print out the length of the array using another special property.
'''
# The array has been created for you
capital_cities=("Sydney" "New York" "Paris")

# Print out the entire array
echo ${capital_cities[@]}

# Print out the array length
echo ${capital_cities[@]}

#Creating associative arrays
''''
Create an empty associative array on one line called model_metrics. Do not add any elements to it here.
Add the following key-value pairs; (model_accuracy, 98), (model_name, "knn"), (model_f1, 0.82).
'''
# Create empty associative array
declare -A model_metrics

# Add the key-value pairs
model_metrics[model_accuracy]=98
model_metrics[model_name]="knn"
model_metrics[model_f1]=0.82

''''
Create the same associative array (model_metrics) all in one line. (model_accuracy, 98), (model_name, "knn"), (model_f1, 0.82). Remember you must add square brackets* around the keys!
Print out the array to see what you created.
'''
# Declare associative array with key-value pairs on one line
declare -A model_metrics=([model_accuracy]=98 [model_name]="knn" [model_f1]=0.82)

# Print out the entire array
echo ${model_metrics[@]}

#Now that you've created an associative array, print out just the keys of this associative array.
# An associative array has been created for you
declare -A model_metrics=([model_accuracy]=98 [model_name]="knn" [model_f1]=0.82)

# Print out just the keys
echo ${!model_metrics[@]}

#Climate calculations in Bash
''''
Create an array with the two temp variables as elements.
Call an external program to get the average temperature. You will need to sum array elements then divide by 2. Use the scale parameter to ensure this is to 2 decimal places.
Append this new variable to your array and print out the entire array.
Run your script.
'''
# Create variables from the temperature data files
temp_b="$(cat temps/region_B)"
temp_c="$(cat temps/region_C)"

# Create an array with these variables as elements
region_temps=($temp_b $temp_c)

# Call an external program to get average temperature
average_temp=$(echo "scale=2; (${region_temps[0]} + ${region_temps[1]}) / 2" | bc)

# Append to array
region_temps+=($average_temp)

# Print out the whole array
echo ${region_temps[@]}

#Use a FOR loop on files in directory
''''
Use a FOR statement to loop through files that end in .R in inherited_folder/ using a glob expansion.
echo out each file name into the console.
'''
# Use a FOR loop on files in directory
for file in inherited_folder/*.R
do  
    # Echo out each file
    echo $file
done

#Cleaning up a directory
# Create a FOR statement on files in directory
for file in robs_files/*.py
do  
    # Create IF statement using grep
    if grep -q 'RandomForestClassifier' $file ; then
        # Move wanted files to to_keep/ folder
        mv $file to_keep/
    fi
done

#Days of the week with CASE
''''
Build a CASE statement that matches on the first ARGV element.
Create a match on each weekday such as Monday, Tuesday etc. using OR syntax on a single line, then a match on each weekend day (Saturday and Sunday) etc. using OR syntax on a single line.
Create a default match that prints out Not a day! if none of the above patterns are matched.
Save your script and run in the terminal window with Wednesday and Saturday to test.
'''
# Create a CASE statement matching the first ARGV element
case $1 in
  # Match on all weekdays
  Monday|Tuesday|Wednesday|Thursday|Friday)
  echo "It is a Weekday!";;
  # Match on all weekend days
  Saturday|Sunday)
  echo "It is a Weekend!";;
  # Create a default
  *) 
  echo "Not a day!";;
esac

#Moving model results with CASE
''''
Use a FOR statement to loop through (using glob expansion) files in model_out/.
Use a CASE statement to match on the contents of the file (we will use cat and shell-within-a-shell to get the contents to match against). It must check if the text contains a tree-based model name and move to tree_models/, otherwise delete the file.
Create a default match that prints out Unknown model in FILE where FILE is the filename then run your script.
'''
# Use a FOR loop for each file in 'model_out'
for file in model_out/*
do
    # Create a CASE statement for each file's contents
    case $(cat $file) in
      # Match on tree and non-tree models
      *"Random Forest"*|*GBM*|*XGBoost*)
      mv $file tree_models/ ;;
      *KNN*|*Logistic*)
      rm $file ;;
      # Create a default
      *) 
      echo "Unknown model in $file" ;;
    esac
done

#Chapter 4
#Functions and Automation

#Uploading model results to the cloud
# Create function
function upload_to_cloud () {
  # Loop through files with glob expansion
  for file in output_dir/*results*
  do
    # Echo that they are being uploaded
    echo "Uploading $file to cloud"
  done
}

# Call the function
upload_to_cloud

#Get the current day
# Create function
what_day_is_it () {

  # Parse the results of date
  current_day=$(date | cut -d " " -f1)

  # Echo the result
  echo $current_day
}

# Call the function
what_day_is_it

#A percentage calculator
# Create a function 
function return_percentage () {

  # Calculate the percentage using bc
  percent=$(echo "scale=2; 100 * $1 / $2" | bc)

  # Return the calculated percentage
  echo $percent
}

# Call the function with 456 and 632 and echo the result
return_test=$(return_percentage 456 632)
echo "456 out of 632 as a percent is $return_test%"

#Sports analytics function
''''
Create a function called get_number_wins using the function-word method.
Create a variable inside the function called win_stats that takes the argument fed into the function to filter the last step of the shell-pipeline presented.
Call the function using the city Etar.
Below the function call, try to access the win_stats variable created inside the function in the echo command presented.
'''
# Create a function
function get_number_wins () {

  # Filter aggregate results by argument
  win_stats=$(cat soccer_scores.csv | cut -d "," -f2 | egrep -v 'Winner'| sort | uniq -c | egrep "$1")

}

# Call the function with specified argument
get_number_wins "Etar"

# Print out the global variable
echo "The aggregated stats are: $win_stats"

#Summing an array
''''
Create a function called sum_array and add a base variable (equal to 0) called sum with local scope. You will loop through the array and increment this variable.
Create a FOR loop through the ARGV array inside sum_array (hint: This is not $1! but another special array property) and increment sum with each element of the array.
Rather than assign to a global variable, echo back the result of your FOR loop summation.
Call your function using the test array provided and echo the result. You can capture the results of the function call using the shell-within-a-shell notation.
'''
# Create a function with a local base variable
function sum_array () {
  local sum=0
  # Loop through, adding to base variable
  for number in "$@"
  do
    sum=$(echo "$sum + $number" | bc)
  done
  # Echo back the result
  echo $sum
  }
# Call function with array
test_array=(14 12 23.5 16 19.34)
total=$(sum_array "${test_array[@]}")
echo "The total sum of the test array is $total"

#Creating cronjobs
''''
Create a crontab schedule that runs script1.sh at 30 minutes past 2am every day.
Create a crontab schedule that runs script2.sh every 15, 30 and 45 minutes past every hour.
Create a crontab schedule that runs script3.sh at 11.30pm on Sunday evening, every week. For this task, assume Sunday is the 0th day rather than the 7th day (as in some unix systems).
'''
# Create a schedule for 30 minutes past 2am every day
30 2 * * * bash script1.sh

# Create a schedule for every 15, 30 and 45 minutes past the hour
15,30,45 * * * * bash script2.sh

# Create a schedule for 11.30pm on Sunday evening, every week
30 23 * * 0 bash script3.sh