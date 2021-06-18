/*Course 10
Introduction to Scala
*/

//Chapter 1
//A Scalable Language

//Scala is object-oriented
// Calculate the difference between 8 and 5
val difference = 8-(5)

// Print the difference
println(difference)

//Define immutable variables (val)

// Define immutable variables for clubs 2♣ through 4♣
var twoClubs: Int = 2
var threeClubs: Int = 3
var fourClubs: Int = 4

//Don't try to change me
// Define immutable variables for player names
val playerA: String = "Alex"
val playerB: String = "Chen"
val playerC: String = "Umberto"

//Define mutable variables (var)
//Define four mutable variables, all of type Int and equal to 1: aceClubs, aceDiamonds, aceHearts, aceSpades.
// Define mutable variables for all aces
var aceClubs = 1
var aceDiamonds = 1
var aceHearts = 1
var aceSpades = 1

//You can change me
/*
Create a var named playerA and with the name Alex as a string literal as its value.
Change the point value of the ace of diamonds so it is worth 11 points, instead of the original value of 1 point it was assigned.
Add jackClubs and aceDiamonds to calculate the value of Alex's hand and print the result.
*/
// Create a mutable variable for player A (Alex)
var playerA = "Alex"


//Chapter 2
//Workflows, Functions, Collections

// Change the point value of A♦ from 1 to 11
aceDiamonds = 11

// Calculate hand value for J♣ and A♦
println(jackClubs + aceDiamonds)

//Call a function
/*
Calculate the hand value for playerA, who has the following cards: queenDiamonds, threeClubs, aceHearts (worth 1), fiveSpades.
Calculate the hand value for playerB, who has the following cards: kingHearts, jackHearts.
Call the maxHand function, passing in handPlayerA and handPlayerB as arguments. Pass this function call into the println function to print out the maximum hand value.
*/
// Calculate hand values
var handPlayerA: Int = queenDiamonds + threeClubs + aceHearts + fiveSpades
var handPlayerB: Int = kingHearts + jackHearts

// Find and print the maximum hand value
println(maxHand(handPlayerA, handPlayerB))

//Create and parameterize an array
//Create and parameterize an array (named hands) of type Int with a length of 3. Explicitly provide the type parameterization.
// Create and parameterize an array for a round of Twenty-One
val hands: Array[Int] = new Array[Int](3)

//Initialize an array
/*
Initialize the first player's hand in the hands array.
Initialize the second player's hand in the hands array.
Initialize the third player's hand in the hands array.
*/
// Create and parameterize an array for a round of Twenty-One
val hands: Array[Int] = new Array[Int](3)

// Initialize the first player's hand in the array
hands(0) = tenClubs + fourDiamonds

// Initialize the second player's hand in the array
hands(1) = nineSpades + nineHearts

// Initialize the third player's hand in the array
hands(2) = twoClubs + threeSpades

//An array, all at once
/*
Create an array named hands.
Initialize the first player's hand to tenClubs + fourDiamonds.
Initialize the second player's hand to nineSpades + nineHearts.
Initialize the third player's hand to twoClubs + threeSpades.
*/
// Create, parameterize, and initialize an array for a round of Twenty-One
val hands = Array(tenClubs + fourDiamonds,
                  nineSpades + nineHearts,
                  twoClubs + threeSpades)

//Updating arrays
/*
Add a fiveClubs to the first player's hand in hands.
Add a queenSpades to the second player's hand in hands.
Add a kingClubs to the third player's hand in hands.
*/
// Initialize player's hand and print out hands before each player hits
hands(0) = tenClubs + fourDiamonds
hands(1) = nineSpades + nineHearts
hands(2) = twoClubs + threeSpades
hands.foreach(println)

// Add 5♣ to the first player's hand
hands(0) = hands(0) + fiveClubs

// Add Q♠ to the second player's hand
hands(1) = hands(1) + queenSpades

// Add K♣ to the third player's hand
hands(2) = hands(2) + kingClubs

// Print out hands after each player hits
hands.foreach(println)

//Initialize and prepend to a list
/*
Initialize a list named prizes with an element for each round's prize, where the first through fifth round's prizes are 10, 15, 20, 25, and 30, respectively.
Prepend to prizes using the cons (::) operator so a new first round is added, worth $5. Name the new list newPrizes.
*/
// Initialize a list with an element for each round's prize
val prizes = List(10, 15, 20, 25, 30)
println(prizes)

// Prepend to prizes to add another round and prize
val newPrizes = 5 :: prizes
println(newPrizes)

//Initialize a list using cons and Nil
/*
Using :: and Nil, initialize a list named prizes with an element each round's prize, where the first through fifth round's prizes are 10, 15, 20, 25, and 30, respectively.
*/
// Initialize a list with an element each round's prize
val prizes = 10 :: 15 :: 20 :: 25 :: 30 :: Nil
println(prizes)

//Concatenate Lists
/*
Create the venuesNTOA list using List().
Create the venuesEuroTO list using the cons operator (::).
Concatenate venuesNTOA and venuesEuroTO to create a new list named venuesTOWorld.
*/
// The original NTOA and EuroTO venue lists
val venuesNTOA = List("The Grand Ballroom", "Atlantis Casino", "Doug's House")
val venuesEuroTO = "Five Seasons Hotel" :: "The Electric Unicorn" :: Nil

// Concatenate the North American and European venues
val venuesTOWorld = venuesNTOA ::: venuesEuroTO

//Chapter 3
//Type Systems, Control Structures, Style

//if and printing

// Point value of a player's hand
val hand = sevenClubs + kingDiamonds + threeSpades

// Congratulate the player if they have reached 21
if (hand == 21) {
  println("Twenty-One!")
}

//if expressions result in a value
// Point value of a player's hand
val hand = sevenClubs + kingDiamonds + threeSpades

// Inform a player where their current hand stands
val informPlayer: String = {
  if (hand > 21)
    "Bust! :("
  else if (hand == 21) 
    "Twenty-One! :)"
  else
    "Hit or stay?"
}

// Print the message
print(informPlayer)

//if and else inside of a function
// Find the number of points that will cause a bust
def pointsToBust(hand: Int): Int = {
  // If the hand is a bust, 0 points remain
  if (bust(hand))
    0
  // Otherwise, calculate the difference between 21 and the current hand
  else
    21 - hand
}

// Test pointsToBust with 10♠ and 5♣
val myHandPointsToBust = pointsToBust(tenSpades + fiveClubs)
println(myHandPointsToBust)

//A simple while loop
// Define counter variable
var i = 0

// Define the number of loop iterations
val numRepetitions = 3

// Loop to print a message for winner of the round
while (i < numRepetitions) {
  if (i < 2)
    println("winner")
  else
    println("chicken dinner")
  // Increment the counter variable
  i = i + 1
}

//Loop over a collection with while
// Define counter variable
var i = 0

// Create list with five hands of Twenty-One
var hands = List(16, 21, 8, 25, 4)

// Loop through hands
while (i < hands.length) {
  // Find and print number of points to bust
  println(pointsToBust(hands(i)))
  // Increment the counter variable
  i += 1
}

//Converting while to foreach
// Find the number of points that will cause a bust
def pointsToBust(hand: Int) = {
  // If the hand is a bust, 0 points remain
  if (bust(hand))
    println(0)
  // Otherwise, calculate the difference between 21 and the current hand
  else
    println(21 - hand)
}

// Create list with five hands of Twenty-One
var hands = List(16, 21, 8, 25, 4)

// Loop through hands, finding each hand's number of points to bust
hands.foreach(pointsToBust)