# Phd Work

## Null game

I started by writing the code for each method in their own folders as we thought up different ways to do it. So if you run the script in one folder it will return a table of averages and create files with the results and plots for that method in the folder. The plot of the path is just for one iteration of the game.

I began consolidating the code into a function for one game in the compare paths folder. This is where I would add the game theory algorithm. In the New folder there is a now a script containing all the functions necessary to do the simulations - similar to the code in each folder but with functions for one game (for each method wanting to be compared) replacing the code inside the for loop. The simulation function returns the averages and plots for each method.

## Improved Null Game - Spatial effects and structured movement

Still to be documented...
- Park class with method to create grid for park or subarea
- Player class with subclasses for Wildlife, Ranger and Poacher
- Methods and functions necessary for running simulations
- Simple pure strategy Stackelberg algorithm
