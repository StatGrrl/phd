# Phd Work

## Null game

The python code needs some work. 

I started by writing the code for each method in their own folders as we thought up different ways to do it. So if you run the script in one folder it will return a table of averages and create files with the results and plots for that method in the folder. The plot of the path is just for one iteration of the game.

I began consolidating the code into a function for one game in the compare paths folder. This is where I would add the game theory algorithm.

I still need to write a function to do the simulations - similar to the code in each folder but with functions for one game (for each method wanting to be compared) replacing the code inside the for loop. The simulation function should then return the averages and plots for each method and suggest which method is preferred.
