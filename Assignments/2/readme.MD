### Assignment 2: Search
#### Due Monday February 24 at 11:59pm. 

### 100 points.

Note: For the programming  portions of the assignment, please provide a file called submission.py that demos your code.
It should run BFS, DFS, DLS and IDS with all of the actions. It should also run A* with the SLD heuristic.

For the written portions, please prepare a document called assignment2.pdf and put your answers in there.
And don't forget to put your name on your assignment!

**On time management!** This assignment has many small components; the key to success will be starting early.
I've added suggested milestones _in italics_. 

_Feb 10: read through assignment. Identify any initial questions._

**Question 1. (10 points)**
(_Feb 12._)
1. What does it mean for a heuristic to be admissible? 

2. In class, we discussed two different heuristics for the 15-puzzle. The first, h1, was the
number of misplaced tiles. The second, h2, was the manhattan distance between each tile and
its correct location. 

Are h1 and h2 admissible? Please prove your answer.

Which heuristic is preferred? Please prove your answer. 

**Question 2** 

One of the first applications of search was in robot planning. In this setting, we define a state, a goal, and a set 
of actions. Our actions transform the state, creating a search space. We can then apply our classic search
algorithms to this problem.

We'll start with a simple version of the Mars Rover problem. This can be found in mars_planner.py. 
I've also included some unit tests to help you get things working. In later assignments, I'll ask you to build 
your own unit tests.

There are three locations: the station, the sample site, and the charger.
Our rover should travel from the station to the sample,  pick the sample up,
bring the sample back to the station, and then go to the charger to recharge. 

It will do this by finding a series of *actions* to take. Those actions should lead us to a state that satisfies the *goal*.

In this case, our state has five variables, indicating robot location, sample location, whether we're holding the sample, and whether we're charged. (there's also a pointer to the previous state.)

We also have a set of actions. Each action is implemented as a function that can be applied to a state.
This approach allows us a great deal of flexibility to solve a variety of problems and easily add or change 
our actions without breaking our existing code.

The RoverState then has a *successor function* that applies possible actions to our current state
and returns a list of tuples, which are the adjacent states and the action needed to get there.

1. **(5 points)**. (_Feb 12_) We need an __eq__ function in RoverState to detect repeated states. Implement this. 
Two states are equal if all of their instance variables are equal. I have provided a unit test that you can use to check this.


2. **(5 points)** (_Feb 13_) Now we need to create a *goal function*. This is a function that will return True if a state is the goal state,
and False otherwise. I've provided an example goal: a battery_goal test that returns True if we are at the battery and False otherwise. 

Create a mission_complete goal function that returns True if we are at the battery, charged, and the sample is at the station.


3. **(5 points)** (_Feb 15_) Run this with the included BFS and DFS implementations. Extend each of these to count the number of states generated. 
Print this out at the end. I have provided unit tests for BFS and DFS. 


4. **(5 points)** (_Feb 16_) Extend the depth_first_search function to implement *depth_limited_search* by using the optional *limit* parameter. 
When you are generating successors, only go to depth=limit in the search tree. You should extend the RoverState class 
by adding a depth variable to keep track of this. Add a unit test for DLS. 

 
5. **(10 points)** (_Feb 18_) Add an additional function for *iterative deepening search*. It should call depth-limited search. Count the total number of states generated. Add a unit test for this.


6. **(10 points)** (_Feb 20_) We found out that the rover cannot extract the sample on its own; it needs a tool. Extend the program as follows: 

Add a holding_tool instance variable to RoverState. Update your constructor and eq methods correctly.
add the following actions: 
- pick_up_tool 
- drop_tool 
- use_tool. 

Pick_up_tool should return a new state with the holding_tool 
variable set to True, drop_tool should return a new state with the holding_tool variable set to false, and use_tool should 
return a new state with the sample_extracted variable set to True, but only if we are holding the tool.

Run each of the four algorithms (breadth_first_search, depth_first_search, depth_limited_search, iterative deepening search) 
on this new problem and count the number of states generated. 

Please add to your written answers a table with the state data for each of the questions above.

**Question 3** 

For this problem, we will solve a classic path-finding problem using A*. This can be found in routefinder.py

We need to find a route for our rover to find its way from the sample site back to the charger. 
Fortunately, we have a map that we can use to find our way. 
There's a picture of it in the marsmap.docx file; the red cells are areas we cannot travel in.
We start at the sample site, with coordinates 4,4, and need to get back to the charger at 1,1.

This data is also captured in the MarsMap file. (note: please feel free to update this if you find errors.)

You will need to:

a) **(5 points)** (_Feb 22_) Complete the read_mars_graph() method so that you can create a graph. 
I've provided a Graph class for you; if you want to modify it, feel free. 

b) **(10 points)** (_Feb 23_) Complete the successors() method. For any state, it should
use the graph you construct above to identify all successor states.

b) **(10 points)** (_Feb 25_) In search_algorithms.py, complete the a_star function. 
You will need to implement a straight-line distance (SLD) heuristic by completing the sld() function. We will assume that the
goal is hardcoded as 1,1. 
This should compute SLD(p1) = sqrt((p1.x - p2.x)^2 + (p1.y - p2.y)^2)) where p2=(1,1)

SLD should be a separate function that is provided to a_star, NOT something encoded within the algorithm. 
You should be able to pass in a different heuristic function to a_star, such as h1, which always returns 0, without changing your code.
I've given you an SLD unit test; improve it to make it more useful in your debugging.

c) **(5 points)** (_Feb 26_) Run both A* and uniform cost search (i.e. using h1: h=0 for all states) 
on the MarsMap and count the number of states generated. Add this to your results.

**Question 4: Constraints.** 
**(10 points)** (_Feb 27_) 

For this question, you'll be using the [(OR-Tools)](https://developers.google.com/optimization) toolkit to solve a simple scheduling problem.

This provides our first introduction to *knowledge-based programming*. This is a style of program design
where the focus is on representing complex problem knowledge (constraints in this case) and then handing that
knowledge to an automated solver to find a solution. 

To begin, get ortools installed and run the included code in nurse_scheduler.py. Read through it and see 
if you can understand what it does. Then follow the instructions below and add a section to your answers explaining
what you learned.

Increase the number of nurses to 7, and add shift requests for them. Does this make the problem easier or harder? How can you tell?

Decrease the number of nurses to 3 and remove the extra shift requests. Does this make the problem easier or harder? How can you tell?

Return to five nurses, but increase the number of shifts to five, and update the shift requests. Does this make the problem 
easier or harder? How can you tell?

In this problem, we've separated the problem knowledge from the algorithmic knowledge. How does this help us
in engineering a solution?

**Question 5: Deep Blue vs AlphaZero** 
**(10 points)**
(_Feb 27_)

In the late 90s, Deep Blue shocked the world by becoming the first computer to beat a human grandmaster, Garry Kasparov. 
[This paper](https://www.sciencedirect.com/science/article/pii/S0004370201001291?ref=pdf_download&fr=RR-2&rr=851930c31a9617ea) 
describes how Deep Blue was constructed - it took advantage of specialized hardware, 
along with hand-crafted heuristics and many optimizations of the alpha-beta pruning technique we've learned about.

20 years later, the Google team has re-revolutionized game search with the development of AlphaZero, 
which is described [in this paper](https://arxiv.org/pdf/1712.01815.pdf).

AlphaZero uses a very different approach - specifically, a deep neural network is used to learn heuristic functions 
through self-play. (We'll look at reinforcement learning later in the semester). This allows the program to learn to 
play any game, as long as it knows the state space, a goal function, and the legal actions.

These articles are both pretty dense, and I don't expect you to grasp every nuance, but you should be able to read the 
introductions and get the gist of things.

In your written answers, please address the following questions: 

a) What were the engineering advances that led to Deep Blue's success? Which of them can be transferred to other problems, 
and which are specific to chess?

b) AlphaZero is compared to a number of modern game-playing programs, such as StockFish, which work similarly to Deep Blue. 
The paper shows that AlphaZero is able to defeat StockFish even when it is given only 1/100 of the computing time. 
Why is that? Please frame your answer in terms of search and the number of nodes evaluated.

