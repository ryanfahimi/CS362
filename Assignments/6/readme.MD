
### Assignment 6: Markov Decision Processes and Reinforcement Learning

#### Due Monday April 28, 11:59pm

##### 100 points.

Note: please make sure that your name is someplace in your assignment! 
This is especially true if it's not obvious from your GitHub handle.

To turn in: please submit everything to your github repo. 
For the written portion of the assignment, please add a document to your repository, as PDF, 
containing the answers to these questions.

Please also include a submission.py that demonstrates value iteration, policy iteration,
and Q-learning.

##### Question 1: Markov Decision Processes. 

For this problem, you'll implement the value iteration and policy iteration algorithms. 

I've provided some setup code for you that loads in the states and probabilities, and computes their expected utility.
I've also provided the setup for two problems:

1. The map shown in R&N, and in the slides. This is in rnGraph. This is a good problem for getting 
started and testing, but not very exciting.

2. In this problem, we return to the Mars lander. Good news: we've gotten the sensors working,
so we know where we are now! Also, we've gotten the rockets working, so we can try to navigate 
to the landing site. We need to land quickly; there are sandstorms coming!

Of course, there are challenges. There are two sandstorms here already; if the lander flies 
through them it will be damaged. Even worse, the rockets are not *quite* working. With p=0.8, 
they go in the intended direction, but with p=0.2, they go in the opposite direction! 

More specifically:
For every state (x,y) there are four possible actions: left, right, up, down.
With p=0.8, the lander moves in the intended direction. with p=0.2, the lander moves in the
opposite direction. If a movement would cause the lander to move off the map, it remains in place.

The lander starts in 1,1, and needs to reach the landing site in 5,5. There are sandstorms in 2,3
and 5,2.

If the lander reaches the landing site, it receives a reward of 1 and lands. Hooray!
If the lander reaches either sandstorm, it receives a reward of -1 and is forced to the ground. 
Bummer. We'll have to go get it after the storm is over.

**Part 1. (1 point)**  Load in rnGraph().
 
**Part 2. (9 points)**. Complete computePolicy(). Given utilities set in self.utilities, compute the 
resulting policy and return it as a dictionary mapping states to actions.

I've given you a unit test for this. Test it with rnGraph.

**Part 3. (10 points)** Complete the lander.actions file with the transitions for all the remaining states.

**Part 3. (10 points)** Implement the value iteration algorithm using r=-0.04 and gamma=0.8.

**Part 4. (10 points)** How can you test the resulting policy? We can use Monte Carlo simulation.
Write a function that takes the generated policy as input, along with a starting state and a number 
of iterations, and returns a list of final states.

**Part 5. (10 points)** Implement the policy iteration algorithm r=-0.04 and gamma=0.8. You can use
Monte Carlo to test this as well.

**Part 6. (10 points)** What if the time pressure is greater, and we really need to land? We
can model that by changing the reward. Set the reward to -0.1, -0.2, and -0.5 and run policy 
iteration. Add a paragraph to assignment6.pdf explaining how the policy changes for the lander. 

##### Question 2. Reinforcement Learning in Approach.

In this task, we'll return to Approach and use Q-learning to learn the optimal policy 
through repeated play.

Recall the dice game "Approach". It works like this: 

There are two players, and they choose
a target number (n). Player 1 repeatedly rolls a six-sided die and adds up their total. Whenever they want, they can "hold." Then player 2 must roll;' 
their goal is to beat player 1's score without going past n.

The problem we want to solve is this:

For player 1, for a given n and a particular total so far (called s) 
should they 'hold' (action 1) or 'roll' (action 2).

(note that player 2's strategy in this game is pretty boring - they just roll 
until they either beat player 1's score or exceed n.)

What we wind up with is a *policy*; that is, a map that tells us what to do for 
any state.

Earlier, we used Monte Carlo simulation to determine the best value to hold. 
This is fine when there's a small number of potential policies to consider, but 
it's not very scalable.

Let's now solve the problem with Q-learning.

Q-learning generates a table that stores Q(s,a). I've started a function called 
approach for you. It takes in a limit n and computes the Q-table for [0,n] with 
the actions (hold, roll). You will need to complete this.

For example, for n=10, you might get:

<pre>
sum:   hold     roll   [action]
0: 0.000000 0.493921 [roll]
1: 0.000000 0.477271 [roll]
2: 0.000000 0.479286 [roll]
3: 0.000000 0.505797 [roll]
4: 0.000000 0.580803 [roll]
5: 0.051270 0.498027 [roll]
6: 0.170403 0.427388 [roll]
7: 0.297536 0.365115 [roll]
8: 0.478196 0.285432 [hold]
9: 0.711051 0.164235 [hold]
10: 1.000000 0.000000 [hold]
</pre>

Your Q-values will vary, but the resulting policy should be the same.

We will do this using epsilon-soft on-policy control to manage exploration. We will select 
the optimal action with probability (1-epsilon) and explore with probability epsilon. 
(use epsilon=0.1 for this.)

**(20 points)** Implement the Q-learning algorithm in Approach.

Start by initializing the Q-table to small random values. 
Our agent should start with a random s and a and then either take the best 
action (prob. 1-epsilon), or the other action (prob. epsilon). At the end of 
the game, it receives reward 1 (if it wins) or 0 (if it loses). 
It then uses this to update the Q value for each state-action pair chosen. 
Run for 1000000 iterations and print out the Q-table, and optimal action for each state.


### Question 3.  Reinforcement Learning with Human Feedback

**(20 points)**

Reinforcement Learning has proved to be an effective way to fine-tune the question-answering 
abilities of large language models. Once the models are initially trained on a corpus, 
humans are used to then generate feedback that helps the LLM learn to distinguish between 
answers.

[This article from HuggingFace explains more about it](https://huggingface.co/blog/rlhf). It's great, but still
is pretty dense, with a lot of terminology.

Use your favorite LLM-based tool (I recommend [NotebookLM](https://notebooklm.google.com/) for this, but you can use
whatever you want) to create a summary of this article that explains the main ideas and helps you to
understand them. Include the results in your repo.

I'm looking for thoroughness, creativity, and effective use of LLMs as a tool for helping to learn complex ideas.
"Hey ChatGPT summarize this article" is not the idea. Feel free to generate links, diagrams, lists,
or whatever learning tools you think are most effective.
Rubric:

completeness: 5 points

effective use of LLMs: 10 points

creativity: 5 points



