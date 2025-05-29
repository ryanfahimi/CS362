### CS 362 - Assignment 4. Text Clustering, Knowledge Representation and Decision Trees

#### Due Mon 3/31, 11:59pm

For the programming  portions of the assignment, please provide a file called submission.py 
that runs your decision tree code.

For the written portions, please prepare a document called assignment4.pdf and put your answers in there.
And don't forget to put your name on your assignment!

**On time management!** This assignment has many small components; the key to success will be starting early.
I've added suggested milestones _in italics_. 


### Problem 1. Inference.

*(milestone: 3/18)*

This is a written question. You can either write it out (legibly, please), 

NASA has reached out to us for help in designing the logic for the Mars rover. 
They would like it to be able to conduct experiments, and only radio back to Earth for help in case of an actual emergency. 
We've decided to build a rule-based system to implement this.

We have the following predicates (with abbreviations) to use:

- a: battery failed
- b: solar panels failed
- c: base station failed
- d: backup battery failed
- e: send emergency signal
- q: backup switch fails
- w: cannot sync base station with rover
- x: battery sensor not responding
- y: battery will not power motor
- z: no current from panels

NASA has given us the following rules to use:

- If the battery sensor is not responding and the battery will not power the motor, then the battery failed.
- If the battery sensor is not responding, we are not able to sync the base station with the rover.
- If the battery sensor is not responding and there is no current from the panels, then the solar panels have failed.
- If the battery will not power the motor, and we cannot sync the base station with the rover, then the base station has failed.
- If the backup switch fails, and there's no current from the panels, and the battery fails, then the backup battery fails.
- If the backup switch fails, then the battery will not power the motor.
- If battery failed and solar panels failed and base station failed and backup battery failed, then send the emergency signal.
- If we cannot sync the base station with the rover, we are unable to get current from the panels.

Our rover has observed the following facts:
- The battery sensor is not responding.
- The backup switch has failed.

**(5 points)**  Show what the initial KB looks like, using the abbreviations for each rule and fact.

**(5 points)**  Use forward chaining to show that we should send an emergency signal. 
For each step, show the rule and fact(s) being matched, and the resulting fact. for example: q, q->y, y.

**(5 points)**  Our NASA contact is skeptical; he's heard that AI can make mistakes, and does not believe that there is 
truly an emergency. We need to prove that this was the right decision. Use backward chaining to show that signaling the 
emergency was the correct decision. For each step, show the stack and the items to be proved.

Our NASA contact is still skeptical - just because we were right this time doesn't mean it could never make mistakes, 
he argues! We need to prove that our rules will always generate the right result. We'll use resolution to verify the 
correctness of our rules.

**(5 points)** a) Convert the KB to Conjunctive Normal Form.

**(5 points)**  b) Add the term !e (do not signal) to the KB, and use resolution to derive a contradiction. 
For each step, show the two terms being resolved and the new sentence that results. 
for example: (!w v z) ^ (!x v w) -> (z v !x)

### Problem 2. using sklearn to do clustering.

In this section, you'll see how to use sklearn to do K-means clustering.  
There's no code to write here, but you will need to run the algorithm with 
different parameters and summarize your results.

The code is derived from  [this tutorial](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html), which describes how to do 
k-means using sklearn.

The modified code is in textcluster.py.

1. **(5 points)** *(3/20)* As we saw in the last assignment, a challenge with 
clustering text is selecting and weighting features. In this question, 
you'll examine how this can affect performance.

This code runs k-means on the 20 newsgroups dataset using
three different vectorization techniques: vanilla bag-of-words, TFIDF, 
and latent semantic indexing.

Run the code and generate a table that shows the homogeneity and completeness for vanilla clustering, TFIDF, and LSA.

2. **(10 points)** *(3/21)* As we saw in the previous assignment, homogeneity and 
completeness are helpful for understanding performance, but like accuracy 
and precision, there's a tension between them. This code includes three 
additional metrics that try to capture the "goodness" of a cluster:
   1. [V-Measure](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html)
   2. [Adjusted Rand index](https://en.wikipedia.org/wiki/Rand_index)
   3. [Silhouette score](https://en.wikipedia.org/wiki/Silhouette_(clustering))

Add these scores to your table. Also, add a concise definition of each metric and what it is trying to measure. 

3. **(10 points)** *(3/22)* Currently we're using 4 groups out of the [20_newsgroup dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html). Based on Assignment 3, 
we hypothesize that clustering works better when the documents within each class are more similar, and the 
documents in different classes are more distinct. Let's test that out with real data. Run an experiment to 
compare the performance of each technique (vanilla, TFIDF, LSA) on:

   1. These four groups (baseline)
   2. 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware' (four similar groups)
   3. 'rec.autos', 'alt.atheism', 'sci.space', 'talk.politics.mideast' (four different groups)
   
Include a table showing the five metrics for each experiment. Was our hypothesis correct?

### Problem 3: Decision tree 

This is the largest part of the assignment - we'll implement the basic decision tree algorithm. 
I've provided three datasets for you, plus some unit tests.
The first two datasets, restaurant and tennis, are toy datasets to use in building our code.
The third dataset, NHPA doctor visits, is the one we'll use to evaluate our tree. You can find more
information about this dataset [here](https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+(npha))

You will need to know all of the metadata (feature names and possible values)
for each dataset. You can find code to set this up for tennis and doctor visits 
in test_decision_tree.py (I'll let you do restaurant yourself.)

**(5 points)**  *(3/23)* I've implemented entropy for you. 
 Use that to implement gain. I have provided a unit test for you. 

**(10 points)** *(3/24)* Use gain to implement select_attribute. I've started a unit test for you; 
feel free to extend it.

**(10 points)** *(3/24)* Use select_attribute to implement make_tree. Please add your own unit test here.

**(10 points)** *(3/26)* Now you are ready to implement classify. Add a unit test for this as well.

**(5 points)** *(3/27)* Once your tree is working, you are ready to evaluate its performance.
For this, we'll use the [NPHA doctor visits dataset](https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+(npha). 
This predicts the number of doctors a person has seen in the past year, based on self-reported survey data. The link 
describes the data in more detail.

Measure the accuracy of your tree on the NPHA data 
using five-fold cross-validation. You are encouraged to 
repurpose or refactor your code from assignment 3 here.

### Problem 4: Knowledge Graphs.

*(3/31)*

Wikipedia sits on top of a knowledge graph (called WikiData). In this question, you'll get a chance to explore this 
knowledge graph.

To begin, open the page for [the Perseverance rover](https://en.wikipedia.org/wiki/Perseverance_(rover)). Go to 'tools' (on the right) and then WikiData to bring 
up the data page.

Every WikiData object has a unique id - this one is Q87749354. 

Scroll down to see some of the statements associated with this object. 
Note that each of the relations is also an object. For example, click on 'video' to learn what this link means.

There are a number of really interesting applications built on top of WikiData that let you
take advantage of the underlying knowledge graph. Please try each of these and add the answers
to the included questions in your PDF.

**(5 points)**  [LinkedPeople](https://linkedpeople.net/) LinkedPeople uses WikiData to 
construct family trees for real and fictional people. Let's look at Abraham Lincoln.
   a. How many children did Lincoln have? What were their names?
   b. Who was the father of the wife of Lincoln's son Robert?

**(5 points)** [Entitree](https://www.entitree.com) provides a tree-based representation for 
Wikidata entries, and allows you to specify what relationship you want to 
consider. Let's start with Lincoln again.
   
   a. What event was he the target of? Who was the perpetrator? Use the pulldown on the right to 
      determine this.
   
   b. Where is the perpetrator buried?
      
   c. Who are three other people buried there?

**(5 points)** [OpenArtBrowser[(https://openartbrowser.org/en/) uses WikiData to provide connections
and context between artworks, artists, and periods. Let's start with Leonardo Da Vinci.
   
   a. What movements is he a part of?
   
   b. Find his work "Vitruvian Man." Where is it located? Who are two other artists whose works 
   are displayed there"
   
   c. One of the motifs used in "Vitruvian Man" is the circle. What are two other
   artworks that use the circle as a motif? 


