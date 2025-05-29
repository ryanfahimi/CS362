### Assignment 7: Neural models. 

#### Due May 16, 11:59pm. 

*note: No late assignments are allowed for this assignment.*

*note also - there is no starter code for this assignment. Just this README!* 

In this assignment, you'll work with several tutorials designed to expose you to different
deep learning approaches, packages, and resources. This assignment is less about
coding and more about working with existing systems. 

Please add all of your answers to a file named assignment7.pdf and add it to this repo.

##### (25 points) Question 1. Linear Algebra. 

In the asynchronous lecture from April 25,
there were a set of questions on basic linear algebra operations. Please include the answers to these questions in your repo.

#### (25 points) Question 2. Convolutional Neural Networks. 

TensorFlow is a well-known and widely used deep learning package. Google has provided a large set of tutorials and resources to help people learn to use it.
Most of them work in CoLab, so you should not need to install anything locally if you don't want to. 

These tutorials are great, but it's easy to wind up just clicking on things and not really absorbing the material. In order to help you think about 
some of the interesting issues being presented, I've developed a set of questions for you to answer. Please add a document with the answers to your repo. 

NOTE: Please answer these questions in your own words - do not just cut and paste from the tutorial. Cut-and-paste answers (or ChatGPT-generated answers) will receive zero credit.

[This tutorial](https://www.tensorflow.org/tutorials/images/cnn) shows how to build a 
simple CNN to classify images from the CIFAR10 dataset.

- How large are the CIFAR10 images? What does the 10 refer to?

- What is a MaxPooling2D layer? What's it do?

- What's Adam? 

- What's the softmax function do? 

- What is CategoricalCrossEntropy? What do we use it for?

- What does the Flatten layer do?

- What does the Dense layer do? 

- Why does the height and width get smaller for each convolutional layer?

- Try changing the number of epochs to 20. What happens to the error for the training set?
What about for the validation set? Why is that?


#### (25 points) Question 3. Deep Q-Learning.

[This tutorial](https://huggingface.co/learn/deep-rl-course/unit3/introduction?fw=pt) from HuggingFace shows how to use the HuggingFace platform and the RLZoo framework to implement 
Deep Q-learning to solve Atari 2600 games.

  (note: it takes about 9 hours to train this network if you train for 10M steps. I recommend reducing this significantly to 1M, or even 100,000, 
   at least the first few times you run.)

1. Work through it. (you might want to look back to Chapter 2 in the tutorial if you've forgotten Q-learning.)

2. Solve another game that's not Space Invaders. I recommend a simpler game like Asteroids, Breakout, or Pong. More open-ended games 
like Adventure and Pitfall have very large policy spaces and are hard to learn with the resources we have available.

3. Publish your model to the Hub and email me a link to it.

Note: These tutorials can be a little finicky, and sometimes things
don't always work as advertised. Tinkering around to get it working is part of the challenge. 
Don't leave it until the last minute! 

You will need to figure out how to save a hyperparameter config file, and also how to
connect to the HuggingFace Hub.

#### (25 points) Question 4. BERT tutorial

[This Kaggle tutorial](https://www.kaggle.com/code/harshjain123/bert-for-everyone-tutorial-implementation/notebook) 
demonstrates how to do sentiment analysis with BERT, an extremely
successful Transformer architecture for natural language processing.

You'll need to do a little editing to get it to work:
- Be sure to set the runtime to TPU
- fix the path in step 1 to: 
```
df = pd.read_csv("../kaggle/input/spamdatatest/spamdata_v2.csv")
```
- In part 7, comment out from transformers import AdamW and change the 
following line to:
```
optimizer = torch.optim.AdamW(model.parameters(),lr = 1e-5) 
```

- Then in the following block change the class_weights assignment to:
```
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
```

Run the tutorial and answer the following questions:

- What does this tutorial do?
- BERT is a contextual model. What does this mean? How is it different
from the bag-of-words approach?
- What are the three embeddings BERT uses?
- What is Masked Language Modeling?
- What is Next Sentence Prediction?
= What is the final performance of BERT on this dataset? 
- Increase to 20 epochs. How does performance change? 




- [This tutorial](https://huggingface.co/docs/transformers/tasks/question_answering) shows how to use HuggingFace transformers to do question answering, a core NLP task.
1. Work through it.
2. Write a wrapper for your question answerer to allow users to chat with it. Include this in your repo. (you can include it in the colab notebook if you want.)
3. Publish model to the Hub and email me a link to it.
4. Include the colab code in your repo.
