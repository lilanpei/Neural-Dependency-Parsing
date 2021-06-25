# Neural-Dependency-Parsing
In this homework, you’ll be implementing a neural-network based dependency parser with the goal of maximizing performance on the UAS (Unlabeled Attachment Score) metric.

A dependency parser analyzes the grammatical structure of a sentence, establishing relationships between head words, and dependent words which modify those heads.
There are several types of dependency parsers, including transition-based parsers, graph-based parsers, and feature-based parsers. Your implementation will be a transition-based parser, which incrementally builds up a parse one step at a time.
The parser maintains a state, which is represented as follows:

- A `stack` of words that are currently being processed.
- A `buffer` of words yet to be processed.
- A `list` of dependencies predicted by the parser.

Initially, the stack only contains `ROOT`, the dependencies list is empty, and the buffer contains the list of words of the sentence. At each step, the parser applies a transition to its state until its buffer is empty and the stack size is 1.
The following transitions can be applied:

- `SHIFT`: removes the first word from the buffer and pushes it onto the stack.
- `LEFT-ARC`: marks the second (second most recently added) item on the stack as a dependent of the first item and removes the second item from the stack, adding a first word → second word dependency to the dependency list.
- `RIGHT-ARC`: marks the first (most recently added) item on the stack as a dependent of the second item and removes the first item from the stack, adding a second word → first word dependency to the dependency list.

On each step, your parser will decide among the three transitions using a neural network classifier.
## 1. Preliminaries
### 1.1 Transitions
Provide the sequence of Attardi’s non-projective transitions for parsing the following sentence:

`The president scheduled a meeting yesterday that nobody attended.`
### 1.2 Features
*What is the difference in terms of features between neural network dependency parsers (e.g. Chen&Manning 2014, https://cs.stanford.edu/~danqi/papers/emnlp2014.pdf) and non-neural network dependency parsers (e.g. parsers with lots of features like Zhang&Nivre 2011, www.anthology.aclweb.org/P/P11/P11-2033.pdf), in particular in terms of sparsity?*
### 1.3 Ambiguity
*What is the ambiguity in parsing the following sentence?*<br/>
`There are statistics about poverty that no one is willing to accept`
### 1.4 Parse Tree
*Mention which errors that make the following an incorrect dependency tree:*

![20210410135823.png](data:image/png;base64,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)
### Exercise 1.
Implement the `__init__` and `step` methods in the `ParseState` class in `parser_state.py`. This implements the transition mechanics your parser will use.
  * Test a single parser step
  * Test parsing a sentence
### Exercise 2
We are now going to train a neural network to predict, given the state of the stack, buffer, and dependencies, which transition should be applied next.<br/>
First, the model extracts a feature vector representing the current state. We will be using the feature set presented in the  paper by  Chen and Manning (2014), "A Fast and Accurate Dependency Parser using Neural Networks", https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf.

The method `ParserState.extract_features()` to extract these features is  implemented in `parser_state.py`.
These features consist of a triple:
- a list of tokens (e.g., the last word in the stack, first word in the buffer, dependent of the second-to-last word in the stack if there is one, etc.).
- a list of POS tags for the same tokens
- a list of DEPRELs for the same tokens.
Each element is represented by an integer ids, and therefore it consists of:

$$[ [w_1,w_2,...,w_m], [p_1, p_2,...,p_m], [d_1, d_2,..., d_m] ]$$

where $m$ is the number of features and each $0 ≤ w_i < |V|$ is the index of a token in the vocabulary ($|V|$ is the vocabulary size) and similarly for $p_i$ and $d_i$.
Then our network looks up an embedding for each word and tags and concatenates them into a single input vector:
$$x = [E_{w_1},...,E_{w_m},Ep_{p_1},...,Ep_{p_m},Ed_{d_1},...,Ed_{d_m}] ∈ \mathbb{R}^{(d+d_p+d_d)m}$$
where $E ∈ \mathbb{R}^{|V|×d}$ is an embedding matrix with each row $E_w$ as the vector for a particular word $w$, and similarly $Ep$ and $Ed$ for tags, with dimesions respectively $d_p$ and $d_d$.<br/>
We then compute our prediction as:
$$h = ReLU(xW + b_1)$$
$$l = hU + b_2$$
$$\hat{y} = softmax(l)$$
where $h$ is referred to as the hidden layer, $l$ is referred to as the logits, $\hat{y}$ is referred to as the predictions, and $ReLU(z) = max(z, 0)$. We will train the model to minimize cross-entropy loss:
$$J(θ) = CE(y,\hat{y}) = \sum_{i=1}^a{−y_i log \hat{y}_i}$$
where $a$ is the number of possible parser actions.
To compute the loss for the training set, we average this $J(θ)$ across all training examples.
We will use UAS score as our evaluation metric. UAS refers to Unlabeled Attachment Score, which is computed as the ratio between number of correctly predicted dependencies and the number of total dependencies irrespective of the relations.

In `model.py` you will find skeleton code to implement this simple neural network using Keras. Complete the `__init__` methods to implement the model.

Then complete the train for epoch and train functions. Finally execute python `run.py` to train your model and compute predictions on test data from Penn Treebank (annotated with Universal Dependencies), available in files `data/traing.gold.conll`, `data/dev.gold.conll` and `data/test.gold.conll`.

##Note:##

  * Load the training data
  * Create the parser
  * Convert to numeric vectors
  * Build the training set
  * Build the datasets
  * Load the embeddings
  * Prepare embedding matrix
  * Create the model
  * Train the model
  * Test the model
### Exercise 3 (optional)
Modify the Parser.parse() method to print the parsed sentences in CoNLL-U format.
### Exercise 4
Let's explore the ability of the parser to handle common mistakes.

Consider the following kind of errors:

1. Prepositional Phrase Attachement Error. This occurs when a prepositional phrase is attached to wrong head word.
2.Verb Phrase Attachement Error. This occurrs when a verb phrase is attached to the wrong head word.
3. Modifier Attachemnt Error. This occurrs when a modifier (adjective or adverb) is attached to the wrong head word.
4. Coordination Attachment Error: This occurrs when a conjuct is attached to the wrong head wor.

Check whether the parser does any of these mistakes on these sentences and classify them according to the 4 types above:

* Moscow sent troops to Afghaninstan
* I disembarked and was heading to a wedding fearing for my death
* It makes me want to rush out and rescue people from dilemmas of their own making.
* Brian has been one of the most crucial elements to the success of Mozilla.
### Submission instructions
1. Make a zip with your code and notebook (without the data folder), i.e.
2. zip <YourName>.zip .ipynb .py
3 .Submit though Moodle on https://elearning.di.unipi.it/mod/assign/view.php
