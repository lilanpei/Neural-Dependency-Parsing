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
