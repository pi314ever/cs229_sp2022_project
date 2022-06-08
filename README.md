# Classification of Children's with Various Supervised and Unsupervised Methods

Stanford University CS229 Spring 2022 Project

Contributors: Daniel Huang, Ruth-Ann Armstrong, Radhika Kapoor

<!-- ## Setup for team members

1. Git clone repo from github
2. Git clone the dataset repo next to this repo (should perform `cd ..` before executing the git clone command)

If things are set up correctly, the dataset should not be included in `git status`.

**IMPORTANT: DO NOT UPLOAD ANY PART OF THE DATASET TO THIS REPO, AS THIS IS A PUBLIC REPO! DOUBLE CHECK EVERY TIME!** -->

## Model summary and usage

For running the code as-is, it is highly recommended to import the environment from `environment.yml` using `conda env create -f environment.yml`.

### Neural network model (`neural_network.py`)

Two classes for neural networks are implemented: `n_layer_neural_network()` and `two_layer_neural_network()`.The n-layer neural network is comprised of a user-specified $n_{\text{hidden layers}} = n_{HL}$ fully connected hidden layers each of uniform size $n_{\text{hidden}} = n_H$ with user-specified activation functions. (`util.py` contains `ReLu` and `Sigmoid` activations with their corresponding derivative functions). More specific documentation can be found in the docstring of each method. An example of creating, training, and evaluating an n-layer neural network is seen below:

```python
from neural_network import n_layer_neural_network
import util

# Specify model sizes
n_features = 100
n_hidden = 10
n_layers = 5
n_classes = 3
nn = n_layer_neural_network(n_features, n_hidden, layers, n_levels,
                            [util.sigmoid] * n_layers, [util.dsigmoid] * n_layers)
# Gather data
train_data, train_labels = load_train_set()
dev_data, dev_labels = load_dev_set()
test_data, test_labels = load_test_set()
# Specify model training parameters
reg = 0
lr = 0.1
epochs = 50
batch_size = 10
# Fit model
cost_train, accuracy_train, cost_dev, accuracy_dev = nn.fit(train_data, train_labels, batch_size=batch_size, num_epochs=epochs, dev_data=dev_data, dev_labels=dev_labels,learning_rate=lr, reg=reg, print_epochs=True)
# Evaluate model
pred_test = nn.predict_one_hot(test_data) # predict() for raw output probabilities
# Confusion matrix on the test set
print((pred_test.T @ test_labels).astype(int))
```

The 2-layer neural network is simply the a n-layer neural network with only 1 hidden layer with a sigmoid activation function. It can be created similar to an n-layer neural network with fewer necessary parameters.

### Naive Bayes model (`naive_bayes.py`)

The Naive Bayes model is implemented in the class `naive_bayes_model()`. Since it derives from the general model class `util.classification_model`, the workflow is extremely similar to that shown above for the n-layer neural network with fewer necessary parameters. See docstring documentation for more specific information.

### K-means model (`cs229_kmeans.py`)

The K-means model was imported from `sklearn.cluster.KMeans` with no additional tweaks.

## TODO list (Complete as of 6/6/2022)

### Data analysis

- [x] Isolate and refine hyperparameter search
- [x] Create plots for k-means
- [x] Create plots for hyperparameter search
- [x] Clean up k-means

### `util.py`

- [x] Unify `util.load_dataset` API with more dataset filter options
  - [x] Group by books (much less data, but more descriptive)
  - [ ] <s>Appending other features into feature list
    - Total number of words in the book
    - Average length of sentences
    - Unique words
    - Sentence repetition?</s>
- [x] Encode the chunks of data using a NLP vectorizer?

### `neural_network.py`

- [x] Implement n-layer model
- [x] Develop code to auto-test multiple
- [x] Complete neural network class (Daniel)
  - [x] fit()
  - [x] forward_prop()
  - [x] backward_prop()
  - [x] predict()
- [x] Write basic neural network test

### `naive_bayes.py`

- [x] Complete naive bayes implementation in a class

### `construct_datafiles.py`

- [x] Process dataset
  - [x] Create class for each book containing attributes:
    - Title (str)
    - ISBN (int64)
    - Level (int) (0:A, 1:B, etc...)
    - Words (list of separated words stripped of ending punctuation)
    - Other features TBD
  - [x] Create word-to-index mapping of entire dataset (Must have all of the relevant words from all batches)
    - [x] Save into a `.csv` file so it can be loaded more easily

### Other

- [x] Develop k-means model
- [x] Import other language models?
