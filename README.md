# README for CS229 Spring 2022 Project

Contributors: Daniel Huang, Ruth-Ann Armstrong, Radhika Kapoor

## Setup for team members

1. Git clone repo from github
2. Git clone the dataset repo next to this repo (should perform `cd ..` before executing the git clone command)

If things are set up correctly, the dataset should not be included in `git status`.

**IMPORTANT: DO NOT UPLOAD ANY PART OF THE DATASET TO THIS REPO, AS THIS IS A PUBLIC REPO! DOUBLE CHECK EVERY TIME!**

## TODO list

### Data analysis

- [ ] Isolate and refine hyperparameter search
- [ ] Create plots for k-means
- [ ] Create plots for hyperparameter search
- [ ] Clean up k-means

### `util.py`

- [x] Unify `util.load_dataset` API with more dataset filter options
  - [x] Group by books (much less data, but more descriptive)
  - [ ] Appending other features into feature list
    - Total number of words in the book
    - Average length of sentences
    - Unique words
    - Sentence repetition?
- [x] Encode the chunks of data using a NLP vectorizer?

### `neural_network.py`

- [x] Develop code to auto-test multiple
- [x] Complete neural network class (Daniel)
  - [x] fit()
  - [x] forward_prop()
  - [x] backward_prop()
  - [x] predict()
- [x] Write basic neural network test
- [ ] <s>Implement n-layer model</s>

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
