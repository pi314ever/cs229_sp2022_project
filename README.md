# README for CS229 Spring 2022 Project

Contributors: Daniel Huang, Ruth-Ann Armstrong, Radhika Kapoor

## Setup for team members

1. Git clone repo from github
2. Download dataset from google drive
3. Unpack the files into `../dataset` (One layer outside of the folder containing this repo)
4. Delete any remaining .zip files containing the dataset

If things are set up correctly, there the dataset should not be included in `git status`.

**IMPORTANT: DO NOT UPLOAD ANY PART OF THE DATASET TO THIS REPO, AS THIS IS A PUBLIC REPO! DOUBLE CHECK EVERY TIME!**

## TODO list

- [ ] Complete neural network class (Daniel)
  - [ ] fit()
  - [ ] forward_prop()
  - [ ] backward_prop()
  - [ ] predict()
- [ ] Write basic neural network test
- [ ] Process dataset
  - [ ] Create class for each book containing attributes:
    - Title (str)
    - ISBN (int64)
    - Level (int) (0:A, 1:B, etc...)
    - Words (list of separated words stripped of ending punctuation)
    - Other features TBD
  - [ ] Create word-to-index mapping of entire dataset (Must have all of the relevant words from all batches)
    - [ ] Save into a `.csv` file so it can be loaded more easily
