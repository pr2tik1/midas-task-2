# MIDAS Labs Summer Internship Task 

This repository contains solution to second task for summer internship/RA 2021. The task is to develop, model and train a Convolutional Neural Network on given datasets. The dataset is images of handwritten characters of english language. The characters are small alphabets, capital alphabets and 0-9 digits. 

## Dependencies
    
    - PyTorch
    - Scikit-Learn
    - Pandas
    - Matplotlib
    - Numpy
    - OpenCV

## Installation
To install libraries simply execute following code,

```
pip3 install -r requirements.txt
```

## Usage
Activate the environment through: 

```
source midastaskenv/bin/activate
```

The 'nbs' folder contains all of the jupyter notebooks. These jupyter notebooks are my solution to the tasks. They are:
    1. Character Recognition: Part 1
    2.
    3.
    
The data for all three tasks are in 'data' folder. The get_labels.py script was used to create a dataframe of images and their respective classes. The script need not be run, because the final csv file is savedas 'Labels.csv' file. Each notebook has its own objective. Necessary explanations are given inside the notebook itself. 

## Author

[Pratik Kumar](https://pr2tik1.github.io)

## Credits

- [MIDAS Labs @IIIT Delhi](http://midas.iiitd.edu.in)

## References

- [PyTorch Tutorials](https://pytorch.org/tutorials/) and [Documentation](https://pytorch.org/docs/stable/index.html) 
- [Udacity Nanodegree Notes](https://github.com/udacity/deep-learning-v2-pytorch)
