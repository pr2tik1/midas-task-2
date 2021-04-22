# Hand Written Characters Image Classification (MIDAS Internship Task 2021)

This repository contains solution to second task for summer internship/RA 2021. The task is to develop, model and train Convolutional Neural Network from scratch and perform comparitive study with transfer learning on given datasets. The dataset is images of handwritten characters of english language. The dataset contains handwritten characters that are small alphabets, capital alphabets and 0-9 digits. 

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
## Description

1. *About Notebooks* : Each part of the second task is solved in each of the following notebooks:

    The data for all three tasks are in 'data' folder. The get_labels.py script was used to create a dataframe of images and their respective classes. The script need not be run, because the final csv file is savedas 'labels.csv' file. The notebook has all the objective listed in it. Necessary explanations are given inside the notebook itself. 
    
    - Part 1: 
        - [character-recognition.ipynb](https://github.com/pr2tik1/midas-task-2/blob/main/character-recognition.ipynb)
    - Part 2: 
        - [digits.ipynb](https://github.com/pr2tik1/midas-task-2/blob/main/digits.ipynb)
    - Part 3:
        - [new-digits-recognition.ipynb](https://github.com/pr2tik1/midas-task-2/blob/main/digits-recognition.ipynb)

2. *About Data*: Data for all three tasks are in 'data' folder. The get_labels.py script was used to create a dataframe of images and their respective classes. The script need not be run, because the final csv file is savedas 'labels.csv' file. The notebook has all the objective listed in it. Necessary explanations are given inside the notebook itself. 

## Future Work: 

    - [ ] Use more models
    - [ ] Develop web application using Dash/Flask.

## Author

[Pratik Kumar](https://pr2tik1.github.io)

## Credits

- [MIDAS Labs @IIIT Delhi](http://midas.iiitd.edu.in) 

## References

- [PyTorch Tutorials](https://pytorch.org/tutorials/) and [Documentation](https://pytorch.org/docs/stable/index.html) 
- [Udacity Nanodegree Notes](https://github.com/udacity/deep-learning-v2-pytorch)
- [AlexNet Paper](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) 

(*Please note the notebook files were trained on GPUs provided by Kaggle. Please take note of directories while reproducing. Thank you!*)
