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

The notebook 'character-recognition-using-cnn' contains all solution to the tasks. 
    
The data for all three tasks are in 'data' folder. The get_labels.py script was used to create a dataframe of images and their respective classes. The script need not be run, because the final csv file is savedas 'labels.csv' file. The notebook has all the objective listed in it. Necessary explanations are given inside the notebook itself. 


## Author

[Pratik Kumar](https://pr2tik1.github.io)

## Credits

- [MIDAS Labs @IIIT Delhi](http://midas.iiitd.edu.in)

## References

- [PyTorch Tutorials](https://pytorch.org/tutorials/) and [Documentation](https://pytorch.org/docs/stable/index.html) 
- [Udacity Nanodegree Notes](https://github.com/udacity/deep-learning-v2-pytorch)
- [AlexNet Paper](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) 