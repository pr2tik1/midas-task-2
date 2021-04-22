# Hand Written Characters Image Classification

The task is to develop, model and train Convolutional Neural Network from scratch and perform comparitive study with transfer learning on given datasets. The dataset is images of handwritten characters of english language. The dataset contains handwritten characters that are small alphabets, capital alphabets and 0-9 digits. 

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


- About *Notebooks* and *Data* : 
    
    - Part 1: 
        - [character-recognition.ipynb](https://github.com/pr2tik1/midas-task-2/blob/main/character-recognition.ipynb) 
        - Data: 62 Classes with around 2700 Images within 'data/train/' directory. Additionally a csv file containing all the labels named 'labels.csv'.
        
    - Part 2:
        - [digits-recognition.ipynb](https://github.com/pr2tik1/midas-task-2/blob/main/digits-recognition.ipynb)
        - Data: Needs to be labeled
        - TODO


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
