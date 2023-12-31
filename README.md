# Project: NLP - Sentiment Analysis

This repository was created for my studies at IU International University for the course Project: NLP.

&nbsp;



### The Repository:
The preprocessing steps are located in the file **preprocess.ipynb**.
The sentiment analysis with supervised machine learning models (BernoulliNB and SVM) are located in file **supervised_NB_SVM.ipynb**.
The Deep Learning model with word embeddings is located in file **deep_learning.ipynb**. 
A small function that allows classification of user customed reviews is located in file **predict.py**. 

In the folders, there are some plots of metrics, the resulting models saved (except the SVM model, out of size restrictions by github) and the logs of tensorboard. 

&nbsp;
&nbsp;

### Prepare your environment:

For stable usage of the application, **python version 3.10** is recommended. Install python from the [official website](https://www.python.org/). Check your python version with entering your command promt and execute the following command:

```python
python --version 
```

It is recommended to use a customized environment to ensure full functionality, e.g. with the distribution anaconda, which can be downloaded [here](https://www.anaconda.com/products/distribution).

Install the required packages with the following command in your command line interface (For more information about pip, please check the [pip documentation](https://pip.pypa.io/en/latest/user_guide/)):

```python
pip install -r requirements.txt 
```

&nbsp;

&nbsp;


### Run sentiment analysis on own data:

First download the code by either downloading the .zip-File or clone it via the command promt. For more information about the later please check the [github docs](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

To predict custom sentiment on movie reviews, either open the project with a python-IDE and run the file *predict.py* or enter the following in the command prompt:

```python
python predict.py 
```

Of course it is also possible to run it in the Jupyter Notebook instances. Look there for the end of the files.
&nbsp;
