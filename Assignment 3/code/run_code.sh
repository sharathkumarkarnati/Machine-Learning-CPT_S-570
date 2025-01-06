#!/bin/bash


echo "Running naive_bayes.py..."
/opt/anaconda3/bin/python "/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 3/code/naive_bayes.py" > output.txt 2>&1


if [ $? -eq 0 ]; then
    echo "Running nb_sklearn.py..."
    /opt/anaconda3/bin/python "/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 3/code/nb_sklearn.py" >> output.txt 2>&1
else
    echo "naive_bayes.py failed. Skipping nb_sklearn.py."
fi

echo "Both scripts have been executed. Output has been saved to output.txt."
