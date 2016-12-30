## Machine Learning - Project 2

In this repository, you can find my work for the Project 2 of the [Machine Learning](http://mlo.epfl.ch/page-136795.html) course at [EPFL](http://epfl.ch).

This project consists in a [Kaggle competition](https://inclass.kaggle.com/c/epfml-rec-sys) concerning a recommender system based on user/item ratings (more than 1 million records with a sparsity of ~ 11%). My team ended at the 9th place (over 33 teams).

For more information about the implementation, see the [PDF report](https://github.com/vfaramond/ML-Project2/blob/master/Report.pdf) and the commented code.

This file describes how to run the provided recommender code on a machine with Python 3 installed with Anaconda.

The script to run is `run.py`. This file uses and imports functions from `helpers.py` which uses functions from `ALS.py`

The data (`data_train.csv` and `sampleSubmission.csv`) should be placed inside a folder named `data` at the same level than the scripts.

To increase computing speed, the algorithm uses dumps already generated previously. Those dumps should be placed inside a
folder named `dumps` at the same level than the scripts.

The algorithm takes around 5 minutes to finish using the dumps.
