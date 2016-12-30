# -*- coding: utf-8 -*-
"""Some helpers functions"""

from itertools import groupby
import csv
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression


def calculate_rmse(predicted_labels, true_labels):
    """Compute the loss (RMSE) of the prediction of nonzero elements."""
    return np.sqrt(
        np.mean((predicted_labels - true_labels) ** 2)
    )


def read_txt(path):
    """Read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def preprocess_data(data):
    """Preprocess the text data, convert to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    data = [deal_line(line) for line in data]

    rows = set([line[0] for line in data])
    cols = set([line[1] for line in data])

    ratings = sp.lil_matrix((max(rows), max(cols)))

    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating

    print(
        'Matrix sparsity: %.2f%%'
        % (100 * ratings.nnz / (ratings.shape[0] * ratings.shape[1]))
    )

    return ratings


def load_data(path_dataset):
    """Load data in text format, one rating per line."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def split_data(ratings, p_test=0.1, seed=988):
    """Split the data in training and test sets."""
    np.random.seed(seed)

    n_users = ratings.shape[1]

    test = sp.lil_matrix(ratings.shape)
    train = ratings.copy()

    for user in range(n_users):
        nb_ratings = len(ratings[:, user].nonzero()[0])
        test_ratings = np.random.choice(
            ratings[:, user].nonzero()[0],
            size=int(p_test * nb_ratings),
            replace=False
        )
        train[test_ratings, user] = 0
        test[test_ratings, user] = ratings[test_ratings, user]

    print('Number of ratings : %d' % ratings.nnz)
    print('Number of train ratings : %d' % train.nnz)
    print('Number of test ratings : %d' % test.nnz)

    return train, test


def group_by(data, index):
    """Group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


def build_index_groups(train):
    """Build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices


def get_movie_user(index):
    """Returns a movie, user tuple."""
    movie, user = index.split('_')
    movie = int(movie.replace('r', '')) - 1
    user = int(user.replace('c', '')) - 1

    return movie, user


def get_linear_blend_clf(X, y):
    """Return a linear classifier for the given models."""
    clf = LinearRegression()
    clf.fit(X.T, y)

    print("Train error: %f" % np.sqrt(np.mean((clf.predict(X.T) - y) ** 2)))

    return clf


def generate_submission(predicted_labels):
    """Generate a CSV submission from a predicted labels matrix."""
    # Get indexes to predict
    with open('data/sampleSubmission.csv', 'r') as submission_sample:
        submission_indexes = []

        for idx, row in enumerate(submission_sample.read().splitlines()):
            if idx != 0:
                pos, _ = row.split(',')
                submission_indexes.append(pos)

    # Write predictions
    with open('data/submission.csv', 'w', newline='\n') as submission_output:
        writer = csv.writer(submission_output)
        writer.writerow(['Id', 'Prediction'])

        for index in submission_indexes:
            label = predicted_labels[get_movie_user(index)]
            writer.writerow([index, label])
