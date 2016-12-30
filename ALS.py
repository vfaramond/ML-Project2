import os
import numpy as np
import helpers


def init_MF(train, n_features):
    """Initialize the parameters for matrix factorization."""
    n_items, n_users = train.shape

    user_features = 2.5 * np.random.rand(n_features, n_users)
    item_features = 2.5 * np.random.rand(n_items, n_features)

    return user_features, item_features


def update_user_feature(train, item_features, lambda_user, nnz_items_per_user, nz_user_itemindices):
    """Update user feature matrix."""
    n_users = train.shape[1]
    n_features = item_features.shape[1]

    user_feature = np.zeros((n_features, n_users))

    for user in range(n_users):
        nnz_items = nnz_items_per_user[user]
        nz_itemindices = nz_user_itemindices[user]
        nz_itemfeatures = item_features[nz_itemindices, :]
        A = (nz_itemfeatures.T).dot(nz_itemfeatures) + lambda_user * nnz_items * np.eye(n_features)
        train_user = train[nz_itemindices, user].toarray()
        b = (nz_itemfeatures.T).dot(train_user)[:, 0]
        user_feature[:, user] = np.linalg.solve(A, b)

    return user_feature


def update_item_feature(train, user_features, lambda_item, nnz_users_per_item, nz_item_userindices):
    """Update item feature matrix."""
    n_items = train.shape[0]
    n_features = user_features.shape[0]

    item_feature = np.zeros((n_items, n_features))

    for item in range(n_items):
        nnz_users = nnz_users_per_item[item]
        nz_userindices = nz_item_userindices[item]
        nz_userfeatures = user_features[:, nz_userindices]
        A = (nz_userfeatures).dot(nz_userfeatures.T) + lambda_item * nnz_users * np.eye(n_features)
        train_item = train[item, nz_userindices].T.toarray()
        b = (nz_userfeatures).dot(train_item)[:, 0]
        item_feature[item, :] = np.linalg.solve(A, b)

    return item_feature


def ALS(train, test, n_features, lambda_user, lambda_item, verbose=1):
    """Alternating Least Squares (ALS) algorithm."""
    print(
        '\nStarting ALS with n_features = %d, lambda_user = %f, lambda_item = %f'
        % (n_features, lambda_user, lambda_item)
    )

    n_epochs = 50

    user_features_file_path = 'dump/user_features_%s_%s_%s_%s.npy' \
        % (n_epochs, n_features, lambda_user, lambda_item)

    item_features_file_path = 'dump/item_features_%s_%s_%s_%s.npy' \
        % (n_epochs, n_features, lambda_user, lambda_item)

    if (os.path.exists(user_features_file_path) and
            os.path.exists(item_features_file_path)):
        user_features = np.load(user_features_file_path)
        item_features = np.load(item_features_file_path)

        train_rmse = helpers.calculate_rmse(
            np.dot(item_features, user_features)[train.nonzero()],
            train[train.nonzero()].toarray()[0]
        )

        test_rmse = helpers.calculate_rmse(
            np.dot(item_features, user_features)[test.nonzero()],
            test[test.nonzero()].toarray()[0]
        )

        print("Train error: %f, test error: %f" % (train_rmse, test_rmse))

        return user_features, item_features

    user_features, item_features = init_MF(train, n_features)

    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    nz_train, nz_row_colindices, nz_col_rowindices = helpers.build_index_groups(train)
    _, nz_user_itemindices = map(list, zip(*nz_col_rowindices))
    nnz_items_per_user = [len(i) for i in nz_user_itemindices]
    _, nz_item_userindices = map(list, zip(*nz_row_colindices))
    nnz_users_per_item = [len(i) for i in nz_item_userindices]

    prev_train_rmse = 100

    for it in range(n_epochs):
        user_features = update_user_feature(
            train,
            item_features,
            lambda_user,
            nnz_items_per_user,
            nz_user_itemindices
        )

        item_features = update_item_feature(
            train,
            user_features,
            lambda_item,
            nnz_users_per_item,
            nz_item_userindices
        )

        train_rmse = helpers.calculate_rmse(
            np.dot(item_features, user_features)[train.nonzero()],
            train[train.nonzero()].toarray()[0]
        )

        test_rmse = helpers.calculate_rmse(
            np.dot(item_features, user_features)[test.nonzero()],
            test[test.nonzero()].toarray()[0]
        )

        if verbose == 1:
            print("[Epoch %d / %d] train error: %f, test error: %f" % (it + 1, n_epochs, train_rmse, test_rmse))

        if (train_rmse > prev_train_rmse or
                abs(train_rmse - prev_train_rmse) < 1e-5):
            if verbose == 1:
                print('Algorithm has converged!')
            break

        prev_train_rmse = train_rmse

    if verbose == 0:
        print("[Epoch %d / %d] train error: %f, test error: %f" % (it + 1, n_epochs, train_rmse, test_rmse))

    np.save(user_features_file_path, user_features)
    np.save(item_features_file_path, item_features)

    return user_features, item_features

def get_ALS_predictions(ratings, train, test, n_features_array, lambda_user, lambda_item):
    """Return differents predictions corresponding to the given parameters

    Args:
        ratings (n_users x n_itens): The global dataset.
        train (n_users x n_items): The train dataset.
        test (n_users x n_items): The test dataset.
        n_features_array (N): Array representing the n_features parameter for the
                          different models to compute.
        lambda_user: This value is for all the models.
        lambda_item: This value is for all the models.

    Returns:
        X (n_users x n_items): Returns the global predictions for all the models.
        X_train: Returns the predictions for the non zero values of the train dataset.
        y_train: Returns the true labels for the train dataset.
        X_test: Returns the predictions for the non zero values of the test dataset.
        y_test: Returns the true labels for the test dataset.
    """
    n_models = len(n_features_array)

    X = np.zeros((n_models, train.shape[0] * train.shape[1]))
    X_train = np.zeros((n_models, train.nnz))
    X_test = np.zeros((n_models, test.nnz))

    y_train = ratings[train.nonzero()].toarray()[0]
    y_test = ratings[test.nonzero()].toarray()[0]

    for idx, n_features in enumerate(n_features_array):
        user_features, item_features = ALS(
            train, test, n_features, lambda_user, lambda_item
        )

        predicted_labels = np.dot(item_features, user_features)
        predicted_labels[predicted_labels > 5] = 5
        predicted_labels[predicted_labels < 1] = 1

        X[idx] = np.asarray(predicted_labels).reshape(-1)
        X_train[idx] = predicted_labels[train.nonzero()]
        X_test[idx] = predicted_labels[test.nonzero()]

    return X, X_train, y_train, X_test, y_test
