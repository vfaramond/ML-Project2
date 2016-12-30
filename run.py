from helpers import load_data, split_data, calculate_rmse, get_linear_blend_clf
from helpers import generate_submission
from ALS import get_ALS_predictions

# Load the data
path_dataset = "data/data_train.csv"

print('Loading the data...')
ratings = load_data(path_dataset)

# Split in training and testing sets
print('\nSplitting the data in train and test sets...')
train, test = split_data(ratings, p_test=0.1)

# Generate predictions for 6 different models
X, X_train, y_train, X_test, y_test = get_ALS_predictions(
    ratings,
    train,
    test,
    n_features_array=range(1, 31),
    lambda_user=0.2,
    lambda_item=0.02
)

# Linear blend of the previous models computed on the test set.
clf = get_linear_blend_clf(X_test, y_test)

print('\nRMSE Train: %f' % calculate_rmse(clf.predict(X_train.T), y_train))

print('RMSE Test: %f' % calculate_rmse(clf.predict(X_test.T), y_test))

print('Weights of the different models:', clf.coef_)

# Final predicted labels matrix
predicted_labels = clf.predict(X.T).reshape(ratings.shape)
predicted_labels[predicted_labels > 5] = 5
predicted_labels[predicted_labels < 1] = 1

# Generate the CSV submission file
generate_submission(predicted_labels)
