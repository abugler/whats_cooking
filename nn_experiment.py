import numpy as np
from config import n_folds, random_state
from util import training_testing_split, split_n_folds
from load_encoded import load_data
from nn import *
random = np.random.RandomState(random_state)

ingredients, cuisines, ingredients_one_hot, cuisines_one_hot = load_data()
del ingredients

# Split into n folds
feature_folds, target_folds = split_n_folds(ingredients_one_hot, cuisines, n_folds)


print("Folds created")
accuracies = []
# Run n_folds number experiments
for f in range(n_folds):
    print(f"Beginning experiment {f+1}")
    # Split folds
    train_features, train_targets, validation_features, validation_targets, test_features, test_targets =\
        training_testing_split(feature_folds, target_folds, f, validation_fold=(f+1) % len(feature_folds))
    print("Training and testing datasets created")

    model = CuisineClassifier(train_features.shape[1], cuisines_one_hot.shape[1])

    # Fit model
    train(model, train_features, train_targets, validation_features, validation_targets)
    print("Model is fitted to training data")

    # Predict on test data
    predicted_targets = model.predict(test_features)
    accuracy = np.sum(predicted_targets == test_targets) / predicted_targets.shape[0]
    print(f"Model is tested, accuracy is {accuracy}.")
    accuracies.append(accuracy)

    # Delete variables, because I only have 8GB RAM
    del model, predicted_targets, train_targets, train_features, test_features, \
        test_targets, validation_features, validation_targets

print(accuracies)
print(f"Mean Accuracy: {sum(accuracies) / len(accuracies)}")