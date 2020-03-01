from sklearn.ensemble import RandomForestClassifier
import numpy as np
from config import random_state, n_folds
random = np.random.RandomState(random_state)
from util import training_testing_split, split_n_folds
from load_encoded import load_data

ingredients, cuisines, ingredients_one_hot, cuisines_one_hot = load_data()
del ingredients_one_hot
del cuisines_one_hot

# Split into n folds
feature_folds, target_folds = split_n_folds(ingredients, cuisines, n_folds, dummy=True)
del ingredients
del cuisines

print("Folds created")
accuracies = []
# Run n_folds number experiments
for f in range(n_folds):
    print(f"Beginning experiment {f+1}")
    # Split folds
    train_features, train_targets, test_features, test_targets =\
        training_testing_split(feature_folds, target_folds, f)
    print("Training and testing datasets created")

    # Initialize model
    model = RandomForestClassifier()

    # Fit model
    model.fit(train_features, train_targets)
    print("Model is fitted to training data")

    # Predict on test data
    predicted_targets = model.predict(test_features)
    accuracy = np.sum(predicted_targets == test_targets) / predicted_targets.shape[0]
    print(f"Model is tested, accuracy is {accuracy}.")
    accuracies.append(accuracy)

    # Delete variables, because I only have 8GB RAM
    del model, predicted_targets, train_targets, train_features, test_features, test_targets

print(accuracies)
print(f"Mean Accuracy: {sum(accuracies) / len(accuracies)}")

