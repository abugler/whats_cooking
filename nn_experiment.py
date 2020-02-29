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
print(f"Beginning NN experiment")
# Split folds
train_features, train_targets, validation_features, validation_targets, test_features, test_targets =\
        training_testing_split(feature_folds, target_folds, 0, validation_fold=1)
print("Training and testing datasets created")

model = CuisineClassifier(train_features.shape[1], cuisines_one_hot.shape[1])

# Fit model
# train(model, train_features, train_targets, validation_features, validation_targets)
model  = torch.load("best_nn")
print("Model is fitted to training data")

# Predict on test data
predicted_targets = np.exp(model(torch.tensor(test_features).float().cuda()).cpu().detach().numpy())
accuracy = np.sum(np.argmax(predicted_targets, axis=1) == test_targets) / test_targets.shape[0]
print(f"Model is tested, accuracy is {accuracy}")

