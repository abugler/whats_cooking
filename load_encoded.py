import numpy as np

def load_data(dtype=np.int32):
    ## Import necessary matrices
    ingredients = np.load("ingredients_one_hot.npy", allow_pickle=True)
    cuisines = np.load("cuisines.npy", allow_pickle=True)
    cuisines_one_hot = np.load("cuisines_one_hot.npy", allow_pickle=True)
    ingredients_one_hot = np.load("ingredients_one_hot.npy", allow_pickle=True)
    ingredients = ingredients.astype(dtype)
    cuisines = cuisines.astype(dtype)
    ingredients_one_hot = ingredients_one_hot.astype(dtype)
    cuisines_one_hot = cuisines_one_hot.astype(dtype)

    return ingredients, cuisines, ingredients_one_hot, cuisines_one_hot
