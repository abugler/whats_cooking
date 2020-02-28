import json
import numpy as np

json_file = open("train.json", "r")
json_data = json.load(json_file)
json_file.close()

cuisine_dict = {}
ingredient_dict = {}

cuisine_list = []
ingredient_list = []
for recipe in json_data:
    cuisine = recipe["cuisine"]
    ingredients = recipe["ingredients"]

    if cuisine not in cuisine_dict.keys():
        cuisine_dict[cuisine] = len(cuisine_dict)
    cuisine_list.append(cuisine_dict[cuisine])

    encode_ingredients = []
    for ingredient in ingredients:
        if ingredient not in ingredient_dict.keys():
            ingredient_dict[ingredient] = len(ingredient_dict)
        encode_ingredients.append(ingredient_dict[ingredient])
    ingredient_list.append(encode_ingredients)


cuisine_list_np = np.zeros((len(cuisine_list), len(cuisine_dict)))
for i in range(len(cuisine_list)):
    cuisine_list_np[i, cuisine_list[i]] = 1

ingredient_list_np = np.zeros((len(ingredient_list), len(ingredient_dict)))
for i in range(len(ingredient_list)):
    for j in ingredient_list[i]:
        ingredient_list_np[i, j] = 1

# One hot
cuisine_dict_np = []
for key, value in cuisine_dict.items():
    cuisine_dict_np.append([key, value])

# One hot
ingredient_dict_np = []
for key, value in ingredient_dict.items():
    ingredient_dict_np.append([key, value])

np.save("cuisines_one_hot.npy", cuisine_list_np)
np.save("cuisines.npy", np.array(cuisine_list))
np.save("ingredients_one_hot.npy", ingredient_list_np)

np.save("cuisine_dict.npy", np.array(cuisine_dict_np))
np.save("ingredient_dict.npy", np.array(ingredient_dict_np))
