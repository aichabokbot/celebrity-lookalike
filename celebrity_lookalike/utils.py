import os
import pickle


def build_image_index():
    # Get the names of the celebrities
    names = set()
    for dirpath, dirnames, filenames in os.walk("data"):
        for filename in filenames:
            if ".npz" in filename:
                names.add(" ".join(filename.split("_")[:-1]))

    # Create two dictionaries
    # id_to_name: {id: name of celebrity}
    # name_to_id: {name of celebrity: id}
    id_to_name = {i: v for i, v in enumerate(names)}
    name_to_id = {v: i for i, v in enumerate(names)}

    # Create two dictionaries
    # id_to_images: {id: path of image}
    # image_to_id: {path of image: id}
    id_to_images = dict()
    image_to_id = dict()

    id = 0
    for dirname, _, filenames in os.walk("data"):
        for filename in filenames:
            if ".npz" in filename:
                name = " ".join(filename.split("_")[:-1])
                id_to_name[id] = name
                id_to_images[id] = os.path.join(dirname, filename)
                id += 1

    return id_to_name, name_to_id, id_to_images, image_to_id
