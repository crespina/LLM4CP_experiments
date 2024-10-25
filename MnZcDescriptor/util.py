import pickle

def save_instances(savepath, instances):

    save_path = savepath + ".pkl"
    with open(save_path, "wb") as file:
        pickle.dump(instances, file)

    print(f"Pickle file saved to {save_path}")


def load_instances(savepath):

    with open(savepath, "rb") as file:
        instances = pickle.load(file)

    print("Instances loaded")
    return instances
