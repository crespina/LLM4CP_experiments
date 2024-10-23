import pickle

def save_instances(savename, instances):

    save_path = "MnZcDescriptor\instances\\" + savename + ".pkl"
    with open(save_path, "wb") as file:
        pickle.dump(instances, file)

    print(f"Pickle file saved to {save_path}")


def load_instances(savename):

    filename = "MnZcDescriptor\instances\\" + savename + ".pkl"

    with open(filename, "rb") as file:
        instances = pickle.load(file)

    print("Instances loaded")
    return instances
