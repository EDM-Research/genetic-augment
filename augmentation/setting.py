import pickle


class AugmentationSetting:
    def __init__(self, config: dict):
        self.augmentations = None
        self.strengths = None
        self.probabilities = None
        self.pick_n = None

        if "augmentation" in config.keys():
            # Manual configuration of augmentation
            if isinstance(config["augmentation"], list):
                self.augmentations = config["augmentation"]
            else:
                self.augmentations = [config["augmentation"]]

            if "augmentation_strength" in config.keys():
                if isinstance(config["augmentation_strength"], list):
                    assert len(config["augmentation_strength"]) == len(self.augmentations)
                    self.strengths = config["augmentation_strength"]
                else:
                    self.strengths = [config["augmentation_strength"] for _ in range(len(self.augmentations))]
            else:
                self.strengths = [1.0 for _ in range(len(self.augmentations))]

            if "augmentation_probability" in config.keys():
                if isinstance(config["augmentation_probability"], list):
                    assert len(config["augmentation_probability"]) == len(self.augmentations)
                    self.probabilities = config["augmentation_probability"]
                else:
                    self.probabilities = [config["augmentation_probability"] for _ in range(len(self.augmentations))]
            else:
                self.probabilities = [1.0 for _ in range(len(self.augmentations))]

            if "augmentation_one_of" in config.keys() and config["augmentation_one_of"]:
                # for backwards compatibility
                self.pick_n = 1

            if "augmentation_pick_n" in config.keys():
                self.pick_n = config["augmentation_pick_n"]

        if "augmentation_file" in config.keys() and config["augmentation_file"] is not None:
            # Read augmentation object from file
            if self.augmentations is not None:
                print(f"Augmentation file and specification both found, overwriting specification")

            with open(config["augmentation_file"], 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    data = data[0]
                self.copy(data)

    def has_augmentations(self):
        return self.augmentations is not None

    def copy(self, data: 'AugmentationSetting'):
        self.augmentations = data.augmentations
        self.strengths = data.strengths
        self.probabilities = data.probabilities
        self.pick_n = data.pick_n

    def __str__(self):
        nested = isinstance(self.augmentations[0], list)
        output = f"Nested size: {len(self.augmentations[0]) if nested else 1}\n"
        if self.pick_n is not None:
            output += f"Mode: Pick-{self.pick_n}\n"
        else:
            output += f"Mode: Sequential\n"

        for i in range(len(self.augmentations)):
            output += f"{i}. \t"
            if nested:
                for j in range(len(self.augmentations[i])):
                    if j != 0:
                        output += " + "
                    output += f"{self.augmentations[i][j]}( p: {self.probabilities[i][j]:.2f}, s: {self.strengths[i][j]:.2f})"
            else:
                output += f"{self.augmentations[i]}( p: {self.probabilities[i]:.2f}, s: {self.strengths[i]:.2f})"

            output += "\n"

        return output
