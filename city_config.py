'''
change this path to the directory where leftImg8bit and gtFine
folders of cityscapes dataset are present
'''
dataset_path = "Data/"




valid_classes = {0: "unlabelled",
                 7: "road",
                 8: "sidewalk",
                 19: "traffic light",
                 20: "traffic sign",
                 24: "person",
                 26: "car",
                 33: "bicycle"} # 7+1 classes 

class_map = dict(zip(valid_classes.keys(), range(len(valid_classes))))
n_classes = len(valid_classes)

num_epochs = 200
batch_size = 8
learning_rate = 0.001