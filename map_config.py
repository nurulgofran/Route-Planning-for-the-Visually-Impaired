'''
change the paths below as described
'''

'''for inference.py'''
'''path of model fine_tuned on mapillary dataset'''
map_ckp_path = "fine_tuned_mapillary.ckpt"

'''directory of images to segment'''
img_dir = "Data/Mapillary-Vistas-1000-sidewalks/testing/images"

'''output directory where segmented images are saved'''
op_dir = "Data/inference"


'''for training_pipeline.py'''
'''directory where mapillary training images and labels folder are present'''
mapillary_train_path = "Data/Mapillary-Vistas-1000-sidewalks/training"

'''directory where mapillary validation images and labels folder are present'''
mapillary_val_path = "Data/Mapillary-Vistas-1000-sidewalks/validation"

'''directory where mapillary testing images and labels folder are present'''
mapillary_test_path = "Data/Mapillary-Vistas-1000-sidewalks/testing"

'''checkpoint path of the model trained on cityscapes dataset'''
city_ckpt_path = "trained_on_cityscapes.ckpt"



'''for convert_masks_to_grayscale.py'''
'''path to config.json file in mapillary dataset'''
json_path = "Data/Mapillary-Vistas-1000-sidewalks/config.json"

'''directory of rgb masks to convert to grayscale'''
masks_path = "Data/Mapillary-Vistas-1000-sidewalks/training/labels"

'''directory where converted grayscale masks will be saved'''
op_path = "Data/Mapillary-Vistas-1000-sidewalks/training/grayscale_labels"








valid_classes = ["unlabeled", "road", "sidewalk", "traffic light",
                 "traffic sign (front)", "person", "car", "bicycle"] # total 7 + 1 classes

label_colors = {0: [0, 0, 0],
                1: [128, 64, 128],
                2: [244, 35, 232],
                3: [250, 170, 30],
                4: [220, 220, 0],
                5: [220, 20, 60],
                6: [0, 0, 142],
                7: [119, 11, 32]}

n_classes = len(valid_classes)

num_epochs = 400
batch_size = 4
learning_rate = 0.001