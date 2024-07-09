import yaml
import os
import shutil
import xml.etree.ElementTree as ET

from glob import glob


def get_classes_from_xml(path):
    """
    Extracts unique class names from XML files in the given directory.

    :param path: Path to the directory containing XML files.
    :return: Set of unique class names.
    """
    classes = set()
    for xml_file in glob(os.path.join(path, '**', '*.xml'), recursive=True):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            classes.add(member.find('name').text)
    return classes

def create_dataset_yaml(dataset_path, yaml_file_path):
    """
    Creates a YAML file with dataset information, inferring classes from XML files.

    :param dataset_path: Path to the root directory of the dataset.
    :param yaml_file_path: Path to save the YAML file.
    """

    # Extract class names and count from XML files
    class_names = list(get_classes_from_xml(os.path.join(dataset_path, "labels")))
    num_classes = len(class_names)

    # Construct the data dictionary
    data = {
        'train': os.path.join(dataset_path, "images", "train"),
        'val': os.path.join(dataset_path, "images", "valid"),
        'nc': num_classes,
        'names': class_names
    }

    # Write the data to the YAML file
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

    print(f'Data has been written to {yaml_file_path}')


def reorganize_dataset(path):
    """
    Reorganizes a dataset from:
        dataset
         - train_images
            - .png
            - .xml
         - valid_images
           - .png
           - .xml

    to:
        dataset
         -images
          -- train
          -- valid
         - labels
          -- train
          -- valid

    :param path: Path to the root directory of the dataset
    """

    # New directory structure
    images_train_path = os.path.join(path, "images", "train")
    images_valid_path = os.path.join(path, "images", "valid")
    labels_train_path = os.path.join(path, "labels", "train")
    labels_valid_path = os.path.join(path, "labels", "valid")

    # Create new directories if they don't exist
    os.makedirs(images_train_path, exist_ok=True)
    os.makedirs(images_valid_path, exist_ok=True)
    os.makedirs(labels_train_path, exist_ok=True)
    os.makedirs(labels_valid_path, exist_ok=True)

    # Function to move files based on extension
    def move_files(src_folder, dest_folder, extension):
        for file in glob(os.path.join(src_folder, f'*{extension}')):
            shutil.move(file, dest_folder)

    # Move .png and .xml files to their new locations
    move_files(os.path.join(path, "train_images"), images_train_path, ".png")
    move_files(os.path.join(path, "train_images"), labels_train_path, ".xml")
    move_files(os.path.join(path, "valid_images"), images_valid_path, ".png")
    move_files(os.path.join(path, "valid_images"), labels_valid_path, ".xml")

    # Remove old folders if they are empty
    if not os.listdir(os.path.join(path, "train_images")):
        shutil.rmtree(os.path.join(path, "train_images"))
    if not os.listdir(os.path.join(path, "valid_images")):
        shutil.rmtree(os.path.join(path, "valid_images"))

    print('Dataset reorganized')


reorganize_dataset(f"detection_single")
create_dataset_yaml(f"detection_single", f"train_configs/detection_single_train_config.yaml")