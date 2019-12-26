import os
import shutil
import math
import argparse
import random

def move_from_raw_to_sorted(input_dataset_dir, label):
    cwd = os.getcwd()
    datasets_dir = input_dataset_dir
    datasets_dir = os.path.join(cwd, datasets_dir)
    output_dir = os.path.join(cwd, 'sorted_{}'.format(input_dataset_dir))
    master_file_list = []
    for root,dirs,files in os.walk(datasets_dir):
        master_file_list.extend([os.path.join(root, f) for f in files if '.jpg' in f])

    total_files = len(master_file_list)

    # Clean out old directory
    for filename in os.listdir(output_dir):
        os.unlink(os.path.join(output_dir, filename))

    num_digits = int((math.log(total_files) / math.log(10)) + 1)

    for i in range(len(master_file_list)):
        filepath = master_file_list[i]
        new_name = str(i)
        new_name = new_name.rjust(num_digits, '0')
        new_name = '{}.{}.jpg'.format(label, new_name)
        new_name = os.path.join(output_dir, new_name)
        shutil.copy(filepath, new_name)

def move_from_sorted_to_validation(input_dir):
    cwd = os.getcwd()
    training_dir = os.path.join(cwd, 'training')
    validation_dir = os.path.join(cwd, 'validation')

    all_files = os.listdir(input_dir)
    random.shuffle(all_files)

    training_percent = 0.80

    training_index = int(training_percent*len(all_files))

    training_files = all_files[:training_index]
    validation_files = all_files[training_index:]

    new_dir = training_files[0].split('.')[0]
    print(new_dir)

    training_dir = os.path.join(training_dir, new_dir)
    validation_dir = os.path.join(validation_dir, new_dir)
    
    try:
        os.mkdir(training_dir)
    except Exception as e:
        pass
    try:
        os.mkdir(validation_dir)
    except Exception as e:
        pass
    for filename in os.listdir(training_dir):
        os.unlink(os.path.join(training_dir, filename))
    for filename in os.listdir(validation_dir):
        os.unlink(os.path.join(validation_dir, filename))

    for f in training_files:
        new_filename = os.path.join(training_dir, f)
        filename = os.path.join(input_dir, f)
        shutil.copy(filename, new_filename)

    for f in validation_files:
        new_filename = os.path.join(validation_dir, f)
        filename = os.path.join(input_dir, f)
        shutil.copy(filename, new_filename)
        
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sorted", required=False,
                        help="move directories of unsorted images into renamed section")
    parser.add_argument("-v", "--validation", required=False,
                        help="take sorted dataset and move into training/validation dirs")
    args = parser.parse_args()
    # args = vars(parser.parse_args())
    return args
        
def main(args):
    pass

if __name__ == '__main__':
    options = create_parser()
    random.seed(0)
    if options.sorted:
        move_from_raw_to_sorted('dataset', 'duck')
        move_from_raw_to_sorted('not_dataset', 'notduck')
    elif options.validation:
        move_from_sorted_to_validation('sorted_dataset')
        move_from_sorted_to_validation('sorted_not_dataset')

    
    # main('dataset', 'duck')
    # main('not_dataset', 'notduck')
