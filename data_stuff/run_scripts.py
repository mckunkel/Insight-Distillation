from get_data import run_data_checker
from clean_data import clean_data
from split_save_data import split_save
from split_and_save_images import run_image_parser

if __name__ == '__main__':
    # in this order
    # 1. Check if data exists. If not get data
    # 2. Clean  and run decoder on data
    # 3. Create the training.csv files
    # 4. Split and save the training and validation sets
    run_data_checker()
    clean_data()
    split_save()
    run_image_parser()