import tensorflow as tf
# import os

# import shutil

# # Define the source and destination directories
# source_folder = 'C:\\Users\\user\\Desktop\\cancer_detection\\cancer_detection\\image_detection\\malignant'  # Replace with the actual path to your folder with the images
# destination_folder = 'C:\\Users\\user\\Desktop\\cancer_detection\\cancer_detection\\image_detection\\malignant_masks'  # Replace with the path to the folder where you want to copy the mask images

# # Create the destination folder if it doesn't exist
# if not os.path.exists(destination_folder):
#     os.makedirs(destination_folder)

# # Loop through all the files in the source folder
# for filename in os.listdir(source_folder):
#     # Check if the file is a mask image (ends with '_mask.png')
#     if filename.endswith('_mask.png'):
#         # Construct full file path
#         full_file_name = os.path.join(source_folder, filename)
#         # Check if it's a file and copy to the destination folder
#         if os.path.isfile(full_file_name):
#             shutil.copy(full_file_name, destination_folder)
# import os
# import shutil

# # Define the source and destination directories
# source_folder = 'C:\\Users\\user\\Desktop\\cancer_detection\\cancer_detection\\image_detection\\malignant'  # Replace with the actual path to your folder with the images
# destination_folder = 'C:\\Users\\user\\Desktop\\cancer_detection\\cancer_detection\\image_detection\\malignant_upload'  # Replace with the path to the folder where you want to copy the mask images

# # Create the destination folder if it doesn't exist
# if not os.path.exists(destination_folder):
#     os.makedirs(destination_folder)

# # Loop through all the files in the source folder
# for filename in os.listdir(source_folder):
#     # Check if the file is a mask image (ends with '_mask.png')
#     if not filename.endswith('_mask.png'):
#         # Construct full file path
#         full_file_name = os.path.join(source_folder, filename)
#         # Check if it's a file and copy to the destination folder
#         if os.path.isfile(full_file_name):
#             shutil.copy(full_file_name, destination_folder)

# print("Mask images have been copied successfully!")
