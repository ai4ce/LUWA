import os
from PIL import Image
import zipfile



def extract_zip_file(zip_file_path, extracted_folder_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder_path)



def crop_and_save_images(image_folder):
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path)


        width, height = image.size


        image_base_name = os.path.splitext(image_name)[0]
        output_folder = os.path.join(image_folder, image_base_name)
        os.makedirs(output_folder, exist_ok=True)

        crop_number = 1


        for top in range(0, height, 256):
            for left in range(0, width, 256):

                if left + 256 <= width and top + 256 <= height:
                    cropped_image = image.crop((left, top, left + 256, top + 256))
                    cropped_image_name = f"{image_base_name}_{crop_number}.bmp"
                    cropped_image.save(os.path.join(output_folder, cropped_image_name))
                    crop_number += 1



def create_zip_from_folder(folder_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w') as zip_file:
        for folder_name in os.listdir(folder_path):
            folder_path = os.path.join(folder_path, folder_name)
            if os.path.isdir(folder_path):
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    zip_file.write(image_path, os.path.relpath(image_path, folder_path))



zip_file_path = 'path_to_your_zip_file.zip'
extracted_folder_path = 'path_to_extracted_folder'
output_zip_path = 'path_to_output_zip.zip'


extract_zip_file(zip_file_path, extracted_folder_path)


crop_and_save_images(extracted_folder_path)


create_zip_from_folder(extracted_folder_path, output_zip_path)
