import os
from PIL import Image
import zipfile

def crop_image_without_black_edges(image_path, output_folder, crop_size=256):
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure there are no alpha channels causing black edges
    img_width, img_height = img.size


    img_no_black_edges = img.crop(img.getbbox())  # Crop to the bounding box that excludes black areas
    img_no_black_edges_width, img_no_black_edges_height = img_no_black_edges.size


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    crop_count = 0
    for i in range(0, img_no_black_edges_width, crop_size):
        for j in range(0, img_no_black_edges_height, crop_size):
            box = (i, j, min(i + crop_size, img_no_black_edges_width), min(j + crop_size, img_no_black_edges_height))
            cropped_img = img_no_black_edges.crop(box)

            crop_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{crop_count}.bmp"
            cropped_img.save(os.path.join(output_folder, crop_filename))
            crop_count += 1



def process_images_in_folder(input_folder, output_folder_base, crop_size=256):
    for image_file in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_file)
        if image_file.endswith('.bmp'):

            output_folder = os.path.join(output_folder_base, os.path.splitext(image_file)[0])
            crop_image_without_black_edges(image_path, output_folder, crop_size)



input_folder = 'path_to_extracted_folder'
output_folder_base = 'path_to_output_folder'


process_images_in_folder(input_folder, output_folder_base)


def zip_folder(folder_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))



zip_folder(output_folder_base, 'cropped_images.zip')
