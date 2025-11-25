"""
Might need this if GEE cuts raster into multiple files.
"""

import os
import re
import rasterio
import numpy as np
from rasterio.merge import merge
import tqdm

INPUT_FOLDER = "../tifs/lgb/rainfall_3H_2023-07-01_2023-10-01/"
OUTPUT_FOLDER = "../tifs/lgb/merged_rainfall_3H_2023-07-01_2023-10-01/"
FILE_PREFIX = "3H_Rain_"

def merge_tif_files(input_folder, output_folder, file_prefix):
    os.makedirs(output_folder, exist_ok=True)

    file_groups = {}
    pattern = rf"{file_prefix}(\d+)-\d{{10}}-\d{{10}}\.tif$"

    for file in os.listdir(input_folder):
        match = re.match(pattern, file)
        if match:
            index = match.group(1)  # Extract index number
            file_groups.setdefault(index, []).append(os.path.join(input_folder, file))

    for index, file_list in tqdm.tqdm(file_groups.items()):
        if len(file_list) > 1:
            print('number of files in group : ', len(file_list))
            datasets = [rasterio.open(f) for f in file_list]
            merged_array, transform = merge(datasets)

            merged_array = merged_array.astype(np.float32)

            out_meta = datasets[0].meta.copy()
            out_meta.update({
                "height": merged_array.shape[1],
                "width": merged_array.shape[2],
                "transform": transform,
                "dtype": "float32",
                "compress": "LZW"
            })

            output_file = os.path.join(output_folder, f"{file_prefix}-{index}_merged.tif")
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(merged_array)

            for ds in datasets:
                ds.close()

            print(f"Merged {len(file_list)} images for index {index} -> {output_file}")
        else:
            print(f"Skipping index {index} (only 1 file found)")

    print("Merging completed! All images are saved in float32 format.")


if __name__ == "__main__":
    merge_tif_files(INPUT_FOLDER, OUTPUT_FOLDER, FILE_PREFIX)




################################### convert the data type ###########################################

# import os
# import rasterio
# import numpy as np

# # Folder containing the merged .tif files
# folder_path = "./merged_rain_dataset"  # Update this to your actual path

# # Process each .tif file in-place
# for file in os.listdir(folder_path):
#     if file.endswith(".tif"):
#         file_path = os.path.join(folder_path, file)

#         # Open the raster file
#         with rasterio.open(file_path, "r+") as src:
#             data = src.read(1)  # Read first band
#             data_float32 = data.astype(np.float32)  # Convert to float32

#             # Update metadata
#             src.meta.update(dtype="float32")

#             # Overwrite with converted data
#             src.write(data_float32, 1)

#         print(f"Updated {file} to Float32")

# print("All files updated to Float32!")


########################################### Rename Images #####################################################
