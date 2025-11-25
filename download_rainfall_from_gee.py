import ee

ee.Initialize()

# Dates
start_date = ee.Date('2023-07-01')
end_date   = ee.Date('2023-10-01')   # include September fully

# GDRIVE_FOLDER = 'lower_ganga_basin_07_to_09'
GDRIVE_FOLDER = 'Mahesh_srinivas/lbg_rainfall'

BASIN_NAME = 'Lower_ganga'
# All basin names:
# List (34 elements)
# 0: Krishna
# 1: Qura_qush
# 2: Indus
# 3: Jhelum
# 4: Chenab
# 5: Ravi
# 6: Beas
# 7: Sutlej
# 8: Ghaghar
# 9: Churu
# 10: Barmer
# 11: Luni
# 12: Kutch
# 13: Bhadar
# 14: Sabarmati
# 15: Mahi
# 16: Chambal
# 17: Upper_ganga
# 18: Tapi
# 19: Yamuna
# 20: Brahmani
# 21: Mahanadi
# 22: Narmada
# 23: Lower_ganga
# 24: Imphal
# 25: Surma
# 26: Bhatsol
# 27: Vamsadhara
# 28: Godavari
# 29: Pennar
# 30: Periyar
# 31: Vaippar
# 32: Cauvery
# 33: Brahmputra


# Load basin table and filter to Brahmani
pan_basin_table = ee.FeatureCollection('projects/corestack-datasets/assets/datasets/Basin_pan_india')
basin_feature = pan_basin_table.filter(ee.Filter.eq('ba_name', BASIN_NAME)).first()
basin_geom = basin_feature.geometry()

print("Basin feature:", basin_feature.getInfo())


# Your feature collection (replace with actual asset)
# table = ee.FeatureCollection('users/your_username/your_table')

# Rainfall dataset
# each image is
#     Spatial Resolution: 0.1° × 0.1° (approximately 11.1 km at the equator)
#     Temporal Resolution: Hourly
#     Raw hourly precipitation rate (mm/hr)
rainfall_collection = (
    ee.ImageCollection('JAXA/GPM_L3/GSMaP/v6/operational')
    .filterBounds(basin_geom)
    .filterDate(start_date, end_date)
    .select('hourlyPrecipRate')
)

print('Total Images in Collection:', rainfall_collection.size().getInfo())

# Function: sum every 3 consecutive images
# so, raster would contain 3hour rainfall sum. (in )
def sum_every_3(collection):
    size = collection.size()
    num_groups = size.divide(3).floor()
    collection_list = collection.toList(size)

    def group_fn(i):
        i = ee.Number(i)
        start = i.multiply(3)
        images = collection_list.slice(start, start.add(3))
        summed = ee.ImageCollection(images).sum().set('group', i)
        return summed

    summed_list = ee.List.sequence(0, num_groups.subtract(1)).map(group_fn)
    return ee.ImageCollection(summed_list)

summed_rainfall = sum_every_3(rainfall_collection)
print("Summed Rainfall Collection size:", summed_rainfall.size().getInfo())
# Export each summed image for Brahmani
def export_image(img):
    group = img.get('group').getInfo()
    task = ee.batch.Export.image.toDrive(
        image = img.clip(basin_feature.geometry()),
        description = f'Summed_Rainfall_Group_{group}',
        folder = GDRIVE_FOLDER,
        scale = 30,
        maxPixels = 1e13
    )
    task.start()

# Trigger exports
for img in summed_rainfall.toList(summed_rainfall.size()).getInfo():
    ee_img = ee.Image(img['id'])
    export_image(ee_img)
