import os, shutil

raw_dataset_dir = 'C:/Users/izanp/Downloads/train'

splitted_dataset_dir = 'C:/Users/izanp/Downloads/cats_and_dogs_data'

train_cats_dir = os.path.join(splitted_dataset_dir, 'train/cat')
validation_cats_dir = os.path.join(splitted_dataset_dir, 'validation/cat')
test_cats_dir = os.path.join(splitted_dataset_dir, 'test/cat')
train_dogs_dir = os.path.join(splitted_dataset_dir, 'train/dog')
validation_dogs_dir = os.path.join(splitted_dataset_dir, 'validation/dog')
test_dogs_dir = os.path.join(splitted_dataset_dir, 'test/dog')

if os.path.exists(splitted_dataset_dir):
    shutil.rmtree(splitted_dataset_dir)

os.makedirs(splitted_dataset_dir)
os.makedirs(train_cats_dir)
os.makedirs(validation_cats_dir)
os.makedirs(test_cats_dir)
os.makedirs(train_dogs_dir)
os.makedirs(validation_dogs_dir)
os.makedirs(test_dogs_dir)

src_cat_format = 'cat.{}.jpg'
src_dog_format = 'dog.{}.jpg'
dst_format = '{}.jpg'

#1000 primeras imagenes para entrenamiento
for i in range(1000):
 src = os.path.join(raw_dataset_dir, src_cat_format.format(i))
 dst = os.path.join(train_cats_dir, dst_format.format(i))
 shutil.copyfile(src, dst)

 src = os.path.join(raw_dataset_dir, src_dog_format.format(i))
 dst = os.path.join(train_dogs_dir, dst_format.format(i))
 shutil.copyfile(src, dst)
print("Imagenes de entrenamiento.")

# 500 siguientes imagenes para validacion
for i in range(1000, 1500):
 src = os.path.join(raw_dataset_dir, src_cat_format.format(i))
 dst = os.path.join(validation_cats_dir, dst_format.format(i))
 shutil.copyfile(src, dst)

 src = os.path.join(raw_dataset_dir, src_dog_format.format(i))
 dst = os.path.join(validation_dogs_dir, dst_format.format(i))
 shutil.copyfile(src, dst)
print("Imagenes de validacion.")

#500 siguientes imagenes para test
for i in range(1500, 2000):
 src = os.path.join(raw_dataset_dir, src_cat_format.format(i))
 dst = os.path.join(test_cats_dir, dst_format.format(i))
 shutil.copyfile(src, dst)

 src = os.path.join(raw_dataset_dir, src_dog_format.format(i))
 dst = os.path.join(test_dogs_dir, dst_format.format(i))
 shutil.copyfile(src, dst)
print("Imagenes de tests.")