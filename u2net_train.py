

import os
import glob
from keras.optimizers import Adam
from torchvision import transforms
from data_loader import RescaleT, RandomCrop, ToTensorLab, sal_generator

from model import U2NET
from model import U2NETP

# ------- 2. set the directory of training dataset --------

model_name = 'u2net' #'u2netp'

data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
tra_image_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'im_aug' + os.sep)
tra_label_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'gt_aug' + os.sep)

image_ext = '.jpg'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

epoch_num = 100000
batch_size_train = 12
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
	img_name = img_path.split(os.sep)[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = sal_generator( batch_size_train,
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))


# ------- 3. define model --------

if(model_name == 'u2net'):
    net = U2NET(3, 1)
elif(model_name == 'u2netp'):
    net = U2NETP(3,1)

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = Adam(lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")

save_frq = 2000 # save the model every 2000 iterations

net.compile(Adam(lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0), loss='binary_crossentropy')
net.fit(
        salobj_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        workers=1,
        epochs=epoch_num)

