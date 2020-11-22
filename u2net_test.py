

import os
import glob
import numpy as np
from PIL import Image
from skimage import io
from torchvision import transforms
from data_loader import RescaleT, ToTensorLab, SalObjDataset, sal_generator

from model import U2NET
from model import U2NETP

def normPRED(d):
    ma = np.max(d)
    mi = np.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def main():

    # --------- 1. get image path and name ---------
    model_name='u2net' # u2netp

    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------

    test_salobj_dataset = SalObjDataset(
                                        img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
                                        )

    if(model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif(model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)

    for i_test, data_test in enumerate(len(img_name_list)):
        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        d1, d2, d3, d4, d5, d6, d7 = net.predict(test_salobj_dataset, steps=1)
        pred = d1[:,0,:,:]
        pred = normPRED(pred)
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)
        del d1, d2, d3, d4, d5, d6, d7

if __name__ == "__main__":
    main()
