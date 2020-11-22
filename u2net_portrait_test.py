

import os
import glob
import numpy as np
from PIL import Image
from skimage import io
from torchvision import transforms
from data_loader import RescaleT, ToTensorLab, sal_generator

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

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
    imo.save(d_dir+'/'+imidx+'.png')

def main():

    # --------- 1. get image path and name ---------
    model_name = 'u2net_portrait' #u2netp

    image_dir = './test_data/test_portrait_images/portrait_im'
    prediction_dir = './test_data/test_portrait_images/portrait_results'
    if(not os.path.exists(prediction_dir)):
        os.mkdir(prediction_dir)
    model_dir = './saved_models/u2net_portrait/u2net_portrait.pth'
    img_name_list = glob.glob(image_dir+'/*')
    print("Number of images: ", len(img_name_list))

    # --------- 2. dataloader ---------

    test_salobj_dataset = sal_generator(batch_size=1,
                                        img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(512),
                                                                      ToTensorLab(flag=0)])
                                        )

    # --------- 3. model define ---------

    print("...load U2NET---173.6 MB")
    net = U2NET(3,1)

    net.load(model_dir)

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataset):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        d1,d2,d3,d4,d5,d6,d7= net(data_test, steps=1)

        # normalization
        pred = 1.0 - d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
