import caption
import numpy as np
import pandas as pd

root_dir = '../../Codes/data/'
imgfile_0 = pd.read_csv('full_picture.csv')
imgfile_caption_2d_list = [['post_id', 'caption', 'CLASSNAME']]
for row in range(imgfile_0.shape[0]):
    im_id = imgfile_0.at[row, 'post_id']
    class_label = imgfile_0.at[row, 'CLASSNAME']
    img_filename = root_dir+str(class_label)+'/'+im_id+'.jpg'
    predicted_caption = '@@@@@'
    try:
        predicted_caption = caption.generate_caption(img_filename, 50)
        print(predicted_caption)
    except Exception as ex:
        print(ex)
    imgfile_caption_2d_list.append([im_id, predicted_caption, class_label])

image_captions = pd.DataFrame(imgfile_caption_2d_list[1:], columns=imgfile_caption_2d_list[0])
image_captions.to_csv('image_captions.csv', encoding='utf8')
