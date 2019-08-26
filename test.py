#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: 
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2019年7月11日
#######################################################################
from PIL import Image
import numpy as np
def produceImage(file_in, width, height, file_out):
    image = Image.open(file_in)
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    resized_image.save(file_out)
 
if __name__ == '__main__':
    file_path = 'C:/Users/cuiyufei/Desktop/'
    file_name = 'profile.jpg'
    file_in = file_path + file_name
    file_out_s = file_path + 's.jpg'
    file_out_b = file_path + 'b.jpg'
    width_mini = 371
    height_mini = 238
    width_scale = 1600
    height_scale = 800
    img_obj = Image.open(file_in)
    img_array = np.array(img_obj, dtype=np.uint8)
    print img_array
    produceImage(file_in, width_mini, height_mini, file_out_s)
    produceImage(file_in, width_scale, height_scale, file_out_b)