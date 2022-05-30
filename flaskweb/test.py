from cgi import print_arguments
import imp
import paddle
import paddlers as pdrs
import numpy as np
import os.path as osp
from skimage.io import imread, imsave
from util_crop_recons.crop import crop_img, recons_prob_map, quantize

predictor = pdrs.deploy.Predictor('/home/guangjun/PaddleRS/flask_paddlers/inference_model', use_gpu=True)
# image_befor_path = '/home/guangjun/PaddleRS/flask_paddlers/A_train_5.png'
# image_after_path = '/home/guangjun/PaddleRS/flask_paddlers/B_train_5.png'
image_befor_path = '/home/guangjun/PaddleRS/flask_paddlers/A_train_4_0_3.png'
image_after_path = '/home/guangjun/PaddleRS/flask_paddlers/B_train_4_0_3.png'

result = predictor.predict(img_file=(image_befor_path, image_after_path))
score_map = result[0]['score_map'][:, :, -1]
imsave('./change_403_infer.png', quantize(score_map>0.5), check_contrast=False)


# im_befor = imread(image_befor_path)
# im_after = imread(image_after_path)
# imgs = [im_befor, im_after]
# patches = crop_img(imgs, (1024, 1024), 256, 64)
# print(len(patches))
# print(len(patches[0]))


# patch_res = []
# for t in patches:
#     print('-----')
#     result = predictor.predict(img_file=(t[0], t[1]))
#     patch_res.append(result[0]['score_map'][:,:,1])
# prob = recons_prob_map(patch_res, (1024,1024), 256, 64)
# out = quantize(prob>0.5)
# # print(out.shape)
# imsave('./change_5.png', out, check_contrast=False)



##############################################################
# img_file = (image_befor_path, image_after_path)
# if isinstance(img_file, (str, np.ndarray, tuple)):
#     # images = [img_file]
#     print(type(img_file))
#     images = [img_file]
#     print(type(images))
# else:
#     img_file = img_file
#     print("no")

# pred = paddle.zeros
# for imgb, imga in patch:


# result = predictor.predict(img_file=img_file)
# label_map = result[0]['label_map']
# score_map = result[0]['score_map']
# print(type(label_map)) 
# print(type(score_map))
# print(label_map.shape)
# print(score_map.shape)
# print(label_map)
# print(score_map)


# score = np.sum(score_map, axis=2)
# print(score)
# mask = np.unique(label_map)
# tmp = {}
# for v in mask:
#     tmp[v] = np.sum(label_map == v)
# print(tmp)

# for i in range(256):
#     for j in range(256):
#         if(label_map[i, j] == 1):
#             print(score_map[i,j,:])





