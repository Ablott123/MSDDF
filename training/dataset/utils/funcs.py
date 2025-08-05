import cv2
import albumentations as alb
import numpy as np
from PIL import Image
"""
目的：在数据增强过程中随机降低图像的尺寸。
它可以用于改变图像的分辨率和控制图像的复杂度，以增加数据的多样性
"""

#对源图像进行调整，将图像的高度和宽度进行调整，
# 并且将调整大小的图像进行零填充或者中心裁剪，具有和原始图像相同的大小
def crop_face(img, landmark, margin=False, only_img=False,abs_coord = False,phase='train'):
	assert phase in ['train', 'val', 'test']
	# crop face------------------------------------------
	H, W = len(img), len(img[0])

	assert landmark is not None

	H, W = len(img), len(img[0])
	x0, y0 = landmark[:68, 0].min(), landmark[:68, 1].min()
	x1, y1 = landmark[:68, 0].max(), landmark[:68, 1].max()
	w = x1 - x0
	h = y1 - y0
	w0_margin = w / 8  # 0#np.random.rand()*(w/8)
	w1_margin = w / 8
	h0_margin = h / 2  # 0#np.random.rand()*(h/5)
	h1_margin = h / 5

	if margin:
		w0_margin *= 4
		w1_margin *= 4
		h0_margin *= 2
		h1_margin *= 2
	elif phase == 'train':
		w0_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
		w1_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
		h0_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
		h1_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
	else:
		w0_margin *= 0.5
		w1_margin *= 0.5
		h0_margin *= 0.5
		h1_margin *= 0.5

	y0_new = max(0, int(y0 - h0_margin))
	y1_new = min(H, int(y1 + h1_margin) + 1)
	x0_new = max(0, int(x0 - w0_margin))
	x1_new = min(W, int(x1 + w1_margin) + 1)

	img_cropped = img[y0_new:y1_new, x0_new:x1_new]
	if landmark is not None:
		landmark_cropped = np.zeros_like(landmark)
		for i, (p, q) in enumerate(landmark):
			landmark_cropped[i] = [p - x0_new, q - y0_new]
	else:
		landmark_cropped = None

	if only_img:
		return img_cropped
	if abs_coord:
		return img_cropped, landmark_cropped, (y0 - y0_new, x0 - x0_new, y1_new - y1, x1_new - x1), y0_new, y1_new, x0_new, x1_new
	else:
		return img_cropped, landmark_cropped, (y0 - y0_new, x0 - x0_new, y1_new - y1, x1_new - x1)


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
	#randomdownscale对图像进行随机降采样
	def apply(self,img,**params):
		return self.randomdownscale(img)

	def randomdownscale(self,img):
		keep_ratio=True
		keep_input_shape=True
		H,W,C=img.shape
		ratio_list=[2,4]
		r=ratio_list[np.random.randint(len(ratio_list))]
		img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_NEAREST)
		if keep_input_shape:
			img_ds=cv2.resize(img_ds,(W,H),interpolation=cv2.INTER_LINEAR)

		return img_ds