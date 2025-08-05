'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30

The code is designed for scenarios such as disentanglement-based methods where it is necessary to ensure an equal number of positive and negative samples.
'''

import torch
import cv2
import random
import numpy as np
from training.dataset.abstract_dataset import DeepfakeAbstractBaseDataset
import albumentations as alb
from training.dataset.utils.funcs import RandomDownScale,crop_face
from PIL import Image
class pairDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        
        # Get real and fake image lists
        # Fix the label of real images to be 0 and fake images to be 1
        self.fake_imglist = [(img, label, 1) for img, label in zip(self.image_list, self.label_list) if label != 0]
        self.real_imglist = [(img, label, 0) for img, label in zip(self.image_list, self.label_list) if label == 0]
        self.img_lines = self.fake_imglist + self.real_imglist
        random.shuffle(self.img_lines)
        self.source_transforms = self.get_source_transforms()
    def __getitem__(self, index, norm=True):
        # Get the fake and real image paths and labels
        # fake_image_path, fake_spe_label, fake_label = self.fake_imglist[index]
        # real_index = random.randint(0, len(self.real_imglist) - 1)  # Randomly select a real image
        # real_image_path, real_spe_label, real_label = self.real_imglist[real_index]

        real_image_path, real_spe_label, real_label = self.img_lines[index]
        fake_label = real_label
        fake_spe_label = real_spe_label
        # Get the mask and landmark paths for fake and real images
        # fake_mask_path = fake_image_path.replace('frames', 'masks')
        # fake_landmark_path = fake_image_path.replace('frames', 'landmarks').replace('.png', '.npy')
        
        real_mask_path = real_image_path.replace('frames', 'masks')
        real_landmark_path = real_image_path.replace('frames', 'landmarks').replace('.png', '.npy')
        # Load the fake and real images
        # fake_image = self.load_rgb(fake_image_path)
        real_image = self.load_rgb(real_image_path)

        # fake_image = np.array(fake_image)  # Convert to numpy array for data augmentation
        real_image = np.array(real_image)  # Convert to numpy array for data augmentation
        fake_image = real_image

        # Load mask and landmark (if needed) for fake and real images
        if self.config['with_mask']:
            # fake_mask = self.load_mask(fake_mask_path)
            real_mask = self.load_mask(real_mask_path)
        else:
            fake_mask, real_mask = None, None

        if self.config['with_landmark']:
            # fake_landmarks = self.load_landmark(fake_landmark_path)
            real_landmarks = self.load_landmark(real_landmark_path)
            fake_landmarks = real_landmarks
        else:
            fake_landmarks, real_landmarks = None, None
        #产生 real_image ,fake_image
        size = self.config['resolution']
        real_image,fake_image,fake_mask = self.generate_fake(real_image, real_landmarks)

        # real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
        # real_image = cv2.resize(real_image, (size, size), interpolation=cv2.INTER_CUBIC)
        # fake_image = cv2.cvtColor(fake_image, cv2.COLOR_BGR2RGB)
        # fake_image = cv2.resize(fake_image, (size, size), interpolation=cv2.INTER_CUBIC)
        # Image.fromarray(np.array(real_image, dtype=np.uint8))
        # Image.fromarray(np.array(fake_image, dtype=np.uint8))
        # real_image = np.array(real_image)
        # fake_image = np.array(fake_image)

        fake_mask = cv2.resize(fake_mask, (size, size)) / 252
        fake_mask = np.expand_dims(fake_mask, axis=2)
        # Do transforms for fake and real images

        fake_image_trans, fake_landmarks_trans, fake_mask_trans = self.data_aug(fake_image, fake_landmarks, fake_mask)
        real_image_trans, real_landmarks_trans, real_mask_trans = self.data_aug(real_image, real_landmarks, real_mask)

        if not norm:
            return {"fake": (fake_image_trans, fake_label), 
                    "real": (real_image_trans, real_label)}

        # To tensor and normalize for fake and real images
        fake_image_trans = self.normalize(self.to_tensor(fake_image_trans))
        real_image_trans = self.normalize(self.to_tensor(real_image_trans))

        # Convert landmarks and masks to tensors if they exist
        if self.config['with_landmark']:
            fake_landmarks_trans = torch.from_numpy(fake_landmarks_trans)
            real_landmarks_trans = torch.from_numpy(real_landmarks_trans)
        if self.config['with_mask']:
            fake_mask_trans = torch.from_numpy(fake_mask_trans)
            real_mask_trans = torch.from_numpy(real_mask_trans)

        return {"fake": (fake_image_trans, fake_label, fake_spe_label, fake_landmarks_trans, fake_mask_trans), 
                "real": (real_image_trans, real_label, real_spe_label, real_landmarks_trans, real_mask_trans)}

    def __len__(self):
        return len(self.fake_imglist)

    def get_source_transforms(self):
        return alb.Compose([
            alb.Compose([
                alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                       val_shift_limit=(-0.3, 0.3), p=1),
                alb.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1),
            ], p=1),

            alb.OneOf([
                RandomDownScale(p=1),
                alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ], p=1),

        ], p=1.)
    def generate_fake(self, real_image, real_landmarks):
        if self.config['mode'] == 'train':
            if np.random.random()<0.5:
                real_image,real_1_landmarks = self.hflip(real_image,real_landmarks)
            img_target,img_source,mask_source = self.generate_source_taget(real_image.copy(),real_landmarks.copy())
        return img_target,img_source,mask_source
    def hflip(self, img, landmark):
        H, W = img.shape[:2]
        landmark = landmark.copy()
        if landmark is not None:
            landmark_new = np.zeros_like(landmark)

            landmark_new[:17] = landmark[:17][:, ::-1]
            landmark_new[17:27] = landmark[17:27][:, ::-1]
            landmark_new[27:31] = landmark[27:31][:, ::-1]
            landmark_new[31:36] = landmark[31:36][::-1]

            landmark_new[36:40] = landmark[42:46][::-1]
            landmark_new[40:42] = landmark[46:48][::-1]

            landmark_new[42:46] = landmark[36:40][::-1]
            landmark_new[46:48] = landmark[40:42][::-1]

            landmark_new[48:55] = landmark[48:55][::-1]
            landmark_new[55:60] = landmark[55:60][::-1]

            landmark_new[60:65] = landmark[60:65][::-1]
            landmark_new[65:68] = landmark[65:68][::-1]
            if len(landmark) == 68:
                pass
            elif len(landmark) == 81:
                landmark_new[68:81] = landmark[68:81][::-1]
            else:
                raise NotImplementedError
            landmark_new[:, 0] = W - landmark_new[:, 0]
        else:
            landmark_new = None

        img = img[:, ::-1].copy()
        # landmark = landmark.astype(np.int64)
        return img, landmark_new

    def randaffine(self, img, mask):
        f = alb.Affine(
            translate_percent={'x': (-0.03, 0.03), 'y': (-0.015, 0.015)},
            scale=[0.95, 1 / 0.95],
            fit_output=False,
            p=1)

        g = alb.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )

        transformed = f(image=img, mask=mask)
        img = transformed['image']

        mask = transformed['mask']
        transformed = g(image=img, mask=mask)
        mask = transformed['mask']
        return img, mask
    def generate_source_taget(self,img, landmark):
        H, W = img.shape[:2]
        if np.random.rand() < 0.25:
            landmark = landmark[:81]
        mask = np.zeros_like(img[:, :, 0])
        landmark = landmark.astype(np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(landmark),1.)
        source = img.copy()
        if np.random.random() < 0.5:
            source = self.source_transforms(image=source.astype(np.uint8))['image']
        else:
            img = self.source_transforms(image=img.astype(np.uint8))['image']
        # 对源图像进行仿射和弹性变换
        if np.random.rand() < 0.25:
            source, mask = self.randaffine(source, mask)
        return img, source, mask

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                        the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors for fake and real data
        fake_images, fake_labels, fake_spe_labels, fake_landmarks, fake_masks = zip(*[data["fake"] for data in batch])
        real_images, real_labels, real_spe_labels, real_landmarks, real_masks = zip(*[data["real"] for data in batch])

        # Stack the image, label, landmark, and mask tensors for fake and real data
        fake_images = torch.stack(fake_images, dim=0)
        fake_labels = torch.LongTensor(fake_labels)
        fake_spe_labels = torch.LongTensor(fake_spe_labels)
        real_images = torch.stack(real_images, dim=0)
        real_labels = torch.LongTensor(real_labels)
        real_spe_labels = torch.LongTensor(real_spe_labels)

        # Special case for landmarks and masks if they are None
        if fake_landmarks[0] is not None:
            fake_landmarks = torch.stack(fake_landmarks, dim=0)
        else:
            fake_landmarks = None
        if real_landmarks[0] is not None:
            real_landmarks = torch.stack(real_landmarks, dim=0)
        else:
            real_landmarks = None

        if fake_masks[0] is not None:
            fake_masks = torch.stack(fake_masks, dim=0)
        else:
            fake_masks = None
        if real_masks[0] is not None:
            real_masks = torch.stack(real_masks, dim=0)
        else:
            real_masks = None

        # Combine the fake and real tensors and create a dictionary of the tensors
        images = torch.cat([real_images, fake_images], dim=0)
        labels = torch.cat([real_labels, fake_labels], dim=0)
        spe_labels = torch.cat([real_spe_labels, fake_spe_labels], dim=0)
        
        if fake_landmarks is not None and real_landmarks is not None:
            landmarks = torch.cat([real_landmarks, fake_landmarks], dim=0)
        else:
            landmarks = None

        if fake_masks is not None and real_masks is not None:
            masks = torch.cat([real_masks, fake_masks], dim=0)
        else:
            masks = None

        data_dict = {
            'image': images,
            'label': labels,
            'label_spe': spe_labels,
            'landmark': landmarks,
            'mask': masks
        }
        return data_dict

