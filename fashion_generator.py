from fashion_helper import *
import cv2
import os


class DataLoader(object):
    """
    A data generator for preprocessing on CPU
    """
    def __init__(self,
                 data_dir='train_fashion',
                 mode='train'):
        """
        init
        :param data_dir: str
        :param mode: str, train or test
        """
        self.curr = 0
        self.mode = mode
        if mode == 'train':
            self.img_paths, self.labels = prepare_data(data_dir=data_dir, is_shuffle=True)
        else:
            annotations_dir = os.path.join(data_dir, 'test.csv')
            self.img_paths, self.labels = prepare_test_paths(annotations_dir=annotations_dir,
                                                             test_dir=data_dir)
        self.n = len(self.img_paths)

    def generator(self, n=0):
        i = 0
        if n == 0:
            n = self.n
        while i < n:
            img_path = self.img_paths[i]
            label = self.labels[i]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # padding
            h, w, _ = img.shape
            img = cv2.copyMakeBorder(img, 0, 512 - h, 0, 512 - w, cv2.BORDER_CONSTANT)
            heatmap, num_visible = label_to_heatmap(label, img)

            img = 2 * (img / 255.0) - 1.0
            if self.mode == 'train':
                img_flip, heatmap_flip = self.flip(img, heatmap)
                yield img_path, img_flip, heatmap_flip
                img_rotate, heatmap_rotate = self.rotate(img, heatmap)
                yield img_path, img_rotate, heatmap_rotate

            yield img_path, img, heatmap
            if self.mode == 'test':
                img = img[:, ::-1, :]
                yield img_path, img, heatmap
            i += 1

    @staticmethod
    def rotate(img, heatmap):
        angle = np.random.uniform(-30, 30)
        center = (256, 256)
        rot_mat = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)
        new_img = cv2.warpAffine(img, rot_mat, (512, 512))
        new_heat = cv2.warpAffine(heatmap, rot_mat, (512, 512))

        return new_img, new_heat

    @staticmethod
    def flip(img, heatmap):
        new_img = img[:, ::-1, :]
        new_heat = flip_to_origin(heatmap)

        return new_img, new_heat
