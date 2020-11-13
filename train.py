import cv2
import h5py
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from keras.models import Model
from keras.optimizers import Nadam
from keras.preprocessing import image
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, concatenate
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping

FINAL_DATA = "data"

class CustomGenerator(image.ImageDataGenerator):
    def __init__(self, rescale = None, horizontal_flip = False, brighten_range = 0):
        super().__init__(rescale = rescale, horizontal_flip = horizontal_flip)
        self.horizontal_flip = horizontal_flip
        self.brighten_range = brighten_range
        self.rescale = rescale

    def flow(self, x, prev_x = None, y = None, batch_size = 32, drop_percent = 0.5, roi = None):
        return CustomIterator(x = x, prev_x = prev_x, y = y, image_gen = self, batch_size = batch_size, drop_percent = drop_percent, roi = roi)

    def transform(self, x):
        flipped = False
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                    x = image.image.flip_axis(x, self.col_axis)
                    flipped = True
        if self.brighten_range != 0:
            brightness = np.random.uniform(low = 1 - self.brighten_range, high = 1 + self.brighten_range)
            img = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
            img[:, :, 2] = np.clip(img[:, :, 2] * brightness, 0, 255)
            x = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return x, flipped

class CustomIterator(image.Iterator):
    def __init__(self, x, prev_x = None, y = None, image_gen = None, batch_size = 32, drop_percent = 0.5, roi = None):
        super().__init__(x.shape[0], 32, False, None)
        self.data_format = K.image_data_format()
        self.x = x
        self.batch_size = batch_size
        self.drop_percent = drop_percent
        self.roi = roi
        self.prev_x = prev_x
        self.y = y
        self.image_gen = image_gen

    def next(self):
        with self.lock:
            id_arr = next(self.index_generator)
        return self._get_indexes(id_arr)

    def _get_indexes(self, id_arr):
        id_arr = sorted(id_arr)
        batch_x_image = np.zeros(tuple([self.batch_size] + list(self.x.shape)[1:]), dtype = K.floatx())
        batch_x_prev_x = np.zeros(tuple([self.batch_size] + list(self.prev_x.shape)[1:]), dtype = K.floatx())
        if self.roi:
            batch_x_image = batch_x_image[:, self.roi[0] : self.roi[1], self.roi[2] : self.roi[3], :]
        idx = []
        flipped = []
        for i, j in enumerate(id_arr):
            x = self.x[j]
            if self.roi:
                x = x[self.roi[0] : self.roi[1], self.roi[2] : self.roi[3], :]
            transform_gen = self.image_gen.transform(x.astype(K.floatx()))
            x = transform_gen[0]
            flipped.append(transform_gen[1])
            x = self.image_gen.standardize(x)
            batch_x_image[i] = x
            prev_x = self.prev_x[j]
            if transform_gen[1]:
                prev_x[0] *= -1.0
            batch_x_prev_x[i] = prev_x
            idx.append(j)

        xb = [np.asarray(batch_x_image), np.asarray(batch_x_prev_x)]
        idx = sorted(idx)
        yb = self.y[idx]
        idx = []
        for i in range(len(flipped)):
            if flipped[i]:
                yb[i] *= -1
            if np.isclose(yb[i], 0):
                if np.random.uniform(0, 1) > self.drop_percent:
                    idx.append(True)
                else:
                    idx.append(False)
            else:
                idx.append(False)

        yb = yb[idx]
        print(np.amax(yb))
        noise = np.random.normal(-1, 1, (len(yb), 1))
        yb += noise
        yb = np.clip(yb, -1.0, 1.0)
        xb[0] = xb[0][idx]
        xb[1] = xb[1][idx]
        return xb, yb

    def _get_batches_of_transformed_samples(self, id_arr):
        return self._get_indexes(id_arr)

train_data = h5py.File(FINAL_DATA + "/train.h5", 'r')
val_data = h5py.File(FINAL_DATA + "/val.h5", 'r')
test_data = h5py.File(FINAL_DATA + "/test.h5", 'r')

data_gen = CustomGenerator(rescale = 1./255, horizontal_flip = True, brighten_range = 0.4)
train_gen = data_gen.flow(train_data['image'], train_data['previous_state'], train_data['label'], batch_size = 32, drop_percent = 0.0, roi = [76, 135, 0, 255])
val_gen = data_gen.flow(val_data['image'], val_data['previous_state'], val_data['label'], batch_size = 32, drop_percent = 0.0, roi = [76, 135, 0, 255])

xb, yb = next(train_gen)
inp = Input(shape = xb[0].shape[1:])
conv = Conv2D(16, (3, 3), name = 'conv0', padding = 'same', activation = 'relu')(inp)
conv = MaxPooling2D(pool_size = (2, 2))(conv)
conv = Conv2D(32, (3, 3), name = 'conv1', padding = 'same', activation = 'relu')(conv)
conv = MaxPooling2D(pool_size = (2, 2))(conv)
conv = Conv2D(32, (3, 3), name = 'conv2', padding = 'same', activation = 'relu')(conv)
conv = MaxPooling2D(pool_size = (2, 2))(conv)
flatten = Flatten()(conv)
flatten = Dropout(0.2)(flatten)
state = Input(shape = xb[1].shape[1:])
flatten = concatenate([flatten, state])
dense = Dense(64, name = 'dense0', activation = 'relu')(flatten)
dense = Dropout(0.2)(dense)
dense = Dense(10, name = 'dense1', activation = 'relu')(dense)
dense = Dropout(0.2)(dense)
dense = Dense(1, name = 'output')(dense)

nadam = Nadam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
model = Model(inputs = [inp, state], outputs = dense)
model.compile(optimizer = nadam, loss = 'mse')

MODEL_DIR = "./model/"
plateau_cb = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, min_lr = 0.0001, verbose = 1)
chk_path = MODEL_DIR + "model.{0}-{1}.h5".format('{epoch:02d}', '{val_loss:.7f}')
ckpt_cb = ModelCheckpoint(chk_path, save_best_only = True, verbose = 1)
csv_cb = CSVLogger(MODEL_DIR + "train_log.csv")
early_stop_cb = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1)
callbacks = [plateau_cb, csv_cb, ckpt_cb, early_stop_cb]

history = model.fit(train_gen, steps_per_epoch = train_data['image'].shape[0] // 32, epochs = 500, callbacks = callbacks, validation_data = val_gen, validation_steps = val_data['image'].shape[0] // 32,verbose = 2)

xb, yb = next(train_gen)
preds = model.predict([xb[0], xb[1]])
for i in range(len(yb)):
    img, label = xb[0][i], yb[i]
    theta = label * 0.69
    pil_img = image.array_to_img(img, K.image_data_format(), scale = True)
    draw_img = pil_img.copy()
    img_draw = ImageDraw.Draw(draw_img)
    fp = (int(img.shape[1] / 2), img.shape[0])
    sp = (int((img.shape[1] / 2) + (50 * np.sin(theta))),int(img.shape[0] - (50 * np.cos(theta))))
    img_draw.line([fp, sp], fill = (255, 0, 0), width = 3)
    theta = preds[i] * 0.69
    sp = (int((img.shape[1] / 2) + (50 * np.sin(theta))),int(img.shape[0] - (50 * np.cos(theta))))
    img_draw.line([fp, sp], fill = (0, 0, 255), width = 3)
    del img_draw
    plt.imshow(draw_img)
    plt.show()
