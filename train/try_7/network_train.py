# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.io as sio
from include import loss_functions
import keras.layers as layers
from tensorflow.keras.optimizers import Adam, SGD
from keras.models import Model
from train.try_7.encoder import Encoder
from train.try_7.decoder import Decoder
from keras.datasets import cifar10
import os
from tqdm import tqdm
from train.try_7.discriminator import Discriminator



devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

keras = tf.keras
# layers = keras.layers
K = keras.backend

# input image dimensions
img_rows, img_cols = 32, 32
block_size = 8
offset = 0 # To be able to continue training
steps = 10000 #int(np.ceil(60000 / batch_size))
w_rows = int((img_rows) / block_size)
w_cols = int((img_cols) / block_size)
ssim_win_size = 8


def scalar_output_shape(input_shape):
    return input_shape

def UniformNoise(x, val):
    noise = tf.random.uniform(shape=[64, 4, 4, 64], minval=-val, maxval=val, dtype=tf.float32, seed=None)
    return x + noise


def slice_img(x):
    output_container = x[:, :, :, :64]
    return output_container

def slice_noise(x):
    output_noise = x[:, :, :, :64]
    noise = tf.random.normal(output_noise.shape)
    return noise

def slice_secret(x):
    x_part1, x_part2 = tf.split(x, [64, 64], axis=-1)
    return x_part2



class AEEQ_DN():
    def __init__(self, batch_size, epochs):
        # input image dimensions
        self.epochs = epochs
        self.img_rows = 32
        self.img_cols = 32
        self.block_size = 8
        self.batch_size = batch_size
        self.w_rows = int(self.img_rows / self.block_size)
        self.w_cols = int(self.img_cols / self.block_size)
        self.trainable_transform = False
        self.num_of_filters = self.block_size ** 2
        self.use_circular = True
        self.input_img = layers.Input(shape=(self.img_rows, self.img_cols, 1), batch_size=self.batch_size, name='input_img')
        self.input_watermark = layers.Input(shape=(self.w_rows, self.w_cols, 1), batch_size=self.batch_size, name='input_watermark')
        self.dct_layer = layers.Conv2D(self.num_of_filters, (1, 1), activation='linear', padding='same', use_bias=False, name='dct', trainable=False)
        self.idct_layer = layers.Conv2D(self.num_of_filters, (1, 1), activation='linear', padding='same', use_bias=False, name='idct',trainable=False)
        self.encoder = Encoder()
        self.decoder = Decoder()

        optimizer = Adam(learning_rate=1e-4)

        # Build and compile the discriminator
        self.discriminator = Discriminator().discriminator
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])






    def build_encoder(self, x):
        input_img, input_watermark = x
        rearranged_img = l1 = layers.Lambda(tf.nn.space_to_depth, arguments={'block_size': self.block_size},
                                            name='rearrange_img')(input_img)
        dct_layer_img = self.dct_layer(rearranged_img)
        encoder_model = self.encoder([dct_layer_img, input_watermark])
        encoder_model = self.idct_layer(encoder_model)
        encoder_model = layers.Add(name='residual_add')([encoder_model, l1])
        x = layers.Lambda(tf.nn.depth_to_space, arguments={'block_size': self.block_size},
                          name='enc_output_depth2space')(encoder_model)
        # result
        return x, encoder_model


    def build_attack(self, x):
        Q = 10
        jpeg_noise = 0.55
        q_mtx = sio.loadmat('C:/code/PROJECT/ReDMark-2.15/transforms/jpeg_qm.mat')['qm']
        q_mtx = q_mtx.astype('float32')
        if (Q < 50):
            S = 5000 / Q
        else:
            S = 200 - 2 * Q
        q_mtx = np.floor((S * q_mtx + 50.0) / 100.0)
        q_mtx = np.reshape(q_mtx, (64, 1)) * 1.0
        q_mtx = np.repeat(q_mtx[np.newaxis, ...], 4, axis=0)
        q_mtx = np.repeat(q_mtx[np.newaxis, ...], 4, axis=0)
        q_mtx = np.squeeze(q_mtx)
        q_mtx[q_mtx == 0] = 1

        #####################  Jpeg_attake   ############################
        jpeg_attaked = self.dct_layer(x)
        jpeg_attaked = layers.Lambda(lambda x: (x * 255) / q_mtx, output_shape=scalar_output_shape, name='jpg1')(
            jpeg_attaked)
        jpeg_attaked = layers.Lambda(UniformNoise, arguments={'val': jpeg_noise})(jpeg_attaked)
        jpeg_attaked = layers.Lambda(lambda x: (x / 255) * q_mtx, output_shape=scalar_output_shape, name='jpg2')(
            jpeg_attaked)
        jpeg_attaked = self.idct_layer(jpeg_attaked)

        #####################  Add GaussianNoise   ############################
        rounding_noise = layers.GaussianNoise(stddev=0.003, name='rounding_noise')(jpeg_attaked)

        return rounding_noise


    def build_decoder(self, x):
        rounding_noise = x
        decoder_model = self.dct_layer(rounding_noise)
        decoder_model = self.decoder(decoder_model)
        decoder_model = layers.Conv2D(filters=1, kernel_size=(1, 1), dilation_rate=1, activation='sigmoid', padding='same', name='dec_output_depth2space')(decoder_model)
        return decoder_model


    def build_model(self):
        output_img, output_encoder = self.build_encoder([self.input_img, self.input_watermark])
        output_attack = self.build_attack(output_encoder)
        output_decoder = self.build_decoder(output_attack)
        validity = self.discriminator(output_img)
        model = Model(inputs=[self.input_img, self.input_watermark], outputs=[output_img, output_decoder, validity])



        # Set weights
        dct_mtx = sio.loadmat('C:/code/PROJECT/ReDMark-2.15/transforms/DCT_coef.mat')['DCT_coef']
        dct_mtx = np.reshape(dct_mtx, [1, 1, self.num_of_filters, self.num_of_filters])
        model.get_layer('dct').set_weights(np.array([dct_mtx]))

        idct_mtx = sio.loadmat('C:/code/PROJECT/ReDMark-2.15/transforms/IDCT_coef.mat')['IDCT_coef']
        idct_mtx = np.reshape(idct_mtx, [1, 1, self.num_of_filters, self.num_of_filters])
        model.get_layer('idct').set_weights(np.array([idct_mtx]))


        # Define loss
        ssim_win_size = 8
        loss_object = loss_functions.SSIM_MSE_LOSS(ssim_relative_loss=1.0, mse_relative_loss=0.0,
                                                   ssim_win_size=ssim_win_size)
        ssimmse_loss = loss_object.ssimmse_loss

        lr = 1e-4
        enc_output_weight = 1.0
        dec_output_weight = 1.0
        discriminator_weight = 0.0001

        model.compile(loss={'enc_output_depth2space': ssimmse_loss,
                            'dec_output_depth2space': 'binary_crossentropy',
                            'model': 'binary_crossentropy'
                            },
                      loss_weights={'enc_output_depth2space': enc_output_weight,
                                    'dec_output_depth2space': dec_output_weight,
                                    'model': discriminator_weight
                                    },
                      optimizer=SGD(learning_rate=lr, momentum=0.98))

        return model

    def get_data(self):
        combine_cifar_pascal = True
        selected_dataset = 'cifar'  # cifar or pascal

        # the data, split between train and test sets
        print('Loading dataset...')
        # (x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = datasets.cifar10.load_data()
        (x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = cifar10.load_data()
        x_train_cifar = x_train_cifar.reshape(x_train_cifar.shape[0], img_rows, img_cols, 3)
        x_train_cifar = x_train_cifar[:, :, :, 1]
        x_train_cifar = x_train_cifar.reshape(x_train_cifar.shape[0], img_rows, img_cols, 1)
        x_test_cifar = x_test_cifar.reshape(x_test_cifar.shape[0], img_rows, img_cols, 3)
        x_test_cifar = x_test_cifar[:, :, :, 1]
        x_test_cifar = x_test_cifar.reshape(x_test_cifar.shape[0], img_rows, img_cols, 1)

        x_train_pascal = sio.loadmat('C:/code/PROJECT/ReDMark-2.15/images/pascal/pascal_resampled.mat')['patches']
        x_train_pascal = x_train_pascal[..., np.newaxis]

        # Combine
        if combine_cifar_pascal == False:
            x_train = x_train_cifar if selected_dataset == 'cifar' else x_train_pascal
        else:
            x_train = np.concatenate([x_train_cifar, x_train_pascal], axis=0)

        x_train = x_train.astype('float32')
        x_train = (x_train - 128.0) / 255.0
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')

        return x_train

    def train_epoch(self):
        exp_id = 'new_model_no_discriminator'

        if os.path.exists('C:/code/PROJECT/ReDMark-2.15/logs/{}'.format(exp_id)) == False:
            os.mkdir('C:/code/PROJECT/ReDMark-2.15/logs/{}'.format(exp_id))
            os.mkdir('C:/code/PROJECT/ReDMark-2.15/logs/{}/Weights'.format(exp_id))

        log_dir = 'C:/code/PROJECT/ReDMark-2.15/logs/{}'.format(exp_id)
        tf_logger = tf.summary.create_file_writer(log_dir)

        data = self.get_data()
        model = self.build_model()

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))


        for e in range(self.epochs):
            print('Epochs {}...'.format(e + 1))
            loss_w = []
            loss_I = []
            loss_D = []
            for step in tqdm(range(steps)):
                img_idx = np.random.randint(0, data.shape[0], self.batch_size)
                I = data[img_idx, :, :, :]

                W = np.random.randint(low=0, high=2, size=(self.batch_size, w_rows, w_cols, 1)).astype(np.float32)

                encoder_output = I
                decoder_output = W

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Generate a batch of new images
                model_output = model(inputs=[I, W], training=False)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(I, valid)
                d_loss_fake = self.discriminator.train_on_batch(model_output[0], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train network
                # ---------------------

                model.train_on_batch(x=[I, W], y=[encoder_output, decoder_output, valid])

                model_output = model(inputs=[I, W], training=False)

                loss_I.append(tf.reduce_mean(tf.square(model_output[0] - encoder_output)))
                loss_w.append(tf.reduce_mean(tf.square(model_output[1] - decoder_output)))
                loss_D.append(d_loss[0])

            mean_error_w = np.mean(loss_w)
            mean_error_I = np.mean(loss_I)
            mean_error_D = np.mean(loss_D)
            psnr = 10 * np.log10(1 ** 2 / mean_error_I)

            print('\tI Error = {} And W Error = {}'.format(mean_error_I, mean_error_w))
            print('\tD Error = {}'.format(mean_error_D))
            print('PSNR is: ', psnr)
            with tf_logger.as_default():
                tf.summary.scalar('W_MSE', mean_error_w, e + 1)
                tf.summary.scalar('I_MSE', mean_error_I, e + 1)
                tf.summary.scalar('PSNR', psnr, e + 1)

                if (e + 1) % 10 == 0:
                    model.save_weights(
                        'C:/code/PROJECT/ReDMark-2.15/logs/{}/Weights/weights_{}.h5'.format(exp_id, e + 1 + offset))

            model.save_weights('C:/code/PROJECT/ReDMark-2.15/logs/{}/Weights/weights_final.h5'.format(exp_id))

if __name__ == '__main__':
    AEEQ_DN_model = AEEQ_DN(batch_size=64, epochs=150)
    AEEQ_DN_model.train_epoch()









