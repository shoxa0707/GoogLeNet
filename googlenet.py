import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import concatenate

class GoogLeNet():
    def __init__(self, include_top=True):
        self.include_top = include_top
        # Pools and Norms
        self.maxpool2 = keras.layers.MaxPooling2D(2, name='maxpool2')
        self.localrespnorm = keras.layers.Lambda(tf.nn.local_response_normalization, name='local_resp_norm')
        self.incepmax = keras.layers.MaxPooling2D(pool_size=(3,3), strides=1, padding='same', name='inception_maxpool')

        self.avpool1 = keras.layers.AveragePooling2D(pool_size=(5,5), padding='valid', strides=3, name='average_pool1')
        self.soft1conv1 = keras.layers.Conv2D(filters=1024, kernel_size=(1,1), strides=2, activation='relu', padding='same', name='soft1_conv1')
        self.fc1_1 = keras.layers.Dense(1024, activation='relu', name='fc1_1')
        self.fc1_2 = keras.layers.Dense(1000, activation='softmax', name='fc1_2')

        self.avpool2 = keras.layers.AveragePooling2D(pool_size=(5,5), padding='valid', strides=3, name='average_pool2')
        self.soft2conv1 = keras.layers.Conv2D(filters=1024, kernel_size=(1,1), strides=2, activation='relu', padding='same', name='soft2_conv1')
        self.fc2_1 = keras.layers.Dense(1024, activation='relu', name='fc2_1')
        self.fc2_2 = keras.layers.Dense(1000, activation='softmax', name='fc2_2')

        self.avpool3 = keras.layers.AveragePooling2D(pool_size=(7,7), padding='valid', strides=1, name='average_pool3')
        self.dropout = keras.layers.Dropout(0.4)
        self.fc3 = keras.layers.Dense(1000, activation='softmax', name='fc3')

        # BLOCK1
        self.conv1 = keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=2, activation='relu', padding='same', name='conv1')
        # BLOCK2
        self.conv2_1 = keras.layers.Conv2D(filters=192, kernel_size=(1,1), activation='relu', padding='same', name='conv2_1')
        self.conv2_2 = keras.layers.Conv2D(filters=192, kernel_size=(3,3), activation='relu', padding='same', name='conv2_2')
        # BLOCK3 - Inception(3a)
        self.conv3a_11 = keras.layers.Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='same', name='conv3a_11')
        self.conv3a_21 = keras.layers.Conv2D(filters=96, kernel_size=(1,1), activation='relu', padding='same', name='conv3a_21')
        self.conv3a_31 = keras.layers.Conv2D(filters=16, kernel_size=(1,1), activation='relu', padding='same', name='conv3a_31')

        self.conv3a_22 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='conv3a_22')
        self.conv3a_32 = keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu', padding='same', name='conv3a_32')
        self.conv3a_42 = keras.layers.Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same', name='conv3a_42')
        # BLOCK4 - Inception(3b)
        self.conv3b_11 = keras.layers.Conv2D(filters=128, kernel_size=(1,1), activation='relu', padding='same', name='conv3b_11')
        self.conv3b_21 = keras.layers.Conv2D(filters=128, kernel_size=(1,1), activation='relu', padding='same', name='conv3b_21')
        self.conv3b_31 = keras.layers.Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same', name='conv3b_31')

        self.conv3b_22 = keras.layers.Conv2D(filters=192, kernel_size=(3,3), activation='relu', padding='same', name='conv3b_22')
        self.conv3b_32 = keras.layers.Conv2D(filters=96, kernel_size=(5,5), activation='relu', padding='same', name='conv3b_32')
        self.conv3b_42 = keras.layers.Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='same', name='conv3b_42')
        # BLOCK5 - Inception(4a)
        self.conv4a_11 = keras.layers.Conv2D(filters=192, kernel_size=(1,1), activation='relu', padding='same', name='conv4a_11')
        self.conv4a_21 = keras.layers.Conv2D(filters=128, kernel_size=(1,1), activation='relu', padding='same', name='conv4a_21')
        self.conv4a_31 = keras.layers.Conv2D(filters=16, kernel_size=(1,1), activation='relu', padding='same', name='conv4a_31')

        self.conv4a_22 = keras.layers.Conv2D(filters=208, kernel_size=(3,3), activation='relu', padding='same', name='conv4a_22')
        self.conv4a_32 = keras.layers.Conv2D(filters=48, kernel_size=(5,5), activation='relu', padding='same', name='conv4a_32')
        self.conv4a_42 = keras.layers.Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='same', name='conv4a_42')
        # BLOCK6 - Inception(4b)
        self.conv4b_11 = keras.layers.Conv2D(filters=160, kernel_size=(1,1), activation='relu', padding='same', name='conv4b_11')
        self.conv4b_21 = keras.layers.Conv2D(filters=112, kernel_size=(1,1), activation='relu', padding='same', name='conv4b_21')
        self.conv4b_31 = keras.layers.Conv2D(filters=24, kernel_size=(1,1), activation='relu', padding='same', name='conv4b_31')

        self.conv4b_22 = keras.layers.Conv2D(filters=224, kernel_size=(3,3), activation='relu', padding='same', name='conv4b_22')
        self.conv4b_32 = keras.layers.Conv2D(filters=64, kernel_size=(5,5), activation='relu', padding='same', name='conv4b_32')
        self.conv4b_42 = keras.layers.Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='same', name='conv4b_42')
        # BLOCK7 - Inception(4c)
        self.conv4c_11 = keras.layers.Conv2D(filters=128, kernel_size=(1,1), activation='relu', padding='same', name='conv4c_11')
        self.conv4c_21 = keras.layers.Conv2D(filters=128, kernel_size=(1,1), activation='relu', padding='same', name='conv4c_21')
        self.conv4c_31 = keras.layers.Conv2D(filters=24, kernel_size=(1,1), activation='relu', padding='same', name='conv4c_31')

        self.conv4c_22 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='conv4c_22')
        self.conv4c_32 = keras.layers.Conv2D(filters=64, kernel_size=(5,5), activation='relu', padding='same', name='conv4c_32')
        self.conv4c_42 = keras.layers.Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='same', name='conv4c_42')
        # BLOCK8 - Inception(4d)
        self.conv4d_11 = keras.layers.Conv2D(filters=112, kernel_size=(1,1), activation='relu', padding='same', name='conv4d_11')
        self.conv4d_21 = keras.layers.Conv2D(filters=144, kernel_size=(1,1), activation='relu', padding='same', name='conv4d_21')
        self.conv4d_31 = keras.layers.Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same', name='conv4d_31')

        self.conv4d_22 = keras.layers.Conv2D(filters=288, kernel_size=(3,3), activation='relu', padding='same', name='conv4d_22')
        self.conv4d_32 = keras.layers.Conv2D(filters=64, kernel_size=(5,5), activation='relu', padding='same', name='conv4d_32')
        self.conv4d_42 = keras.layers.Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='same', name='conv4d_42')
        # BLOCK9 - Inception(4e)
        self.conv4e_11 = keras.layers.Conv2D(filters=256, kernel_size=(1,1), activation='relu', padding='same', name='conv4e_11')
        self.conv4e_21 = keras.layers.Conv2D(filters=160, kernel_size=(1,1), activation='relu', padding='same', name='conv4e_21')
        self.conv4e_31 = keras.layers.Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same', name='conv4e_31')

        self.conv4e_22 = keras.layers.Conv2D(filters=320, kernel_size=(3,3), activation='relu', padding='same', name='conv4e_22')
        self.conv4e_32 = keras.layers.Conv2D(filters=128, kernel_size=(5,5), activation='relu', padding='same', name='conv4e_32')
        self.conv4e_42 = keras.layers.Conv2D(filters=128, kernel_size=(1,1), activation='relu', padding='same', name='conv4e_42')
        # BLOCK10 - Inception(5a)
        self.conv5a_11 = keras.layers.Conv2D(filters=256, kernel_size=(1,1), activation='relu', padding='same', name='conv5a_11')
        self.conv5a_21 = keras.layers.Conv2D(filters=160, kernel_size=(1,1), activation='relu', padding='same', name='conv5a_21')
        self.conv5a_31 = keras.layers.Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same', name='conv5a_31')

        self.conv5a_22 = keras.layers.Conv2D(filters=320, kernel_size=(3,3), activation='relu', padding='same', name='conv5a_22')
        self.conv5a_32 = keras.layers.Conv2D(filters=128, kernel_size=(5,5), activation='relu', padding='same', name='conv5a_32')
        self.conv5a_42 = keras.layers.Conv2D(filters=128, kernel_size=(1,1), activation='relu', padding='same', name='conv5a_42')
        # BLOCK11 - Inception(5b)
        self.conv5b_11 = keras.layers.Conv2D(filters=384, kernel_size=(1,1), activation='relu', padding='same', name='conv5b_11')
        self.conv5b_21 = keras.layers.Conv2D(filters=192, kernel_size=(1,1), activation='relu', padding='same', name='conv5b_21')
        self.conv5b_31 = keras.layers.Conv2D(filters=48, kernel_size=(1,1), activation='relu', padding='same', name='conv5b_31')

        self.conv5b_22 = keras.layers.Conv2D(filters=384, kernel_size=(3,3), activation='relu', padding='same', name='conv5b_22')
        self.conv5b_32 = keras.layers.Conv2D(filters=128, kernel_size=(5,5), activation='relu', padding='same', name='conv5b_32')
        self.conv5b_42 = keras.layers.Conv2D(filters=128, kernel_size=(1,1), activation='relu', padding='same', name='conv5b_42')

    def __call__(self, x):
        # BLOCK1
        x = self.conv1(x)
        x = self.maxpool2(x)
        x = self.localrespnorm(x)
        # BLOCK2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.localrespnorm(x)
        x = self.maxpool2(x)
        # BLOCK3 - Inception(3a)
        x_input = x
        x1 = self.conv3a_11(x_input)
        x21 = self.conv3a_21(x_input)
        x31 = self.conv3a_31(x_input)
        x_max41 = self.incepmax(x_input)
        x22 = self.conv3a_22(x21)
        x32 = self.conv3a_32(x31)
        x42 = self.conv3a_42(x_max41)

        x = concatenate([x1, x22, x32, x42], axis=-1)
        # BLOCK4 - Inception(3b)
        x_input = x
        x1 = self.conv3b_11(x_input)
        x21 = self.conv3b_21(x_input)
        x31 = self.conv3b_31(x_input)
        x_max41 = self.incepmax(x_input)
        x22 = self.conv3b_22(x21)
        x32 = self.conv3b_32(x31)
        x42 = self.conv3b_42(x_max41)

        x = concatenate([x1, x22, x32, x42], axis=-1)
        x = self.maxpool2(x)
        # BLOCK5 - Inception(4a)
        x_input = x
        x1 = self.conv4a_11(x_input)
        x21 = self.conv4a_21(x_input)
        x31 = self.conv4a_31(x_input)
        x_max41 = self.incepmax(x_input)
        x22 = self.conv4a_22(x21)
        x32 = self.conv4a_32(x31)
        x42 = self.conv4a_42(x_max41)

        x = concatenate([x1, x22, x32, x42], axis=-1)

        if self.include_top:
            # Softmax part1
            soft1 = self.avpool1(x)
            soft1 = self.soft1conv1(x)
            soft1 = self.fc1_1(x)
            soft1 = self.fc1_2(x)

        # BLOCK6 - Inception(4b)
        x_input = x
        x1 = self.conv4b_11(x_input)
        x21 = self.conv4b_21(x_input)
        x31 = self.conv4b_31(x_input)
        x_max41 = self.incepmax(x_input)
        x22 = self.conv4b_22(x21)
        x32 = self.conv4b_32(x31)
        x42 = self.conv4b_42(x_max41)

        x = concatenate([x1, x22, x32, x42], axis=-1)
        # BLOCK7 - Inception(4c)
        x_input = x
        x1 = self.conv4c_11(x_input)
        x21 = self.conv4c_21(x_input)
        x31 = self.conv4c_31(x_input)
        x_max41 = self.incepmax(x_input)
        x22 = self.conv4c_22(x21)
        x32 = self.conv4c_32(x31)
        x42 = self.conv4c_42(x_max41)

        x = concatenate([x1, x22, x32, x42], axis=-1)
        # BLOCK8 - Inception(4d)
        x_input = x
        x1 = self.conv4d_11(x_input)
        x21 = self.conv4d_21(x_input)
        x31 = self.conv4d_31(x_input)
        x_max41 = self.incepmax(x_input)
        x22 = self.conv4d_22(x21)
        x32 = self.conv4d_32(x31)
        x42 = self.conv4d_42(x_max41)

        x = concatenate([x1, x22, x32, x42], axis=-1)

        if self.include_top:
            # Softmax part2
            soft2 = self.avpool2(x)
            soft2 = self.soft2conv1(x)
            soft2 = self.fc2_1(x)
            soft2 = self.fc2_2(x)

        # BLOCK9 - Inception(4e)
        x_input = x
        x1 = self.conv4e_11(x_input)
        x21 = self.conv4e_21(x_input)
        x31 = self.conv4e_31(x_input)
        x_max41 = self.incepmax(x_input)
        x22 = self.conv4e_22(x21)
        x32 = self.conv4e_32(x31)
        x42 = self.conv4e_42(x_max41)

        x = concatenate([x1, x22, x32, x42], axis=-1)
        x = self.maxpool2(x)
        # BLOCK10 - Inception(5a)
        x_input = x
        x1 = self.conv5a_11(x_input)
        x21 = self.conv5a_21(x_input)
        x31 = self.conv5a_31(x_input)
        x_max41 = self.incepmax(x_input)
        x22 = self.conv5a_22(x21)
        x32 = self.conv5a_32(x31)
        x42 = self.conv5a_42(x_max41)

        x = concatenate([x1, x22, x32, x42], axis=-1)
        # BLOCK11 - Inception(5b)
        x_input = x
        x1 = self.conv5b_11(x_input)
        x21 = self.conv5b_21(x_input)
        x31 = self.conv5b_31(x_input)
        x_max41 = self.incepmax(x_input)
        x22 = self.conv5b_22(x21)
        x32 = self.conv5b_32(x31)
        x42 = self.conv5b_42(x_max41)

        # Softmax part end
        x = concatenate([x1, x22, x32, x42], axis=-1)
        if self.include_top:
            x = self.avpool3(x)
            x = self.dropout(x)
            x = self.fc3(x)

        return x
    