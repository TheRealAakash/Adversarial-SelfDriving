import matplotlib.pyplot as plt
from PIL import ImageFont

plt.rcParams["font.family"] = "Times New Roman"
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Dropout, Activation, BatchNormalization, MaxPooling2D
from tensorflow.keras import Input, Model
from settings import settings
# import wandb
# from wandb.keras import WandbCallback
import visualkeras


def build_model(n_out):
    model = Sequential()
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    # Fully connected layer

    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(n_out))

    # model.add(Convolution2D(10,3,3, border_mode='same'))
    # model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    img = Input(shape=settings.IMG_SHAPE)
    validity = model(img)
    model2 = Model(img, validity)
    model2.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    return model2


if __name__ == '__main__':
    font = ImageFont.truetype("times.ttf", 25)
    img = visualkeras.layered_view(build_model(43), legend=True, font=font)
    img.save('model.png')
