from collections import defaultdict

import visualkeras
from PIL import ImageFont
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, SeparableConv2D
from tensorflow.keras.optimizers import Adam

from settings import main_settings
from tensorflow_addons.layers import InstanceNormalization
settings, configs = main_settings.get_settings()

font = ImageFont.truetype("times.ttf", 20)  # using comic sans is strictly prohibited!
def buildModel(input_shape, action_space):
    # X_input = Input(input_shape)
    # model = Flatten()(model)
    # model = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(model)
    base_model = Xception(weights=None, include_top=False, input_shape=input_shape)
    model = base_model.output
    model = GlobalAveragePooling2D()(model)
    predictions = Dense(action_space, activation="linear")(model)
    # output = Dense(action_space, activation="tanh")(model)

    actor = Model(inputs=base_model.input, outputs=predictions)
    actor.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
    return actor


color_map = defaultdict(dict)
# color_map[Dropout]['fill'] = 'grey'

model = buildModel((256, 256, 3), 4)
visualkeras.layered_view(model, scale_xy=1, scale_z=1, max_z=1000)
img = visualkeras.layered_view(model, color_map=color_map, legend=True, font=font)
# img.show()
img.save("test.png")
