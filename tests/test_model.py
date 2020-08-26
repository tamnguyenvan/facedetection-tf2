import context

import tensorflow as tf
from model import model_factory
from config import cfg

devices = tf.config.experimental.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)

model = model_factory.create_model(cfg)
model.summary()

x = tf.random.normal((1, 320, 320, 3))
outputs = model(x)
for o in outputs:
    print(o.shape)
