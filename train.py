from vit.model import ViT
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    model = ViT()
    model.build(input_shape=(8, 224, 224, 3))
    model.summary()

    images = np.random.rand(8, 224, 224, 3)
    output = model(images)
    print(output.shape)
