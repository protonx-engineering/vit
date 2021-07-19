from vit.model import ViT
from tensorflow import keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    (x_train, y_train), (x_valid, y_valid) = keras.datasets.cifar10.load_data()
    x_train = x_train / 255.0
    x_valid = x_valid / 255.0
    model = ViT(num_classes=10, patch_size=2, image_size=32,
                norm_eps=1e-6, num_heads=4, D=64, mlp_dim=128, num_layers=2)

    model.build(input_shape=(32, 32, 32, 3))
    # model.summary()

    optimizer = Adam(learning_rate=0.001)
    loss = SparseCategoricalCrossentropy()
    model.compile(optimizer, loss=loss,
                  metrics=['accuracy'])

    model.fit(x=x_train,
              y=y_train,
              epochs=100,
              batch_size=32,
              validation_data = (x_valid, y_valid) 
              )
