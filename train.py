from vit.model import ViT
from tensorflow import keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    (x_train, y_train), _ = keras.datasets.cifar100.load_data()
    x_train = x_train/255

    model = ViT(num_classes=100, patch_size=6, image_size=72,
                norm_eps=1e-6, num_heads=4, hidden_dim=64, mlp_dim=1024)

    model.build(input_shape=(64, 32, 32, 3))
    model.summary()

    optimizer = Adam(learning_rate=0.001)
    loss = SparseCategoricalCrossentropy()
    model.compile(optimizer, loss=loss,
                  metrics=['accuracy'])

    model.fit(x=x_train,
              y=y_train,
              epochs=100,
              batch_size=64,
              validation_split=0.1)
