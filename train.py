from vit.model import ViTBase
import tensorflow_datasets as tfds
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    train_ds = tfds.load("cats_vs_dogs",
                         split='train[:80%]',
                         batch_size=64,
                         shuffle_files=True,
                         as_supervised=True)
    val_ds = tfds.load("cats_vs_dogs",
                       split='train[80%:]',
                       batch_size=64,
                       shuffle_files=True,
                       as_supervised=True)

    model = ViTBase()
    model.build(input_shape=(64, 224, 224, 3))
    model.summary()

    optimizer = Adam(learning_rate=0.01)
    loss = SparseCategoricalCrossentropy()
    model.compile(optimizer, loss=loss,
                  metrics=['accuracy'])

    model.fit(train_ds,
              epochs=10,
              batch_size=64,
              validation_data=val_ds)
