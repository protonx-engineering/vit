from tensorflow.keras.layers import Layer, Embedding, Dense
import tensorflow as tf


class Patches(Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        dim = patches.shape[-1]

        patches = tf.reshape(patches, (batch_size, -1, dim))
        return patches


class PatchEmbedding(Layer):
    def __init__(self, patch_size, image_size, projection_dim):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2

        self.patches = Patches(patch_size)
        self.projection = Dense(units=projection_dim)

        self.position_embedding = Embedding(
            input_dim=self.num_patches,
            output_dim=projection_dim
        )

    def call(self, images):
        e_pos = tf.range(start=0, limit=self.num_patches, delta=1)
        embedded = self.position_embedding(e_pos)

        patch = self.patches(images)

        encoded = self.projection(patch) + embedded
        return encoded
