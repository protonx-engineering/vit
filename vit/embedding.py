from tensorflow.keras.layers import Layer, Embedding, Dense
import tensorflow as tf


class Patches(Layer):
    def __init__(self, patch_size):
        """ Patches
            Parameters
            ----------
            patch_size: int
                size of a patch (P)
        """
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        """ Pass images to get patches
            Parameters
            ----------
            images: tensor,
                images from dataset
                shape: (..., W, H, C). Example: (64, 32, 32, 3)
            Returns
            -------
            patches: tensor,
                patches extracted from images
                shape: (..., S, P^2 x C) with S = (HW)/(P^2) Example: (64, 64, 48)
        """
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
        """ PatchEmbedding
            Parameters
            ----------
            patch_size: int
                size of a patch (P)
            image_size: int
                size of a image (H or W)
            projection_dim: D
                size of project dimension before passing patches through transformer
        """
        super(PatchEmbedding, self).__init__()

        # S = self.num_patches: Number of patches
        self.num_patches = (image_size // patch_size) ** 2

        # cls token used for last mlp network
        self.cls_token = self.add_weight(
            "cls_token",
            shape=[1, 1, projection_dim],
            initializer=tf.keras.initializers.RandomNormal(),
            dtype=tf.float32
        )
        self.patches = Patches(patch_size)

        self.projection = Dense(units=projection_dim)

        # self.position_embedding shape: (..., S + 1, D)
        self.position_embedding = self.add_weight(
            "position_embeddings",
            shape=[self.num_patches + 1, projection_dim],
            initializer=tf.keras.initializers.RandomNormal(),
            dtype=tf.float32
        )

    def call(self, images):
        """ Pass images to embed position information 
            Parameters
            ----------
                        images: tensor,
                images from dataset
                shape: (..., W, H, C). Example: (64, 32, 32, 3)
            Returns
            -------
            encoded_patches: tensor,
                embed patches with position information and concat with cls token
                shape: (..., S + 1, D) with S = (HW)/(P^2) Example: (64, 65, 768)
        """

        # Get patches from images
        # patch shape: (..., S, NEW_C)
        patch = self.patches(images)

        # encoded_patches shape: (..., S, D)
        encoded_patches = self.projection(patch)

        batch_size = tf.shape(images)[0]

        hidden_size = tf.shape(encoded_patches)[-1]

        # cls_broadcasted shape: (..., 1, D)
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls_token, [batch_size, 1, hidden_size]),
            dtype=images.dtype,
        )

        # encoded_patches shape: (..., S + 1, D)
        encoded_patches = tf.concat([cls_broadcasted, encoded_patches], axis=1)

        # encoded_patches shape: (..., S + 1, D)
        encoded_patches = encoded_patches + self.position_embedding

        return encoded_patches
