
import tensorflow as tf
from vit.embedding import Patches, PatchEmbedding


class EmbeddingTest(tf.test.TestCase):
    def test_patches(self):
        batch = 4
        W = 32
        H = 32
        C = 3
        P = 8

        S = (H * W)/(P * P)
        D = P * P * C

        # Input shape: [batch, H, W, C]
        # Expected output shape: [batch, S, D]

        images = tf.ones([batch, H, W, C])
        patches_splitter = Patches(patch_size=P)

        patches = patches_splitter(images)

        self.assertEqual(patches.shape.as_list(), [batch, S, D])

    def test_patch_embedding(self):
        batch = 4
        image_size = 32
        channels = 3
        patch_size = 8
        projection_dim = 16

        S = (image_size * image_size)/(patch_size * patch_size)

        # Input shape: [batch, image_size, image_size, channels]
        # Expected output shape: [batch, S + 1, projection_dim]

        images = tf.ones([batch, image_size, image_size, channels])
        patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            image_size=image_size,
            projection_dim=projection_dim
        )

        patches = patch_embedding(images)

        self.assertEqual(patches.shape.as_list(),
                         [batch, S + 1, projection_dim])
