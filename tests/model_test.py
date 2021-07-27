import tensorflow as tf
from vit.model import ViT


class ModelTest(tf.test.TestCase):
    def test_vit(self):
        num_classes = 10
        image_size = 32
        patch_size = 8
        batch = 2

        # Input shape: [batch, image_size, image_size, channels]
        # Expected output shape: [batch, num_classes]

        inputs = tf.ones((batch, image_size, image_size, 3))
        model = ViT(num_classes=num_classes,
                    image_size=image_size,
                    patch_size=patch_size)

        logits = model(inputs)
        self.assertEqual(logits.shape, (batch, num_classes))
