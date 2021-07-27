import tensorflow as tf
from vit.encoder import MLPBlock, TransformerBlock, TransformerEncoder


class EncoderTest(tf.test.TestCase):
    def test_mlp_block(self):
        batch = 2
        last_axis_size = 3
        last_layer_size = 8
        hidden_layers = [2, 4, last_layer_size]

        # Input shape: [..., last_axis_size]
        # Expected output shape: [..., last_layer_size]

        inputs = tf.ones([batch, last_axis_size])
        mlp_block = MLPBlock(hidden_layers)

        outputs = mlp_block(inputs)

        self.assertEqual(outputs.shape.as_list(), [batch, last_layer_size])

    def test_transformer_block(self):
        batch = 2
        S = 4
        D = 16
        hidden_layers = [2, D]

        # Input shape: [batch, S, D]
        # Expected output shape: [batch, S, D]

        inputs = tf.ones([batch, S, D])
        transformer_block = TransformerBlock(
            num_heads=2, D=D, hidden_layers=hidden_layers)

        outputs = transformer_block(inputs)

        self.assertEqual(outputs.shape.as_list(), [batch, S, D])

    def test_transformer_encoder(self):
        batch = 2
        S = 4
        D = 16

        # Input shape: [batch, S, D]
        # Expected output shape: [batch, S, D]

        inputs = tf.ones([batch, S, D])
        transformer_encoder = TransformerEncoder(
            num_layers=1,
            num_heads=2,
            D=D,
            mlp_dim=8
        )

        outputs = transformer_encoder(inputs)

        self.assertEqual(outputs.shape.as_list(), [batch, S, D])
