from tensorflow.keras.layers import Layer, LayerNormalization, MultiHeadAttention, Dense, Dropout, Flatten
from tensorflow.keras import Sequential


class MLPBlock(Layer):
    def __init__(self, hidden_layers, dropout=0.1, activation='gelu'):
        super(MLPBlock, self).__init__()

        layers = []
        for num_units in hidden_layers:
            layers.extend([
                Dense(num_units, activation=activation),
                Dropout(dropout)
            ])

        self.mlp = Sequential(layers)

    def call(self, inputs, *args, **kwargs):
        return self.mlp(inputs, *args, **kwargs)


class TransformerBlock(Layer):
    def __init__(self, num_heads, dim, hidden_layers, dropout=0.1, norm_eps=1e-12):
        super(TransformerBlock, self).__init__()

        # Attention
        self.attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=dim, dropout=dropout
        )
        self.norm_attention = LayerNormalization(epsilon=norm_eps)

        # MLP
        self.mlp = MLPBlock(hidden_layers, dropout)
        self.norm_mlp = LayerNormalization(epsilon=norm_eps)

    def call(self, inputs):
        # Feed attention
        norm_attention = self.norm_attention(inputs)
        attention = self.attention(query=norm_attention, value=norm_attention)
        attention += inputs

        # Feed MLP
        outputs = self.mlp(self.norm_mlp(attention))
        outputs += attention

        return outputs


class TransformerEncoder(Layer):
    def __init__(self, num_layers, num_heads, embed_dim, mlp_dim, dropout=0.1, norm_eps=1e-12):
        super(TransformerEncoder, self).__init__()

        # Create num_layers of TransformerBlock
        self.encoder = Sequential(
            [
                TransformerBlock(num_heads=num_heads,
                                 dim=embed_dim,
                                 hidden_layers=[mlp_dim, embed_dim],
                                 dropout=dropout,
                                 norm_eps=norm_eps)
                for _ in range(num_layers)
            ]
        )

    def call(self, inputs, *args, **kwargs):
        return self.encoder(inputs, *args, **kwargs)
