from tensorflow.keras.layers import Layer, LayerNormalization, MultiHeadAttention, Dense, Dropout, Flatten
from tensorflow.keras import Sequential


class MLPBlock(Layer):
    def __init__(self, hidden_layers, dropout=0.1, activation='gelu'):
        """ MLP Block in Transformer Encoder

            Parameters
            ----------
            hidden_layers: Python array
                list of layers for mlp block
            dropout: float,
                dropout rate of mlp block
            activation: string
                activation of mlp layer
        """
        super(MLPBlock, self).__init__()

        layers = []
        for num_units in hidden_layers:
            layers.extend([
                Dense(num_units, activation=activation),
                Dropout(dropout)
            ])

        self.mlp = Sequential(layers)

    def call(self, inputs, *args, **kwargs):
        """ Pass output of multi-head attention to mlp block
            Parameters
            ----------
            inputs: tensor,
                multi-head attention outputs
                shape: (..., S, D). Example: (64, 100, 768)
            Returns
            -------
            outputs: tensor,
                attention + mlp outputs
                shape: (..., S, D). Example: (64, 100, 768)
        """

        outputs = self.mlp(inputs, *args, **kwargs)
        return outputs


class TransformerBlock(Layer):
    def __init__(self, num_heads, D, hidden_layers, dropout=0.1, norm_eps=1e-12):
        """ Transformer blocks which includes multi-head attention layer and mlp block

            Parameters
            ----------
            num_heads: int,
                number of heads of multi-head attention layer
            D: int, 
                size of each attention head for value
                        hidden_layers: Python array
                list of layers for mlp block
            dropout: float,
                dropout rate of mlp block
            norm_eps: float,
                eps of layer norm
        """
        super(TransformerBlock, self).__init__()
        # Attention
        self.attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=D, dropout=dropout
        )
        self.norm_attention = LayerNormalization(epsilon=norm_eps)

        # MLP
        self.mlp = MLPBlock(hidden_layers, dropout)
        self.norm_mlp = LayerNormalization(epsilon=norm_eps)

    def call(self, inputs):
        """
            Pass Embedded Patches through the layers
            Parameters
            ----------
            inputs: tensor,
                Embedded Patches
                shape: (..., S, D). Example: (64, 100, 768)
            Returns
            -------
            outputs: tensor,
                attention + mlp outputs
                shape: (..., S, D). Example: (64, 100, 768)
        """
        # Feed attention
        norm_attention = self.norm_attention(inputs)

        attention = self.attention(query=norm_attention, value=norm_attention)

        # Skip Connection
        attention += inputs

        # Feed MLP
        outputs = self.mlp(self.norm_mlp(attention))

        # Skip Connection
        outputs += attention

        return outputs


class TransformerEncoder(Layer):
    def __init__(self, num_layers, num_heads, D, mlp_dim, dropout=0.1, norm_eps=1e-12):
        """
            Transformer Encoder which comprises several transformer layers
            Parameters
            ----------
            num_layers: int,
                number of transformer layers
                Example: 12
            num_heads: int,
                number of heads of multi-head attention layer
            D: int
                size of each attention head for value
            mlp_dim: 
                mlp size or dimension of hidden layer of mlp block
            dropout: float,
                dropout rate of mlp block
            norm_eps: float,
                eps of layer norm
        """
        super(TransformerEncoder, self).__init__()

        # Create num_layers of TransformerBlock
        self.encoder = Sequential(
            [
                TransformerBlock(num_heads=num_heads,
                                 D=D,
                                 hidden_layers=[mlp_dim, D],
                                 dropout=dropout,
                                 norm_eps=norm_eps)
                for _ in range(num_layers)
            ]
        )

    def call(self, inputs, *args, **kwargs):
        """
            Pass Embedded Patches through the layers
            Parameters
            ----------
            inputs: tensor,
                Embedded Patches
                shape: (..., S, D). Example: (64, 100, 768)
            Returns
            -------
            outputs: tensor,
                attention + mlp outputs
                shape: (..., S, D). Example: (64, 100, 768)
        """
        outputs = self.encoder(inputs, *args, **kwargs)
        return outputs
