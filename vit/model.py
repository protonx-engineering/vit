from tensorflow.python.keras.layers.core import Dropout
from vit.embedding import PatchEmbedding
from vit.encoder import TransformerEncoder
from tensorflow.keras.layers import Dense, LayerNormalization
from tensorflow.keras.layers.experimental.preprocessing import Resizing, RandomFlip, RandomRotation, RandomZoom, Rescaling
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model


class ViT(Model):
    def __init__(self, num_layers=12, num_heads=12, D=768, mlp_dim=3072, num_classes=10, patch_size=16, image_size=224, dropout=0.1, norm_eps=1e-12):
        """
            VIT Model
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
            num_classes:
                number of classes
            patch_size: int
                size of a patch (P)
            image_size: int
                size of a image (H or W)
            dropout: float,
                dropout rate of mlp block
            norm_eps: float,
                eps of layer norm
        """
        super(ViT, self).__init__()
        # Data augmentation
        self.data_augmentation = Sequential([
            Rescaling(scale=1./255),
            Resizing(image_size, image_size),
            RandomFlip("horizontal"),
            RandomRotation(factor=0.02),
            RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ])

        # Patch embedding
        self.embedding = PatchEmbedding(patch_size, image_size, D)

        # Encoder with transformer
        self.encoder = TransformerEncoder(
            num_heads=num_heads,
            num_layers=num_layers,
            D=D,
            mlp_dim=mlp_dim,
            dropout=dropout,
            norm_eps=norm_eps,
        )

        # MLP head
        self.mlp_head = Sequential([
            LayerNormalization(epsilon=norm_eps),
            Dense(mlp_dim),
            Dropout(dropout),
            Dense(num_classes, activation='softmax'),
        ])

        self.last_layer_norm = LayerNormalization(epsilon=norm_eps)

    def call(self, inputs):
        # Create augmented data
        # augmented shape: (..., image_size, image_size, c)
        augmented = self.data_augmentation(inputs)

        # Create position embedding + CLS Token
        # embedded shape: (..., S + 1, D)
        embedded = self.embedding(augmented)

        # Encode patchs with transformer
        # embedded shape: (..., S + 1, D)
        encoded = self.encoder(embedded)

        # Embedded CLS
        # embedded_cls shape: (..., D)
        embedded_cls = encoded[:, 0]
        
        # Last layer norm
        y = self.last_layer_norm(embedded_cls)
        
        # Feed MLP head
        # output shape: (..., num_classes)

        output = self.mlp_head(y)

        return output


class ViTBase(ViT):
    def __init__(self, num_classes=10, patch_size=16, image_size=224, dropout=0.1, norm_eps=1e-12):
        super().__init__(num_layers=12,
                         num_heads=12,
                         D=768,
                         mlp_dim=3072,
                         num_classes=num_classes,
                         patch_size=patch_size,
                         image_size=image_size,
                         dropout=dropout,
                         norm_eps=norm_eps)


class ViTLarge(ViT):
    def __init__(self, num_classes=10, patch_size=16, image_size=224, dropout=0.1, norm_eps=1e-12):
        super().__init__(num_layers=24,
                         num_heads=16,
                         D=1024,
                         mlp_dim=4096,
                         num_classes=num_classes,
                         patch_size=patch_size,
                         image_size=image_size,
                         dropout=dropout,
                         norm_eps=norm_eps)


class ViTHuge(ViT):
    def __init__(self, num_classes=10, patch_size=16, image_size=224, dropout=0.1, norm_eps=1e-12):
        super().__init__(num_layers=32,
                         num_heads=16,
                         D=1280,
                         mlp_dim=5120,
                         num_classes=num_classes,
                         patch_size=patch_size,
                         image_size=image_size,
                         dropout=dropout,
                         norm_eps=norm_eps)
