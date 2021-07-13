from tensorflow.python.keras.layers.core import Dropout
from vit.embedding import PatchEmbedding
from vit.encoder import TransformerEncoder
from tensorflow.keras.layers import Dense, Flatten, LayerNormalization
from tensorflow.keras.layers.experimental.preprocessing import Normalization, Resizing, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model


class ViT(Model):
    def __init__(self, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, num_classes=10, patch_size=16, image_size=224, dropout=0.1, norm_eps=1e-12):
        super(ViT, self).__init__()

        # Data augmentation
        self.data_augmentation = Sequential([
            Normalization(),
            Resizing(image_size, image_size),
            RandomFlip("horizontal"),
            RandomRotation(factor=0.02),
            RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ])

        # Patch embedding
        self.embedding = PatchEmbedding(patch_size, image_size, hidden_dim)

        # Encoder with transformer
        self.encoder = TransformerEncoder(
            num_heads=num_heads,
            num_layers=num_layers,
            embed_dim=hidden_dim,
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

    def call(self, inputs):
        # Create augmented data
        augmented = self.data_augmentation(inputs)

        # Create position embedding
        embedded = self.embedding(augmented)

        # Encode patchs with transformer
        encoded = self.encoder(embedded)

        # Feed MLP head
        output = self.mlp_head(encoded[:, 0])

        return output


class ViTBase(ViT):
    def __init__(self, num_classes=10, patch_size=16, image_size=224, dropout=0.1, norm_eps=1e-12):
        super().__init__(num_layers=12,
                         num_heads=12,
                         hidden_dim=768,
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
                         hidden_dim=1024,
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
                         hidden_dim=1280,
                         mlp_dim=5120,
                         num_classes=num_classes,
                         patch_size=patch_size,
                         image_size=image_size,
                         dropout=dropout,
                         norm_eps=norm_eps)
