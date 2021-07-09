from vit.embedding import PatchEmbedding
from vit.encoder import TransformerEncoder
from tensorflow.keras.layers import Dense, Flatten
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
            Flatten(),
            Dense(mlp_dim),
            Dense(num_classes),
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
