from vit.model import ViT, ViTBase, ViTHuge, ViTLarge
from tensorflow import keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.data import Dataset
import tensorflow_addons as tfa
import numpy as np

from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--model', default='custom', type=str,
                        help='Type of ViT model, valid option: custom, base, large, huge')
    parser.add_argument('--num-classes', default=10,
                        type=int, help='Number of classes')
    parser.add_argument('--patch-size', default=2,
                        type=int, help='Size of image patch')
    parser.add_argument('--num-heads', default=4,
                        type=int, help='Number of attention heads')
    parser.add_argument('--att-size', default=64,
                        type=int, help='Size of each attention head for value')
    parser.add_argument('--num-layer', default=2,
                        type=int, help='Number of attention layer')
    parser.add_argument('--mlp-size', default=128,
                        type=int, help='Size of hidden layer in MLP block')
    parser.add_argument('--lr', default=0.001,
                        type=float, help='Learning rate')
    parser.add_argument('--weight-decay', default=1e-4,
                        type=float, help='Weight decay')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of training epoch')
    parser.add_argument('--image-size', default=224,
                        type=int, help='Size of input image')
    parser.add_argument('--image-channels', default=3,
                        type=int, help='Number channel of input image')
    parser.add_argument('--train-folder', default='', type=str,
                        help='Where training data is located')
    parser.add_argument('--valid-folder', default='', type=str,
                        help='Where validation data is located')
    parser.add_argument('--model-folder', default='.output/',
                        type=str, help='Folder to save trained model')
    
    
    args = parser.parse_args()
    print('---------------------Welcome to ProtonX MLP Mixer-------------------')
    print('Github: bangoc123 and tiena2cva')
    print('Email: protonxai@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training Vit Transformer model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')

    

    if args.train_folder != '' and args.valid_folder != '':
        # Load train images from folder
        train_ds = image_dataset_from_directory(
            args.train_folder,
            seed=123,
            image_size=(args.image_size, args.image_size),
            shuffle=True,
            batch_size=args.batch_size,
        )
        val_ds = image_dataset_from_directory(
            args.valid_folder,
            seed=123,
            image_size=(args.image_size, args.image_size),
            shuffle=True,
            batch_size=args.batch_size,
        )
    else:
        print("Data folder is not set. Use CIFAR 10 dataset")

        args.image_channels = 3
        args.num_classes = 10

        (x_train, y_train), (x_val, y_val) = keras.datasets.cifar10.load_data()
        x_train = (x_train.reshape(-1, args.image_size, args.image_size,
                                   args.image_channels)).astype(np.float32)
        x_val = (x_val.reshape(-1, args.image_size, args.image_size,
                               args.image_channels)).astype(np.float32)

        # create dataset
        train_ds = Dataset.from_tensor_slices((x_train, y_train))
        train_ds = train_ds.batch(args.batch_size)

        val_ds = Dataset.from_tensor_slices((x_val, y_val))
        val_ds = val_ds.batch(args.batch_size)

    if args.model == 'base':
        model = ViTBase()
    elif args.model == 'large':
        model = ViTLarge()
    elif args.model == 'huge':
        model = ViTHuge()
    else:
        model = ViT(
            num_classes=args.num_classes,
            patch_size=args.patch_size,
            image_size=args.image_size,
            num_heads=args.num_heads,
            D=args.att_size,
            mlp_dim=args.mlp_size,
            num_layers=args.num_layer
        )

    model.build(input_shape=(None, args.image_size,
                             args.image_size, args.image_channels))

    optimizer = tfa.optimizers.AdamW(
        learning_rate=args.lr, weight_decay=args.weight_decay)
    loss = SparseCategoricalCrossentropy()
    model.compile(optimizer, loss=loss,
                  metrics=['accuracy'])

    # Traning
    model.fit(train_ds,
              epochs=args.epochs,
              batch_size=args.batch_size,
              validation_data=val_ds)

    # Save model
    model.save(args.model_folder)
