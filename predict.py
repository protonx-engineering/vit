from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import tensorflow as tf
from argparse import ArgumentParser
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--test-image", default='./test.png', type=str, required=True)
    parser.add_argument(
        "--model-folder", default='./output/', type=str)

    args = parser.parse_args()

    # Loading Model
    model = load_model(args.model_folder)

    # Load test image
    image = preprocessing.image.load_img(args.test_file_path)
    input_arr = preprocessing.image.img_to_array(image)
    x = np.array([input_arr])

    predictions = model.predict(x)
    print('Result: {}'.format(np.argmax(predictions), axis=1))
