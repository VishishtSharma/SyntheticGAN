# Code to synthesize artificial data
# Sarthak Mishra 18388

from keras import Model
from keras.models import load_model

from data_generator import SentinelDataGenerator
from plotter import Plotter

if __name__ == '__main__':
    generator_model: Model = load_model('generator.h5')                        #loading the gan model for prediction
    generator_model.summary()

    data_generator = SentinelDataGenerator('bdot', satellite_image_shape=(4, 256, 256),                 #preparing the image data for predicting
                                           landcover_mask_shape=(1, 256, 256), feature_range=(-1, 1))
    plotter = Plotter(generator_model, data_generator, sub_dir='predict')               #plotting the results

    plotter.predict_and_plot_images(batch=50)
    print('Finished prediction using %s model' % generator_model.name)
