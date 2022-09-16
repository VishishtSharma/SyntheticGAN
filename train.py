# code to train the network
# Shivan Bhatt 18249

import sys
sys.path.insert(0, r'C:\studym\sem7\AI\t69')                    #adding folder path

from cgan import CGAN                                           #importing necessary modules
from discriminative_network import DiscriminativeNetwork
from generative_network import GenerativeNetwork
from data_generator import SentinelDataGenerator

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'                     #uncomment to not use GPU acceleration

if __name__ == '__main__':
    satellite_image_shape = (256, 256, 4)                       #set the size for image, 4 channels for rgb and infrared
    mask_shape = (256, 256, 1)                                  #set size for mask
    output_channels = satellite_image_shape[2]
    init_filters = 64                                           #no of channels in the output of conv layer

    data_generator = SentinelDataGenerator('bdot', satellite_image_shape=(4, 256, 256),
                                           landcover_mask_shape=(1, 256, 256), feature_range=(-1, 1))    #preparing the data, change the satellite images to raster data
    dn = DiscriminativeNetwork()
    dn_model = dn.build(init_filters=init_filters, input_shape=satellite_image_shape, condition_shape=mask_shape,    #building the discriminative network
                        kernel_size=(3, 3))

    gn = GenerativeNetwork()
    gn_model = gn.build(init_filters=init_filters, input_shape=mask_shape, output_channels=output_channels,          #building the generative network
                        compile=False, dropout_rate=0.5, kernel_size=(7, 7))
    cgan = CGAN(data_generator, dn_model, gn_model, input_shape=satellite_image_shape, condition_shape=mask_shape)

    history = cgan.fit(epochs=200, batch=50)                    #fitting the cgan model

    print('Sentinel CGAN has been fitted')
