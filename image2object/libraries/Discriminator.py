# Discriminator
import tensorflow as tf

class Discriminator:
    #nWidth = 512
    #nHeight = 512
    
    # input_image, generator_output and target_image have 4 dimensions
    # (batch_size, image_width, image_height, color_channel)
    # input_image : input for GAN (also input for generator)
    # generator_output : generated image by generator
    # target_image : expected image, which is input for GAN
    def __init__(self, input_image, generator_output, target_image, **options):
        self.input_image = input_image
        self.generator_output = generator_output 
        self.target_image = target_image
        
        self.layer_generator_output = self.prepare_model(generator_output, "discriminator_for_generator_output")
        self.layer_target_image = self.prepare_model(target_image, "discriminator_for_target_image")
        self.loss = self.calc_loss()
        ## TODO:this may be necessary
        ##self.prepare_session()

        #with tf.Graph().as_default():
        #with tf.get_default_graph():
        #    self.layer_generator_output = self.prepare_model(generator_output, "discriminator_for_generator_output")
        #    self.layer_target_image = self.prepare_model(target_image, "discriminator_for_target_image")
        #    self.loss = self.cals_loss()
            # TODO:this may be necessary
            #self.prepare_session()

    # discriminate_target is generator_output or target_image
    # this method returns last layer 
    def prepare_model(self, discriminate_target, scope_name):
        with tf.variable_scope(scope_name):
            input = tf.concat([self.input_image, discriminate_target], axis=3);
            print(input)

            # Leaky ReLU
            def lrelu(x, a):
                with tf.name_scope("LeakyReLU"):
                    x = tf.identity(x)
                    # leak[a*x/2 - a*abs(x)/2] + linear[x/2 + abs(x)/2]
                    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

            # Batch Normalization
            def batchnorm(input):
                with tf.variable_scope("BatchNormalization"):
                    input = tf.identity(input)

                    channels = input.get_shape()[3]
                    offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer)
                    scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
                    mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
                    variance_epsilon = 1e-5
                    normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
                    return normalized

            # convolutiobn for discriminator
            def convolve(input, filters, stride, krnl_size):
                # Add constant 0 padding to image width and height dimension
                with_padding = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")

                conved = tf.layers.conv2d(with_padding, filters, kernel_size=krnl_size, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02));
                return conved
           
            num_channels = 1
            out_channels_base = 32
            # list consists of elements(out_channels, stride)
            layer_spec = [
                (64,   2), # disc_layer_1: [batch_size, 512, 512, 1]  => [batch_size, 256, 256, 64]
                (128,  2), # disc_layer_2: [batch_size, 256, 256, 64] => [batch_size, 128, 128, 128]
                (256,  2), # disc_layer_3: [batch_size, 128, 128, 128] => [batch_size, 64, 64, 256]
                (512,  2), # disc_layer_4: [batch_size, 64, 64, 256] => [batch_size, 32, 32, 512]
                (1024, 1), # disc_layer_5: [batch_size, 32, 32, 512] => [batch_size, 31, 31, 1024]
            ]

            # discriminator layers
            layers = []
            
            kernel_size = 4
            for layer_index, (out_channels, stride) in enumerate(layer_spec):
                with tf.variable_scope("discriminator_layer_%d" % (len(layers) + 1)):
                    
                    if layer_index == 0:
                        convolved = convolve(input, out_channels, stride, kernel_size)
                        activated = lrelu(convolved, 0.2)
                    else:
                        convolved = convolve(layers[-1], out_channels, stride, kernel_size)
                        normalized = batchnorm(convolved)
                        activated = lrelu(convolved, 0.2)

                    layers.append(activated)
                    print(activated)
            
            # disc_layer_last: [batch_size, 31, 31, 1024] => [batch_size, 30, 30, 1]
            with tf.variable_scope("discriminator_layer_%d" % (len(layers) + 1)):
                # 2nd and 3rd arguments are number of output channel and stride for convolution layer
                convolved = convolve(layers[-1], 1, 2, kernel_size)
                activated = tf.sigmoid(convolved)
                
            layers.append(activated)

            return layers[-1]
    
    def calc_loss(self):
        eps = 1e-7
        return tf.reduce_mean( -tf.log(self.layer_target_image + eps) + tf.log(1 - self.layer_generator_output + eps) )
