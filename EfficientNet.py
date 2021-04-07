import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from math import ceil

base_model = [
    # expand_ratio, channels, repeats, strides, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3]
]

phi_values = {
    # alpha (depth), beta (width), gamma (resolution) (e.g. depth = alpha ** phi)
    # tuple of: (phi_value, resolution, droprate)
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_out', distribution='truncated_normal')

DENSE_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(
            scale=1. / 3., mode='fan_out', distribution='uniform')

class CNNBlock(layers.Layer):
    def __init__(
                self, 
                filters, 
                kernel_size, 
                strides, 
                padding,
                groups=1,
    ):
        super(CNNBlock, self).__init__()
        
        self.cnn = keras.Sequential([
            
            layers.ZeroPadding2D(padding=(padding, padding)),
            layers.Conv2D(
                filters = filters,
                kernel_size = kernel_size,
                strides = strides,
                use_bias = False,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                groups=groups,
            )
        ])
         
        self.bn = layers.BatchNormalization()
        self.silu = layers.Activation('swish')
        
    def call(self, inputs):
        return self.silu(self.bn(self.cnn(inputs)))
    
"""
class CNNBlockDepthwise(layers.Layer):
    def __init__(
                self, 
                filters=32, 
                kernel_size=3, 
                strides=2, 
                padding='valid'
    ):
        super(CNNBlockDepthwise, self).__init__()
        
        self.cnn = layers.DepthwiseConv2D(
            kernel_size = kernel_size,
            strides = strides,
            padding = padding,
            use_bias = False,
            kernel_initializer=CONV_KERNEL_INITIALIZER
        )
        self.bn = layers.BatchNormalization()
        self.silu = layers.Activation('swish')
        
    def call(self, inputs):
        return self.silu(self.bn(self.cnn(inputs)))
"""
class SqueezeExcitation(layers.Layer):
    def __init__(
            self, 
            filters, 
            filters_reduced_dim
    ):
        
        super(SqueezeExcitation, self).__init__()
        
        self.se = keras.Sequential([
            layers.GlobalAveragePooling2D(), # C x H x W -> C x 1 x 1
            layers.Reshape((1, 1, filters)),
            layers.Conv2D(
                filters = filters_reduced_dim, 
                kernel_size = 1,
                padding = 'same',
                kernel_initializer=CONV_KERNEL_INITIALIZER
            ),
            layers.Activation('swish'),
            layers.Conv2D(
                filters = filters, 
                kernel_size = 1,
                padding = 'same',
                kernel_initializer=CONV_KERNEL_INITIALIZER
            ),
            layers.Activation('sigmoid')
        ])
        
    def call(self, inputs):
        return layers.multiply([inputs, self.se(inputs)])
        
class InvertedResidualBlock(layers.Layer):
    def __init__(
            self, 
            filters_in, 
            filters_out,
            kernel_size, 
            strides, 
            padding,
            expand_ratio, 
            reduction=4, # squeeze excitation (e.g 1/4)
            survial_prob=0.8, # stochastic depth
    ):
        super(InvertedResidualBlock, self).__init__()
        
        self.survial_prob = survial_prob
        
        self.training = False
        
        filters = filters_in * expand_ratio
        filters_reduced_dim = int(filters_in / reduction)
        
        self.use_residual = filters_in == filters_out and strides == 1
        self.expand = filters_in != filters
        
        if self.expand:
            self.expand_conv = CNNBlock(
                filters, kernel_size=3, strides=1, padding=padding
            )
        
        if self.use_residual:
            self.conv = keras.Sequential([
            CNNBlock(filters, kernel_size, strides, padding=2,groups=filters),
            SqueezeExcitation(filters,filters_reduced_dim),
            layers.Conv2D(filters_out, kernel_size=3, use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER),
            layers.BatchNormalization()
        ])
        else:
            self.conv = keras.Sequential([
                CNNBlock(filters, kernel_size, strides, padding=padding,groups=filters),
                SqueezeExcitation(filters,filters_reduced_dim),
                layers.Conv2D(filters_out, kernel_size=1, use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER),
                layers.BatchNormalization()
            ])
    def stochastic_depth(self, x):
        if not self.training:
            return x
        
        binary_tensor = tf.random.uniform(x.shape[0],0,1) < self.survial_prob
        return tf.divide(x, self.survial_prob) * binary_tensor
    
    def call(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        
        if self.use_residual:
            return self.conv(x) + inputs
            #return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)
        
class EfficientNet(layers.Layer):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = layers.GlobalAveragePooling2D()
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = keras.Sequential([
            layers.Dropout(dropout_rate),
            layers.Flatten(),
            layers.Dense(last_channels),
            layers.Dense(num_classes, activation='softmax', kernel_initializer=DENSE_KERNEL_INITIALIZER)
        ])
        
    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate
    
    def create_features(self, width_factor, depth_factor, last_channels):
        filters = int(32 * width_factor)
        features = [
            
            CNNBlock(
            filters, 
            kernel_size=3,
            strides=2,
            padding=1
            )
        ]
        filters_in = filters
        
        for expand_ratio, channels, repeats, strides, kernel_size in base_model:
            filters_out = 4 * ceil(int(channels * width_factor))
            layers_repeats = ceil(repeats * depth_factor)
            
            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        filters_in = filters_in,
                        filters_out = filters_out,
                        kernel_size = kernel_size,
                        strides = strides if layer == 0 else 1,
                        expand_ratio = expand_ratio,
                        padding=kernel_size//2, #if  k=1,pad=0; k=3:pad=1, k=5:pad=2
                    )
                )
                filters_in = filters_out
        
        features.append(
            CNNBlock(
                last_channels, kernel_size=1, strides=1, padding=0
            )
        )
        
        return keras.Sequential(features)
    
    def call(self, inputs):
        
        return keras.Sequential([
            keras.Input(shape=inputs),
            self.features, 
            self.pool,
            self.classifier
        ],name="EfficientNet")
    

version = "b0"
phi, res, drop_rate = phi_values[version]
num_classes = 10
input_shape = (64, 64, 3)

model = EfficientNet(version, num_classes)    

model = model(input_shape)

model.compile(
        optimizer=keras.optimizers.Adam(lr=3e-4),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['Accuracy']
        )
    
model.summary()


