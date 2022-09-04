import tensorflow as tf
from keras.layers import (Layer,
                          Conv2D, 
                          MaxPooling2D,
                          BatchNormalization,
                          ReLU,
                          Dense,
                          Flatten,
                          Dropout,
                          Embedding,
                          Input,
                          Normalization,
                          Resizing,
                          RandomFlip,
                          RandomRotation,
                          RandomZoom,
                          LayerNormalization,
                          MultiHeadAttention,
                          Add)
from keras.models import Model
from keras.models import Sequential


class ConvNN:
    def __init__(self, height, width, filter_size, 
                 pool_size, stride, num_actions, learning_rate):
        self.height = height
        self.weight = width
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.stride = stride
        self.num_actions = num_actions
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        
        # # CNN
        model = Sequential()
        # model.add(Input(shape=(height, width)))
        model.add(self.stacked_layer('L1', self.filter_size, self.pool_size, 
                                     self.stride, 16, 2))
        model.add(self.stacked_layer('L2', self.filter_size, self.pool_size,
                                     self.stride, 32, 2))
        model.add(Flatten())
        model.add(self.fully_connected(32, name="FC_1"))
        model.add(Dense(self.num_actions, name="FC_2",
                        kernel_initializer="glorot_uniform",
                        bias_initializer="truncated_normal"))
        
        # # Transformer 
        # model = self.create_vit_classifier()
        
        self.model = model
        

    def q_value(self, state, is_training):
        X = tf.reshape(state, [-1, self.height, self.weight, 1])
        mean, variance = tf.nn.moments(X, [0, 1, 2, 3])
        X = self.normalize_input(X, mean, variance)

        rho = self.model(X, training=is_training)
        eta = tf.one_hot(tf.argmax(rho, 1),
                         self.num_actions,
                         on_value=1,
                         off_value=0,
                         dtype=tf.int32)

        return rho, eta

    def normalize_input(self, X, mean, variance):
        return (X - mean) / tf.sqrt(variance)

    def fully_connected(self, units, name):
        fc = Sequential()

        fc.add(Dense(
            units,
            name=name,
            kernel_initializer="glorot_uniform",
            bias_initializer="truncated_normal"))
        fc.add(BatchNormalization())
        fc.add(ReLU())

        return fc

    def stacked_layer(self, name, filter_size, pool_size, stride,
                      output_size, num_layers):
        stack = Sequential()

        for i in range(num_layers):
            stack.add(Conv2D(
                output_size,
                (filter_size, filter_size),
                padding="same",
                strides=(stride, stride),
                name=f"{name}_{i}"))
            stack.add(BatchNormalization())
            stack.add(ReLU())

        stack.add(MaxPooling2D(pool_size=(
            pool_size, pool_size), padding='same'))

        return stack
    
    def mlp(self, x, hidden_units: list, dropout_rate):
        for units in hidden_units:
            x = Dense(units, activation=tf.nn.gelu)(x)
            x = Dropout(dropout_rate)(x)
        return x

    #TODO: refactor
    def optimize_q_old(self, Q, action, target, batch_size, learning_rate):
        # Q : Batch * numaction
        # A : Batch * numaction
    
        # update BatchNorm var first, and then update Loss, Opt
        updates = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.UPDATE_OPS)
        trablevars = tf.compat.v1.trainable_variables()

        with tf.control_dependencies(updates):

            loss = tf.reduce_sum(tf.square(target - (Q*action))) / batch_size
            opt = tf.compat.v1.train.AdamOptimizer(learning_rate)

            grads = opt.compute_gradients(loss)
            minz = opt.minimize(loss)

        return loss, grads, updates, trablevars, minz


    def optimize_q(self, S, action, target, batch_size):
        with tf.GradientTape() as tape:
            rho = self.q_value(S, True)[0]
            loss = tf.reduce_sum(tf.square(target - (rho * action))) / batch_size

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            
        return loss


    def create_vit_classifier(self):
        #TODO: get these in the constructor
        inputs = Input(shape=(18, 18, 1))
        # Augment data ? 
        data_augmentation = Sequential(
            [
                Normalization(),
                Resizing(72, 72),
                RandomFlip("horizontal"),
                RandomRotation(factor=0.02),
                RandomZoom(
                    height_factor=0.2, width_factor=0.2
                ),
            ],
            name="data_augmentation",
        )
        augmented = data_augmentation(inputs)
        # Create patches. 
        patches = Patches(6)(augmented)
        # Encode patches.
        # num_patches = (image_size // patch_size) ** 2
        num_patches = (72 // 6) ** 2
        encoded_patches = PatchEncoder(num_patches, 64)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(8):
            # Layer normalization 1.
            x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = MultiHeadAttention(
                num_heads=4, key_dim=64, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = self.mlp(x3, hidden_units=[128, 64], dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = Flatten()(representation)
        representation = Dropout(0.5)(representation)
        # Add MLP.
        features = self.mlp(representation, hidden_units=[2048, 1024], dropout_rate=0.5)
        # Classify outputs.
        logits = Dense(3)(features)
        # Create the Keras model.
        model = Model(inputs=inputs, outputs=logits)
        return model

class Patches(Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
