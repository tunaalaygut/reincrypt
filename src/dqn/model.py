import tensorflow as tf
from keras.layers import (Layer, Dense, Flatten, Dropout, Embedding, Input,
                          LayerNormalization, MultiHeadAttention, Add, Resizing)
from keras.models import Model, Sequential


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

class ViT:
    def __init__(self, config: dict):
        self.height = config["height"]
        self.width = config["width"]
        self.num_actions = config["num_actions"]
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config["learning_rate"])
        self.patch_size = config["patch_size"]
        self.projection_dim = config["projection_dim"]
        self.mlp_head_units = config["mlp_head_units"]
        self.transformer_units = config["transformer_units"]
        self.num_heads = config["num_heads"]
        self.transformer_layers = config["transformer_layers"]
        self.resized_image_size = config["resized_image_size"]
        self.enable_resizing = config["enable_resizing"]
        self.model = self.create_vit()

    def create_vit(self):
        inputs = Input(shape=(self.height, self.width, 1))
        
        if self.enable_resizing:        
            augmented = Sequential([
                Resizing(self.resized_image_size,
                        self.resized_image_size)])(inputs)
            patches = Patches(self.patch_size)(augmented)
            num_patches = (self.resized_image_size // self.patch_size) ** 2
        else:
            patches = Patches(self.patch_size)(inputs)
            num_patches = (self.height // self.patch_size) ** 2

        encoded_patches = PatchEncoder(num_patches, 
                                       self.projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            # Layer normalization 1.
            x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, 
                dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = self.mlp(x3, hidden_units=self.transformer_units, 
                          dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = Flatten()(representation)
        representation = Dropout(0.5)(representation)
        # Add MLP.
        features = self.mlp(representation, hidden_units=self.mlp_head_units, 
                            dropout_rate=0.5)
        # Classify outputs.
        logits = Dense(3)(features)
        # Create the Keras model.
        model = Model(inputs=inputs, outputs=logits)
        return model

    def mlp(self, x, hidden_units: list, dropout_rate):
        for units in hidden_units:
            x = Dense(units, activation=tf.nn.gelu)(x)
            x = Dropout(dropout_rate)(x)
        return x

    def q_value(self, state, is_training):
        X = tf.reshape(state, [-1, self.height, self.width, 1])
        rho = self.model(X, training=is_training)
        eta = tf.one_hot(tf.argmax(rho, 1),
                         self.num_actions,
                         on_value=1,
                         off_value=0,
                         dtype=tf.int64)

        return rho, eta

    def optimize_q(self, S, action, target, batch_size):
        with tf.GradientTape() as tape:
            rho = self.q_value(S, True)[0]
            loss = self.__custom_loss_func(target, rho, action, batch_size)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, 
                                           self.model.trainable_variables))

        return loss

    @staticmethod
    def __custom_loss_func(target, rho, action, batch_size):
        return tf.reduce_sum(tf.square(target - (rho * action))) / batch_size
