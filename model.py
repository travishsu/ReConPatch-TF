# type: ignore
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


def l2_distance(z):
    diff = tf.expand_dims(z, axis=1) - tf.expand_dims(z, axis=0)
    return tf.reduce_sum(diff ** 2, axis=-1)


class PairwiseSimilarity(layers.Layer):
    def __init__(self, sigma=1.0):
        super(PairwiseSimilarity, self).__init__()
        self.sigma = sigma

    def call(self, z):
        return tf.exp(-l2_distance(z) / self.sigma)


class ContextualSimilarity(layers.Layer):
    def __init__(self, k):
        super(ContextualSimilarity, self).__init__()
        self.k = k

    def call(self, z):
        distances = l2_distance(z)
        kth_nearst = -tf.math.top_k(-distances, k=self.k, sorted=True)[0][:, -1]
        mask = tf.cast(distances <= tf.expand_dims(kth_nearst, axis=-1), tf.float32)

        similarity = tf.matmul(mask, mask, transpose_b=True) / tf.reduce_sum(mask, axis=-1, keepdims=True)
        R = mask * tf.transpose(mask)
        similarity = tf.matmul(similarity, R, transpose_b=True) / tf.reduce_sum(R, axis=-1, keepdims=True)
        return 0.5 * (similarity + tf.transpose(similarity))


class ReConPatch(keras.Model):
    def __init__(self, input_dim, embedding_dim, projection_dim, alpha, margin=0.1):
        super(ReConPatch, self).__init__()
        self.alpha = alpha
        self.margin = margin

        # embedding & projection layers
        self.embedding = layers.Dense(embedding_dim)
        self.projection = layers.Dense(projection_dim)

        # ema ver of embedding & projection layers
        self.ema_embedding = layers.Dense(embedding_dim)
        self.ema_projection = layers.Dense(projection_dim)

        # initialize layers
        self.embedding.build((None, input_dim))
        self.projection.build((None, embedding_dim))
        self.ema_embedding.build((None, input_dim))
        self.ema_projection.build((None, embedding_dim))

        # ema operator
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9)
        self.update_ema()

        self.pairwise_similarity = PairwiseSimilarity(sigma=1.0)
        self.contextual_similarity = ContextualSimilarity(k=3)

    def call(self, x):
        return self.embedding(x)

    def train_step(self, x):
        B, _ = x.shape

        h_ema = self.ema_embedding(x)
        z_ema = self.ema_projection(h_ema)

        p_sim = self.pairwise_similarity(z_ema)
        c_sim = self.contextual_similarity(z_ema)
        w = self.alpha * p_sim + (1 - self.alpha) * c_sim

        with tf.GradientTape() as tape:
            h = self.embedding(x)
            z = self.projection(h)

            # Contrastive loss
            distances = tf.sqrt(l2_distance(z))
            delta = B * distances / tf.reduce_sum(distances, axis=-1, keepdims=True)
            rc_loss = (tf.reduce_sum(w * delta ** 2) + tf.reduce_sum((1 - w) * tf.nn.relu(self.margin - delta) ** 2)) / B

        # Compute gradients
        trainable_vars = self.embedding.trainable_variables + self.projection.trainable_variables
        gradients = tape.gradient(rc_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update EMA
        self.update_ema()

        return {"rc_loss": rc_loss}

    def update_ema(self):
        self.ema.apply(self.embedding.weights + self.projection.weights)

        avg_emb_w, avg_emb_b = self.ema.average(self.embedding.weights[0]), self.ema.average(self.embedding.weights[1])
        avg_proj_w, avg_proj_b = self.ema.average(self.projection.weights[0]), self.ema.average(self.projection.weights[1])

        self.ema_embedding.set_weights([avg_emb_w, avg_emb_b])
        self.ema_projection.set_weights([avg_proj_w, avg_proj_b])
