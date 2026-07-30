import numpy as np
import tensorflow as tf

def get_gradcam(model, beat, class_idx):
    """
    beat: shape (180, 1)
    class_idx: integer (0=A, 1=L, 2=N, 3=R, 4=V)
    returns: heatmap of shape (180,)
    """
    last_conv_layer = model.get_layer('last_conv')
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    input_tensor = tf.convert_to_tensor(beat[np.newaxis, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        conv_outputs, predictions = grad_model(input_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    # grads shape: (1, timesteps, filters)
    # conv_outputs shape: (1, timesteps, filters)

    # Average over timesteps and batch → shape: (filters,)
    pooled_grads = tf.reduce_mean(grads, axis=[0, 1])

    # conv_outputs[0] → shape: (timesteps, filters)
    conv_out = conv_outputs[0]

    # Multiply each filter by its importance weight → (timesteps, filters)
    weighted = conv_out * pooled_grads[tf.newaxis, :]

    # Sum across filters → (timesteps,)
    heatmap = tf.reduce_sum(weighted, axis=-1).numpy()

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)

    # Upsample to 180
    heatmap_resized = np.interp(
        np.linspace(0, len(heatmap) - 1, 180),
        np.arange(len(heatmap)),
        heatmap
    )

    return heatmap_resized