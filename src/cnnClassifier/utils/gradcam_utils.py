# utils/gradcam_utils.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.cm as cm
from PIL import Image

def try_get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
        try:
            if hasattr(layer, "output") and len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if predictions.shape[-1] == 1:
            pred_index = tf.cast(predictions[0, 0] > 0.5, tf.int64)
        else:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap.numpy(), 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    return heatmap.astype(np.float32), int(pred_index.numpy())

def _compute_predictions_and_gradients(inputs, model, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        preds = model(inputs, training=False)
        if preds.shape[-1] == 1:
            p_idx = 0
        else:
            p_idx = target_class_idx
        pred = preds[:, p_idx]
    grads = tape.gradient(pred, inputs)
    return pred, grads

def integrated_gradients(inputs, model, target_class_idx, baseline=None, steps=50):
    if baseline is None:
        baseline = tf.zeros_like(inputs)

    interpolated_inputs = [
        baseline + (float(i) / steps) * (inputs - baseline)
        for i in range(0, steps + 1)
    ]
    interpolated_inputs = tf.concat(interpolated_inputs, axis=0)

    preds, grads = _compute_predictions_and_gradients(interpolated_inputs, model, target_class_idx)
    grads = tf.reshape(grads, (steps + 1,) + grads.shape[1:])
    avg_grads = tf.reduce_mean((grads[:-1] + grads[1:]) / 2.0, axis=0)

    integrated_grads = (inputs - baseline) * avg_grads
    return integrated_grads[0]

def make_integrated_gradients_heatmap(img_array, model):
    preds = model(img_array, training=False).numpy()
    if preds.shape[-1] == 1:
        pred_index = int(preds[0, 0] > 0.5)
    else:
        pred_index = int(np.argmax(preds[0]))

    ig_attrib = integrated_gradients(
        inputs=tf.convert_to_tensor(img_array, dtype=tf.float32),
        model=model,
        target_class_idx=pred_index,
        baseline=None,
        steps=50,
    ).numpy()

    heatmap = np.mean(np.abs(ig_attrib), axis=-1)
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    return heatmap.astype(np.float32), pred_index

def overlay_heatmap_on_image(heatmap, image_path, alpha=0.4, colormap=cm.jet):
    img = Image.open(image_path).convert("RGB")
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_pil = Image.fromarray(heatmap_uint8).resize(img.size)
    colored = colormap(np.array(heatmap_pil) / 255.0)[:, :, :3]
    colored = Image.fromarray((colored * 255).astype("uint8")).resize(img.size)
    overlay = Image.blend(img, colored, alpha=alpha)
    return overlay

def explain_image(img_array, model):
    last_conv = try_get_last_conv_layer_name(model)
    if last_conv is not None:
        heatmap, pred_index = make_gradcam_heatmap(img_array, model, last_conv)
        return heatmap, pred_index, "gradcam"
    else:
        heatmap, pred_index = make_integrated_gradients_heatmap(img_array, model)
        return heatmap, pred_index, "ig"
