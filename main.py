import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import time


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # return None, None, None, None, None
    model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    image_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
    
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    #return None
    
    # To build the decoder portion of FCN-8, we’ll upsample the input to the original image size.  
    # The shape of the tensor after the final convolutional transpose layer will be 4-dimensional:
    # (batch_size, original_height, original_width, num_classes).
    
    # FCN-7
    fcn_layer_7 = tf.layers.conv2d(vgg_layer7_out, num_classes,  kernel_size=1, strides=(1,1), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), name='fcn_layer_7')
    
    # DCONV-7
    fcn_dconv_7 = tf.layers.conv2d_transpose(fcn_layer_7, num_classes, kernel_size=4, strides=(2,2), padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), name='fcn_dconc_7')

    # FCN-4
    fcn_layer_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=1, strides=(1,1), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), name='fcn_layer_4')

    # SKIP LAYER-4
    skip_layer_4 = tf.add(fcn_dconv_7, fcn_layer_4)

    # DCONV-4
    fcn_dconv_4 =tf.layers.conv2d_transpose(skip_layer_4, num_classes, kernel_size=4, strides=(2,2), padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), name='fcn_dconv_4')    

    # FCN-3
    fcn_layer_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=1, strides=(1,1), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), name='fcn_layer_3')

    # SKIP LAYER-3
    skip_layer_3 = tf.add(fcn_dconv_4, fcn_layer_3)

    # DCONV-3
    fcn_dconv_3 =tf.layers.conv2d_transpose(skip_layer_3, num_classes, kernel_size=16, strides=(8,8), padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), name='fcn_dconv_3')

    return fcn_dconv_3

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # return None, None, None

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return logits, train_op, loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    outFile = open("output.txt", 'a')
    print("Training ..." + "\n")
    outFile.write("Training..."+"\n")

    start_time = time.clock()
    
    for epoch in range(epochs):
        batches = get_batches_fn(batch_size)
      
        for batch_input, batch_label in batches:
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: batch_input, correct_label: batch_label, keep_prob: 0.5, learning_rate: 1e-4})
  
        end_time = time.clock()
        train_time = end_time-start_time
        print("Epoch: {}/{} | Execution Time: {} sec | Loss: {}".format(epoch, epochs, train_time, loss))

        outFile.write("Epoch: {}/{} | Execution Time: {} sec | Loss: {}".format(epoch, epochs, train_time, loss))

    outFile.close()
    pass
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    epochs = 25
    batch_size = 16

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.int32)
        learning_rate = tf.placeholder(tf.float32)

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, loss = optimize(last_layer, correct_label, learning_rate, num_classes)


        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss, input_image, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
