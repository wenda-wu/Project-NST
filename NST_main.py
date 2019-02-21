import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from NST_utils import *
from NST_functions import *
import numpy as np
import tensorflow as tf

#%matplotlib inline

# Check on the model
model = load_vgg_model("imagenet-vgg-verydeep-19.mat")
print(model)

# Show the content image and style image
content_image = scipy.misc.imread("images/Dalian_600x375.jpg")
#imshow(content_image)
print("Shape of Dalian night view:",content_image.shape)
style_image = scipy.misc.imread("images/StarSky_600x375.jpg")
#imshow(style_image)
print("Shape of Van Gough star sky:", style_image.shape)


STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)
]


# Start the interactive session
sess = tf.InteractiveSession()

### Compute combined style cost
def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- the tensorflow model
    STYLE_LAYERS -- A python list containing:
                    - the names of the layers we would like to extract style from 
                    - a coefficient for each of them
    Returns:
    J_style -- tensor representing a scalar value, style cost defined by equation
    """
    
    # initialize the overall style cost
    J_style =0
    
    for layer_name, coeff in STYLE_LAYERS:
        
        # Selsect the output tensor of the currently selected layer
        out = model[layer_name]
        
        # Set a_S to be the hidden layer activation from the layer we have selected
        a_S = sess.run(out)
        
        # Set a_G to be the hidden layer activation from same layer.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        
        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer
        
    return J_style


# Process images
content_image = scipy.misc.imread("images/Dalian_600x375.jpg")
content_image = reshape_and_normalize_image(content_image)

style_image = scipy.misc.imread("images/StarSky_600x375.jpg")
style_image = reshape_and_normalize_image(style_image)

generated_image = generate_noise_image(content_image)
'''imshow(generated_image[0])'''# To see the randomly initialized image, comment this in

# Assign the content image to be the input of the VGG model
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from the same layer
a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)

# Assign the input of the model to be the "style" image
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)

# Compute the total cost
J = total_cost(J_content, J_style, alpha = 10, beta = 40)

# Define optimizer
optimizer = tf.train.AdamOptimizer(2.0)

# Define train_step
train_step = optimizer.minimize(J)


### Implementation of the model_nn function
def model_nn(sess, input_image, num_iterations = 300):
    
    # Initialize global variables
    sess.run(tf.global_variables_initializer())
    
    # Run the noisy input image
    sess.run(model['input'].assign(input_image))
    
    # Start the for loop
    for i in range(num_iterations):
        
        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)
        
        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])
        
        # Print every 20 iteration.
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration" + str(i) + ":")
            print("total cost =" + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # Save current generated image in the "/output" directory
            save_image("output/" + str(i) + "DL.png", generated_image)
    # Save the last generated image
    save_image('output/Dalian_VGstyle.jpg', generated_image)
    
    return generated_image

### Run the model
model_nn(sess, generated_image)
