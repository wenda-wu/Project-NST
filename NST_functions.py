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


### Create the content cost function
def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of content_image
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of style_image
    
    Returns:
    J_content -- scalar that you compute as cost function value
    """
    # Retrieve dimensions from a_G
    n, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G
    a_C_unrolled = tf.transpose(tf.reshape(a_C, (n_H*n_W, n_C)))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, (n_H*n_W, n_C)))
    
    # Compute the cost with tensorflow
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (4*n_H*n_W*n_C)
    
    return J_content


### Compute the gram matrix
def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    GA = tf.matmul(A, tf.transpose(A))
    
    return GA


### Define Function: compute_layer_style_cost
def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the given image
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the output image
    
    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined by equation
    """
    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W)
    a_S = tf.transpose(tf.reshape(a_S, (n_H*n_W, n_C)))
    a_G = tf.transpose(tf.reshape(a_G, (n_H*n_W, n_C)))
    
    # Compute gram_matrices fo both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    
    # Compute the loss
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) / (4*n_C*n_C*(n_H*n_W)*(n_H*n_W))
    
    return J_style_layer


### Define the total cost

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded already
    J_style -- style cost coded already
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula
    """
    
    J = alpha * J_content + beta * J_style
    
    return J
