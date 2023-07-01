import matplotlib.pyplot as plt
import tensorflow as tf


def vis_vec(v, axis1=0, axis2=2, label=''):
    # This function visualizes a vector in a 2D plot
    # Input:
    #   v: The vector to be visualized
    #   axis1: The index of the first axis for plotting (default is 0)
    #   axis2: The index of the second axis for plotting (default is 2)
    #   label: Optional label for the vector (default is an empty string)

    # Plotting the vector using matplotlib
    plt.plot([0, v[axis1]], [0, v[axis2]], '-o', label=label)


def block(mat):
    rows = []
    for row in mat:
        rows.append(tf.concat(row, axis=1))
    block_matrix = tf.concat(rows, axis=0)
    return block_matrix


