import cv2
import jax, functools 
from jax import numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jax.scipy import signal
from jax.image import resize

def rgb_to_greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def sepfir2(image, hx, hy):

    c1d = functools.partial( signal.convolve, mode='same' )

    convolve1d_x = jax.vmap( c1d, in_axes=(0, None) )
    convolve1d_y = jax.vmap( c1d, in_axes=(1, None) )

    temp1 = convolve1d_x(image, hx)
    temp2 = convolve1d_y(temp1, hy).T

    return temp2

def second_derivatives(image):

    p = jnp.array([ 0.030320,  0.249724, 0.439911, 0.249724, 0.030320])
    d = jnp.array([0.471147, -0.002668, -0.232905, -0.002668, 0.471147])

    img_xx = sepfir2( image, d, p )
    img_yy = sepfir2( image, p, d )

    return img_xx, img_yy

def first_derivatives(image):
    '''directional derivative in the x and y directions

    Parameters
    ----------
    image : ArrayLike
        a 2D image 

    Returns
    -------
    _type_
        _description_
    '''

    p = jnp.array([ 0.030320,  0.249724, 0.439911, 0.249724, 0.030320])
    d = jnp.array([-0.104550, -0.292315, 0,        0.292315, 0.104550])

    img_x = sepfir2( image, d, p )
    img_y = sepfir2( image, p, d )

    return img_x, img_y

def line_detection(image):

    p = jnp.array([ 0.030320,  0.249724, 0.439911, 0.249724, 0.030320])
    d = jnp.array([-0.104550, -0.292315, 0,        0.292315, 0.104550])

    img_x = sepfir2( image, d, p )
    img_y = sepfir2( image, p, d )

    return img_x, img_y

def gaussian_pyramids(image:ArrayLike, blur_filter:Array, number_of_pyramids:int) -> list[Array]:

    pyramids = [ image ] # add the first Pyramids here
    for _ in range(1, number_of_pyramids):
        image_temp = []
        for channel in range(3):
            image_temp.append( sepfir2(image[:,:,channel], blur_filter, blur_filter) )
        image_temp = jnp.stack(image_temp, axis=2)
        image_temp = image_temp.astype(jnp.uint8)
        image_temp = image_temp[0:-1:2, 0:-1:2, :]
        pyramids.append( image_temp )
        image = image_temp.copy()
        del(image_temp)

    return pyramids 

def laplacian_pyramids(image:ArrayLike, blur_filter:Array, number_of_pyramids:int) -> tuple[list[Array]]:

    gaussian_pyramid = gaussian_pyramids(image, blur_filter, number_of_pyramids)
    laplacian_pyramid = [ gaussian_pyramid[-1].copy() ]
    N = len(gaussian_pyramid)
    for i in range( N - 1 ):
        im_small = gaussian_pyramid[ N-1-i ].copy()
        im_big   = gaussian_pyramid[ N-1-i-1 ].copy()
        new_big  = resize( im_small, im_big.shape, 'cubic' )
        laplacian_pyramid.insert( 0, im_big - new_big )

    return gaussian_pyramid, laplacian_pyramid