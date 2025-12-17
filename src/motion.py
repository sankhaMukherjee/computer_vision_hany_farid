import jax 
from jax import numpy as jnp
from src import img_utils

@jax.jit
def find_velocity(arr):

    M = jnp.array(jnp.vstack([arr[2:4], arr[3:5]])).copy()
    b = arr.copy()[-2:].reshape((2, 1))

    cond1 = jnp.linalg.norm(arr[:2]) < 2
    cond2 = jnp.linalg.cond( M ) > 1e2
    cond = cond1 | cond2

    result = jax.lax.cond(
        cond, 
        lambda M, b: jnp.zeros((2, 1)),
        lambda M, b: -jnp.linalg.inv(M) @ b,
        M, b
    )

    return result

@jax.jit
def velocities_from_video(video_frames):
    p        = jnp.array([0.5, 0.5])
    d        = jnp.array([-1, 1])
    mean_vid = 0.5*video_frames[0] + 0.5*video_frames[1]

    fx = img_utils.sepfir2( mean_vid, d, p )
    fy = img_utils.sepfir2( mean_vid, p, d )
    ft = img_utils.sepfir2( video_frames[0] - video_frames[1], p, p)

    h   = jnp.array([1/16, 4/16, 6/16, 4/16, 1/16])
    fx2 = img_utils.sepfir2( fx*fx, h, h )
    fy2 = img_utils.sepfir2( fy*fy, h, h )
    fxy = img_utils.sepfir2( fx*fy, h, h )
    fxt = img_utils.sepfir2( fx*ft, h, h )
    fyt = img_utils.sepfir2( fy*ft, h, h )

    combined = jnp.array([ fx.flatten(), fy.flatten(), fx2.flatten(), fxy.flatten(), fy2.flatten(), fxt.flatten(), fyt.flatten() ])
    velocity = jax.vmap(find_velocity, in_axes=(1))(combined)
    velocity = velocity.reshape(velocity.shape[0], 2)
    
    image_shape = fx2.shape
    vx = velocity[:, 0].reshape(image_shape)
    vy = velocity[:, 1].reshape(image_shape)

    return vx, vy

