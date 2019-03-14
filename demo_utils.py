import os
import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
import imageio
import glob
from encoder.generator_model import Generator

import matplotlib.pyplot as plt

def generate_image(latent_vector, generator):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img.resize((256, 256))

def move_and_show(latent_vector, direction, coeffs, generator):
    fig,ax = plt.subplots(1, len(coeffs), figsize=(15, 10), dpi=80)
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
        ax[i].imshow(generate_image(new_latent_vector, generator))
        ax[i].set_title('Coeff: %0.1f' % coeff)
    [x.axis('off') for x in ax]
    plt.show()

def save_image(latent_vector, direction, coeffs, generator):
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
        img = generate_image(new_latent_vector, generator)
        img.save(f'print_img/post_img_{i}.jpg')
    print('Images saved to /print_img')

def move_and_show_interactive(generator, latent_vector, direction, coeff):
    fig,ax = plt.subplots(1, 1, figsize=(15, 10), dpi=80)
    
    new_latent_vector = latent_vector.copy()
    new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
    img = generate_image(new_latent_vector, generator)
    #print(type(img))
    ax.imshow(img)
    ax.axis('off')
    plt.show()

def mix_two_styles(src_latent, dst_latent, generator, Gs_network):
    
    fig,ax = plt.subplots(1, 3, figsize=(15, 10), dpi=80)
    # Draw the constructed image of the src, the dest, and turn off the labels
    src_img = generate_image(src_latent, generator)
    dst_img = generate_image(dst_latent, generator)
    ax[0].imshow(src_img)
    ax[2].imshow(dst_img)
    [x.axis('off') for x in ax]
   
    # Now construct the mixed image
    #print(src_latent.shape)
    src_latent_copy = src_latent.copy()
    src_latent_copy[range(0,4), :] = dst_latent[range(0,4), :]
    src_latent_copy = src_latent_copy.reshape((1, 18, 512))
    
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True) # needed for output_transform - defined in nvidia's code
    mix_array = Gs_network.components.synthesis.run(src_latent_copy, randomize_noise=False, output_transform=fmt)
    #mix_img = mix_img.squeze()
    #rint(mix_array[0]- src_array)

    ax[1].imshow(PIL.Image.fromarray(mix_array[0], 'RGB').resize((256,256)))

def mix_with_celeb_styles(src_latent, celebs, Gs_network, generator):
    fig, ax =  plt.subplots(len(celebs), 5, figsize=(15, 10), dpi=80)
    src_img = generate_image(src_latent, generator)
    celeb_pics = [generate_image(lat_var, generator) for lat_var in celebs]
    #ax[0][0].axis('off') #I'm sure this can be done better and nicely with numpy
    
#     # Display top row of celeb pics
#     for i in range(1, len(ax[0])):
#         ax[0][i].imshow(celeb_pics[i-1])
#         ax[0][i].axis('off')
    # put source image first in each row

    for row in range(len(ax)):
        ax[row][0].imshow(src_img)
        ax[row][0].axis('off')
        
        src_latent_copy = src_latent.copy()
        src_latent_copy[range(8,18), :] = celebs[row][range(8,18), :]
        src_latent_copy = src_latent_copy.reshape((1, 18, 512))
        
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True) # needed for output_transform - defined in nvidia's code
        mix_array = Gs_network.components.synthesis.run(src_latent_copy, randomize_noise=False, output_transform=fmt)
        
        ax[row][1].imshow(PIL.Image.fromarray(mix_array[0], 'RGB').resize((256,256)))
        ax[row][1].axis('off')
        
        src_latent_copy = src_latent.copy()
        src_latent_copy[range(4,8), :] = celebs[row][range(4,8), :]
        src_latent_copy = src_latent_copy.reshape((1, 18, 512))
        
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True) # needed for output_transform - defined in nvidia's code
        mix_array = Gs_network.components.synthesis.run(src_latent_copy, randomize_noise=False, output_transform=fmt)
        
        ax[row][2].imshow(PIL.Image.fromarray(mix_array[0], 'RGB').resize((256,256)))
        ax[row][2].axis('off')
        
        src_latent_copy = src_latent.copy()
        src_latent_copy[range(0,4), :] = celebs[row][range(0,4), :]
        src_latent_copy = src_latent_copy.reshape((1, 18, 512))
        
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True) # needed for output_transform - defined in nvidia's code
        mix_array = Gs_network.components.synthesis.run(src_latent_copy, randomize_noise=False, output_transform=fmt)
        
        ax[row][3].imshow(PIL.Image.fromarray(mix_array[0], 'RGB').resize((256,256)))
        ax[row][3].axis('off')
        
        ax[row][4].imshow(celeb_pics[row])
        ax[row][4].axis('off')
        
def interactively_mix_two_styles(src_latent, dst_latent, range_tuple, generator, Gs_network):
    fig,ax = plt.subplots(1, 3, figsize=(15, 10), dpi=80)
    # Draw the constructed image of the src, the dest, and turn off the labels
    src_img = generate_image(src_latent, generator)
    dst_img = generate_image(dst_latent, generator)
    ax[0].imshow(src_img)
    ax[2].imshow(dst_img)
    [x.axis('off') for x in ax]
   
    # Now construct the mixed image
    #print(src_latent.shape)
    src_latent_copy = src_latent.copy()
    src_latent_copy[range(range_tuple[0],range_tuple[1]), :] = dst_latent[range(range_tuple[0],range_tuple[1]), :]
    src_latent_copy = src_latent_copy.reshape((1, 18, 512))
    
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True) # needed for output_transform - defined in nvidia's code
    mix_array = Gs_network.components.synthesis.run(src_latent_copy, randomize_noise=False, output_transform=fmt)
    #mix_img = mix_img.squeze()
    #rint(mix_array[0]- src_array)

    ax[1].imshow(PIL.Image.fromarray(mix_array[0], 'RGB').resize((256,256)))
    ax[1].set_title('Range %d - %d' % (range_tuple[0], range_tuple[1]))


def draw_truncation_trick(dlatent, psis, Gs_network):
    dlatent_avg = Gs_network.get_var('dlatent_avg') # [component]
    fig,ax = plt.subplots(1, len(psis), figsize=(15, 10), dpi=80)
   
    row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(psis, [-1, 1, 1]) + dlatent_avg
    synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=1) # needed for gs_network
    row_images = Gs_network.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)

    for col, image in enumerate(list(row_images)):
        ax[col].imshow(image)
    [x.axis('off') for x in ax]
    plt.show()

def interact_with_truncation_trick(dlatent, Gs_network, coeff):
    dlatent_avg = Gs_network.get_var('dlatent_avg') # [component]
    fig,ax = plt.subplots(1, 1, figsize=(15, 10), dpi=80)
   
    row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * coeff + dlatent_avg
    synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=1) # needed for gs_network
    row_images = Gs_network.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
    ax.imshow(row_images[0])
    ax.axis('off')
    plt.show()

def draw_mbj_truncation_trick(coeff):
    dlatent_avg = Gs_network.get_var('dlatent_avg') # [component]
    fig,ax = plt.subplots(1, 1, figsize=(15, 10), dpi=80)
   
    row_dlatents = (mbj[np.newaxis] - dlatent_avg) * coeff + dlatent_avg
    synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=1) # needed for gs_network
    row_images = Gs_network.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
    ax.imshow(row_images[0])
    ax.axis('off')
    plt.show()

def frange(x, y, jump):
    arr = []
    while x < y:
        arr.append(x)
        x += jump
    return arr

def create_truncation_gif(dlatent):
    dlatent_avg = Gs_network.get_var('dlatent_avg') # [component]

    row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(frange(-1, 1, .1), [-1, 1, 1]) + dlatent_avg
    synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=1) # needed for gs_network
    row_images = Gs_network.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)

    for i, image_array in enumerate(list(row_images)):
        im = PIL.Image.fromarray(image_array)
        im.save("gif_files/gif_img_frame_%0.2d.jpeg" % i)

    with imageio.get_writer('truncation1.gif', mode='I') as writer:
        filenames = glob.glob('gif_files/*')
        filenames = sorted(filenames)
        #print(filenames)
        last = -1
        for i,filename in enumerate(filenames):
#             frame = 2*(i**0.5)
#             if round(frame) > round(last):
#                 last = frame
#             else:
#                 continue
            image = imageio.imread(filename)
            writer.append_data(image)
            if i == len(filenames)-1:
                writer.append_data(image)
                
        for i, filename in enumerate(reversed(filenames)):
            image = imageio.imread(filename)
            writer.append_data(image)
            if i == len(filenames)-1:
                writer.append_data(image)


def draw_one_truncation_trick(dlatent, coeff):
    dlatent_avg = Gs_network.get_var('dlatent_avg') # [component]
    fig,ax = plt.subplots(1, 1, figsize=(15, 10), dpi=80)
   
    row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * coeff + dlatent_avg
    synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=1) # needed for gs_network
    row_images = Gs_network.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
    ax.imshow(row_images[0])
#     ax.axis('off')
#     for col, image in enumerate(list(row_images)):
#         ax[col].imshow(image)
#     [x.axis('off') for x in ax]
    plt.show()

def draw_random_face(rndState, Gs_network):
    rnd = np.random.RandomState(rndState)
    latents = rnd.randn(1, Gs_network.input_shape[1])
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    image = Gs_network.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    img = PIL.Image.fromarray(image[0], 'RGB').resize((256,256))
    
    fig,ax = plt.subplots(1, 1, figsize=(15, 10), dpi=80)
    ax.imshow(img)
    ax.axis('off')
    plt.show()
