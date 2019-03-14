import os
import sys
import bz2
import argparse
import glob
from tqdm import tqdm
import dnnlib
import dnnlib.tflib as tflib
import config
import pickle
import PIL.Image
import numpy as np
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel

from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def align_and_encode_latest_img(src_dir, aligned_dir, generated_images_dir, dlatent_dir, generator, Gs_network, img_name):
    """
    Takes the most recent image in a dir, aligns it, then encodes it. After encoding, saves the latent code and a generated image
    python align_and_encode_image.py /raw_images /aligned_images /generated_images /latent_codes
    """
    URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'

    LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))

    image_size = 256
    lr = 2 
    iterations = 1000
    randomize_noise = False

    landmarks_detector = LandmarksDetector(landmarks_model_path)
    
    # Get most recent file in src_dir
    list_of_files = glob.glob(src_dir + '*')
    raw_img_path = max(list_of_files, key=os.path.getctime)
    face_landmarks = landmarks_detector.get_landmarks(raw_img_path)
    face_img_name = '%s_aligned.png' % (img_name)
    
    #check if face image already exists - if so continue
    aligned_face_path = os.path.join(aligned_dir, face_img_name)
    
    if os.path.isfile(aligned_face_path):
        print('File already exists:', aligned_face_path)
    
    print('raw_img_path ', raw_img_path)
    print('aligned_face_path ', aligned_face_path)
    print('face_landmarks',face_landmarks)

    image_align(raw_img_path, aligned_face_path, next(face_landmarks))

    print('done aligning')

    # Get most recently saved file in aligned_files
    list_of_files = glob.glob(aligned_face_path)
    print(list_of_files)
    aligned_img_path = max(list_of_files, key=os.path.getctime)
    print('aligned_img_path ', aligned_img_path)
    #ref_images = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir)]
    #ref_images = list(filter(os.path.isfile, ref_images))

    #if len(ref_images) == 0:
    #    raise Exception('%s is empty' % args.src_dir)

    #os.makedirs(args.generated_images_dir, exist_ok=True)
    #os.makedirs(args.dlatent_dir, exist_ok=True)

    # Initialize generator and perceptual model
    tflib.init_tf()

    print('imported network')
    perceptual_model = PerceptualModel(image_size, layer=9, batch_size=1)
    perceptual_model.build_perceptual_model(generator.generated_image)

    print('begin optimizing')
    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    perceptual_model.set_reference_images([aligned_face_path])
    op = perceptual_model.optimize(generator.dlatent_variable, iterations=iterations, learning_rate=lr)
    pbar = tqdm(op, leave=False, total=iterations)
    for loss in pbar:
        pbar.set_description('Image Loss: '+' Loss: %.2f' % loss)
    print(' loss:', loss)

    # Generate images from found dlatents and save them
    generated_image = generator.generate_images()[0]
    generated_dlatent = generator.get_dlatents()[0]

    img = PIL.Image.fromarray(generated_image, 'RGB')
    img.save(os.path.join(generated_images_dir, f'{img_name}.png'), 'PNG')
    np.save(os.path.join(dlatent_dir, f'{img_name}.npy'), generated_dlatent)

    generator.reset_dlatents()

    return generated_dlatent 
