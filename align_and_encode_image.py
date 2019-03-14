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


if __name__ == "__main__":
    """
    Takes the most recent image in a dir, aligns it, then encodes it. After encoding, saves the latent code and a generated image
    python align_and_encode_image.py /raw_images /aligned_images /generated_images /latent_codes
    """
    URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'

    LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))

    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    parser.add_argument('src_dir', help='Directory with images for encoding')
    parser.add_argument('aligned_dir', help='Directory to save aligned image')
    parser.add_argument('generated_images_dir', help='Directory for storing generated images')
    parser.add_argument('dlatent_dir', help='Directory for storing dlatent representations')
    parser.add_argument('img_name', help='File name for saved images')

    # Perceptual model params
    parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    parser.add_argument('--lr', default=1., help='Learning rate for perceptual model', type=float)
    parser.add_argument('--iterations', default=1000, help='Number of optimization steps for each batch', type=int)

    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=bool)
    args, other_args = parser.parse_known_args()

    landmarks_detector = LandmarksDetector(landmarks_model_path)
    
    # Get most recent file in src_dir
    list_of_files = glob.glob(args.src_dir + '*')
    raw_img_path = max(list_of_files, key=os.path.getctime)
    face_landmarks = landmarks_detector.get_landmarks(raw_img_path)
    face_img_name = '%s_aligned.png' % (args.img_name)
    
    #check if face image already exists - if so continue
    aligned_face_path = os.path.join(args.aligned_dir, face_img_name)
    
    if os.path.isfile(aligned_face_path):
        print('File already exists:', aligned_face_path)
    
    print('raw_img_path ', raw_img_path, type(raw_img_path))
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
    with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    print('imported network')
    generator = Generator(Gs_network, 1, randomize_noise=args.randomize_noise)
    perceptual_model = PerceptualModel(args.image_size, layer=9, batch_size=1)
    perceptual_model.build_perceptual_model(generator.generated_image)

    print('begin optimizing')
    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    perceptual_model.set_reference_images([aligned_face_path])
    op = perceptual_model.optimize(generator.dlatent_variable, iterations=args.iterations, learning_rate=args.lr)
    pbar = tqdm(op, leave=False, total=args.iterations)
    for loss in pbar:
        pbar.set_description('Image Loss: '+' Loss: %.2f' % loss)
    print(' loss:', loss)

    # Generate images from found dlatents and save them
    generated_images = generator.generate_images()
    generated_dlatents = generator.get_dlatents()
    print(generated_images.shape)
    for img_array, dlatent in zip(generated_images, generated_dlatents):
        img = PIL.Image.fromarray(img_array, 'RGB')
        img.save(os.path.join(args.generated_images_dir, f'{args.img_name}.png'), 'PNG')
        np.save(os.path.join(args.dlatent_dir, f'{args.img_name}.npy'), dlatent)

    generator.reset_dlatents()
