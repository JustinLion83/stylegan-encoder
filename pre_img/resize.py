from PIL import Image
import sys


fn = sys.argv[1]
im = Image.open(fn)
scale = 1200 / float(min(im.width, im.height))
new_width, new_height = int(scale * im.width), int(scale * im.height)
im = im.resize((new_width, new_height))
im.save('resized_' + fn)

