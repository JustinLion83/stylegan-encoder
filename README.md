## StyleGAN &mdash; Encoder for Official TensorFlow Implementation
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![TensorFlow 1.10](https://img.shields.io/badge/tensorflow-1.10-green.svg?style=plastic)
![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-7.3.1-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-green.svg?style=plastic)

Welcome to KUNGFU.AI's 2019 SXSW StyleGAN demo.
First off much thanks to Dmitry Nikitko ([Puzer](https://github.com/Puzer)) from whom we forked the repo. 
Our demo is mostly focused in the jupyter notebook "StyleGAN demo.ipynb." You might need to edit "aei.py" and point it to the correct directory. 
Latent codes for a few celebrities along with some supremely handsome KUNGFU.AI employees can be found in /latent_codes (shocker), and the previews of them can be seen in img_gen. 
If you have questions, concerns, criticisms, or jokes, please open a pull request or issue! Even just adding latent codes would be very welcomed. 

The following text is Puzer's README explaining latent code recovery. 

![Teaser image](./teaser.png)

*These people are real &ndash; latent representation of them was found by using perceptual loss trick. Then this representations were moved along "smiling direction" and transformed back into images*

Short explanation of encoding approach:
0) Original pre-trained StyleGAN generator is used for generating images
1) Pre-trained VGG16 network is used for transforming a reference image and generated image into high-level features space
2) Loss is calculated as a difference between them in the features space
3) Optimization is performed only for latent representation which we want to obtain. 
4) Upon completion of optimization you are able to transform your latent vector as you wish. For example you can find a "smiling direction" in your latent space, move your latent vector in this direction and transform it back to image using the generator. 

**New scripts for finding your own directions will be realised soon. For now you can play with existing ones: smiling, age, gender.**
**More examples you can find in the [Jupyter notebook](https://github.com/Puzer/stylegan/blob/master/Play_with_latent_directions.ipynb)**

### Generating latent representation of your images
You can generate latent representations of your own images using two scripts:
1) Extract and align faces from images
> python align_images.py raw_images/ aligned_images/

2) Find latent representation of aligned images
> python encode_images.py aligned_images/ generated_images/ latent_representations/

3) Then you can play with [Jupyter notebook](https://github.com/Puzer/stylegan/blob/master/Play_with_latent_directions.ipynb)

Feel free to join the research. There is still much room for improvement:
1) Better model for perceptual loss
2) Is it possible to generate latent representations by using other model instead of direct optimization ? (WIP)

Stay tuned!

