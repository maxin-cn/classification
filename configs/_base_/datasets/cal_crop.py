# imagenet cropping 
# refer: https://github.com/tensorflow/tpu/blob/04f6bb6da1502c3551ed4500a4ee396e06243561/models/official/efficientnet/preprocessing.py#L88
CROP_FRACTION = 0.875
image_sizes = [224, 240, 260, 300, 380, 456, 528, 600, 672]
for image_size in image_sizes:
    crop_padding = round(image_size * (1/CROP_FRACTION - 1))
    print(image_size + crop_padding, image_size)
     

