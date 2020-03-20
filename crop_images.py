from PIL import Image

im = Image.open("/home/robot/Pictures/RNN_recon_double.png")
im_crop = im.crop((470, 320, 720, 900))

# im_crop.show()
im_crop.save("/home/robot/Pictures/RNN_recon_double_crop.png")


inds = [0, 5, 10, 15, 20]

for ind in inds:
    im = Image.open("/home/robot/Pictures/interp_{}.png".format(ind))
    im_crop = im.crop((400, 320, 720, 900))

    im_crop.save("/home/robot/Pictures/interp_{}_crop.png".format(ind))

# im_crop.show()