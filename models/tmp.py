import torch
import torchvision.transforms as tvt
import numpy

images = torch.randint(0, 256, (4, 3, 5, 9)).type(torch.uint8)
print(images[0])
print(images.shape)
print(images.max())
max_pixel_locations = (images==images.max()).nonzero()
print(max_pixel_locations)
min_pixel_locations = (images == images.min()).nonzero()
print(min_pixel_locations)

images_showing_min_pixels = (images == images.min())
print(images_showing_min_pixels)

images_scaled = images / images.max().float()
print(images_scaled[0])

images_scaled = torch.zeros_like(images).float()

for i in range(images.shape[0]):
    images_scaled[i] = tvt.ToTensor()(tvt.ToPILImage()(images[i]))
# for i in range(images.shape[0]):
#     images_scaled[i] = tvt.ToTensor()(numpy.transpose(images[i].numpy(), (1, 2, 0)))
print(images_scaled[0])

xform1 = tvt.Compose([tvt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
images_normed = torch.zeros_like(images).float()
for i in range(images.shape[0]):
    images_normed[i] = xform1(images_scaled[i])
print(images_normed[0])

xform2 = tvt.Compose([tvt.ToPILImage(),
                      tvt.ToTensor(),
                      tvt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
images_normed2 = torch.zeros_like(images).float()
for i in range(images.shape[0]):
    images_normed2[i] = xform2(images[i])
print(images_normed2[0])