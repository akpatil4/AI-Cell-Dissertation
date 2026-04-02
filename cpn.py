import torch, cv2, celldetection as cd
import os
import matplotlib.pyplot as plt

# Load pre-trained CPN model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = cd.fetch_model('ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c', check_hash=True).to(device)
model.eval()

# Load and preprocess image
image_name = 'Image2.png'
image = cv2.imread(image_name)
print(image.dtype, image.shape, (image.min(), image.max()))

# Run model
with torch.no_grad():
    x = cd.to_tensor(image, transpose=True, device=device, dtype=torch.float32)
    x = x / 255  # ensure 0..1 range
    x = x[None]  # add batch dimension: Tensor[3, h, w] -> Tensor[1, 3, h, w]
    y = model(x)

# Show results for each batch item
contours = y['contours']
for n in range(len(x)):
    cd.imshow_row(x[n], x[n], figsize=(16, 9), titles=('input', 'contours'))
    cd.plot_contours(contours[n])
    plt.show()
