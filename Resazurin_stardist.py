from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from stardist.models import StarDist2D
import cv2



path = 'C:\\Users\\Modern\\Documents\\Biorad\\Test\\test 2.tif'
# im.show()

img_test=cv2.imread(path,cv2.IMREAD_COLOR)

model = StarDist2D.from_pretrained('2D_versatile_fluo')

img = test_image_nuclei_2d()

print(type(img))
print(type(img_test))


labels, _ = model.predict_instances(normalize(img))

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("input image")

plt.subplot(1,2,2)
plt.imshow(render_label(labels, img=img))
plt.axis("off")
plt.title("prediction + input overlay")

plt.show()