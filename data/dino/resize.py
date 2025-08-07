import cv2

img = cv2.imread('/home/nml/projects/data/dino/test/mouse.png')
img = cv2.resize(img, (512, 512))

cv2.imwrite('mouse.png', img)