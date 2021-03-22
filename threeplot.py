# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from PIL import Image

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
imdir='./tmp/ssdwd0.001 wd0.005 ssdn30/predict979/misclassified/'
imga=imdir+'t0p1pb0.6653308272361755.png'
imgb=imdir+'t0p1pb0.8091893792152405.png'
imgc=imdir+'t1p0pb0.33110538125038147.png'
imgd=imdir+'t1p0pb0.3848233222961426.png'


X = np.arange(0, 33, 1)
Y = np.arange(0, 33, 1)
X, Y = np.meshgrid(X, Y)

Za=np.array(Image.open(imga).transpose(Image.FLIP_TOP_BOTTOM).convert('L'))
Zb=np.array(Image.open(imgb).transpose(Image.FLIP_TOP_BOTTOM).convert('L'))
Zc=np.array(Image.open(imgc).transpose(Image.FLIP_TOP_BOTTOM).convert('L'))
Zd=np.array(Image.open(imgd).transpose(Image.FLIP_TOP_BOTTOM).convert('L'))

# Plot the surface.
#surf = ax.plot_surface(X, Y, Za, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.plot_wireframe(X, Y, Zd, rstride=2, cstride=2)
# Customize the z axis.
ax.set_zlim(0, 255)
ax.zaxis.set_major_locator(LinearLocator(5))
ax.zaxis.set_major_formatter(FormatStrFormatter('%d'))

# Add a color bar which maps values to colors.


plt.show()
#import visdom
#vis=visdom.Visdom()
#vis.surf(Za)
#vis.image(Za)