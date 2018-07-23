import numpy
from math import pi, atan, sqrt, cos, sin
from PIL import Image


GAUSS_KERNEL = [
    [41, 26,  7,  7, 26],
    [24, 16,  4,  4, 16],
    [ 7,  4,  1,  1,  4],
    [ 7,  4,  1,  1,  4],
    [24, 16,  4,  4, 16]
]

SOBEL_KERNEL_X = [
    [0,2,-2],
    [0,1,-1],
    [0,1,-1]
]

SOBEL_KERNEL_Y = [
    [0, 0, 0],
    [2, 1, 1],
    [-2, -1, -1]
]


def sobel(image):
    mag = numpy.zeros(shape=image.shape)
    angle = numpy.zeros(shape=image.shape)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            acc_x = 0
            acc_y = 0
            for i in range (-1,2):
                for j in range (-1,2):
                    if (y + i >= 0
                            and x + j >= 0
                            and y + i < image.shape[0]
                            and x + j < image.shape[1]):
                        acc_x += SOBEL_KERNEL_X[i][j] * image[y+i][x+j]
                        acc_y += SOBEL_KERNEL_Y[i][j] * image[y+i][x+j]
            mag_x = 255*acc_x/2040.0
            mag_y = 255*acc_y/2040.0

            mag[y][x] = int(sqrt(mag_x**2+mag_y**2))
    return mag


def hough_transform(image):
    diagonal = sqrt(image.shape[0]**2 + image.shape[1]**2)

    output = numpy.zeros(shape=(int(diagonal), 360))

    d_theta = (2*pi)/720
    max_theta = 2*pi

    d_rho = 1.0/2.0
    max_rho = diagonal/2

    d_step = 1.0/2.0

    center_x = image.shape[0]/2.0
    center_y = image.shape[1]/2.0

    for rho in range (0, max_rho, d_rho):
        for theta in range (0, max_theta, d_theta):
            slope = atan((theta + pi/4)%(pi/2))
            x = center_x+rho*cos(theta)
            y = center_y+rho*sin(theta)
            d_x = 1/sqrt(1+slope**2)
            d_y = 1/sqrt(1+(1./slope)**2)


    return output


def gauss_blur(image):
    output = numpy.zeros(shape=image.shape)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            acc = 0
            count = 0
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if (y+i >= 0
                            and x+j >= 0
                            and y+i < image.shape[0]
                            and x+j < image.shape[1]):
                        acc += GAUSS_KERNEL[i][j] * image[y+i][x+j]
                        count += GAUSS_KERNEL[i][j]
            output[y][x] = int(acc/count)
    return output


def test():
    image = numpy.asarray(Image.open('test.jpg').convert('L'))
    image.setflags(write=True)

    save_image = Image.fromarray(numpy.uint8(image))
    save_image.save('before.png')
    image = gauss_blur(image)
    image = sobel(image)

    save_image = Image.fromarray(numpy.uint8(image))
    save_image.save('after.png')

    save_image = Image.fromarray(numpy.uint8(hough_transform(image)))
    save_image.save('hough.png')


if __name__ == '__main__':
    print "starting"
    test()
    print "done"


