import numpy as np
from PIL import Image
import click
import cv2
from deblurgan.model import generator_model
from deblurgan.utils import load_images, deprocess_image


def test(batch_size):
    PSNR = 0
    data = load_images('./images/test', batch_size)
    y_test, x_test = data['B'], data['A']
    g = generator_model()
    g.load_weights('generator.h5')
    generated_images = g.predict(x=x_test, batch_size=batch_size)
    generated = np.array([deprocess_image(img) for img in generated_images])
    x_test = deprocess_image(x_test)
    y_test = deprocess_image(y_test)

    for i in range(generated_images.shape[0]):
        y = y_test[i, :, :, :]
        x = x_test[i, :, :, :]
        img = generated[i, :, :, :]
        img1 = Image.fromarray(img.astype(np.uint8))
        img1.save("img1.png")
        img1 = cv2.imread("img1.png")
        img2 = Image.fromarray(y.astype(np.uint8))
        img2.save("img2.png")
        img2 = cv2.imread("img2.png")
        #img2 = cv2.resize(img2, (256,256))
        psnr = cv2.PSNR(img1, img2)
        PSNR += psnr
        #print "The PSNR is : ", psnr
        output = np.concatenate((y, x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save('results{}.png'.format(i))

    PSNR /= batch_size
    print "The average PSNR is" , PSNR

@click.command()
@click.option('--batch_size', default=3, help='Number of images to process')
def test_command(batch_size):
    return test(batch_size)


if __name__ == "__main__":
    test_command()
