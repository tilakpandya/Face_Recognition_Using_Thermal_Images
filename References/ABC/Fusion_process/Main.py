import os
import cv2
import glob
import numpy as np
import matplotlib.image as mimg
import matplotlib.pyplot as plt
from Faces import Faces
from FuseImagesSWT import FuseImagesSWT


def main():
    global print_images, write_images

    img_path = os.path.join('.', 'Images')
    output_path = os.path.join('.', 'Images', 'Fused')
    faces = Faces(verbose=False, boundary_size_px=60)
    fuse_obj = FuseImagesSWT(level=2)
    np_black = np.zeros((540, 960), np.uint8)

    for root, dirs, files in os.walk(img_path):
        if output_path in root:
            continue

        if not files:
            continue

        print(root)

        files = glob.glob(os.path.join(root, '*.jpg'))

        # This dataset is symmetric, so IR images are at the beginning
        # followed by regular colored images at mid point
        mid_idx = int(len(files) / 2)
        idx = range(0, mid_idx)

        for i in idx:
            ir_img = mimg.imread(os.path.abspath(files[i]))
            vi_img = mimg.imread(os.path.abspath(files[i + mid_idx]))
            vi_img = vi_img[105:530, 150:730]

            (h, w, c) = vi_img.shape
            print(h, w, c)

            new_h = ir_img.shape[0]
            new_w = ir_img.shape[1]
            dim = (new_w, new_h)
            vi_img = cv2.resize(vi_img, dim, interpolation=cv2.INTER_AREA)

            (h, w, c) = vi_img.shape
            print(h, w, c)

            if print_images:
                fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))

                ax = axs[0, 0]
                ax.set_title('Thermal Image Original')
                ax.imshow(ir_img)
                ax.axis('off')

                ax = axs[0, 1]
                ax.set_title('Regular Image Original')
                ax.imshow(vi_img)
                ax.axis('off')

                ax = axs[0, 2]
                ax.imshow(np_black, cmap='gray', vmin=0, vmax=1)
                ax.axis('off')

            success, results = faces.detect_face(ir_img)

            if success:
                success, ann_ir_img = faces.draw_face(ir_img, results)

                if success and print_images:
                    ax = axs[1, 0]
                    ax.set_title('Thermal Annotated')
                    ax.imshow(ann_ir_img)
                    ax.axis('off')

            success, results = faces.detect_face(vi_img)

            if success:
                success, ann_vi_img = faces.draw_face(vi_img, results)

                if success and print_images:
                    ax = axs[1, 1]
                    ax.set_title('Regular Annotated')
                    ax.imshow(ann_vi_img)
                    ax.axis('off')

                fused_image = fuse_obj.fuseImages(ir_img, vi_img)

                if write_images:
                    img_name = os.path.join(output_path, root.split('\\')[-1], 'Fused-{}.jpg'.format(i))
                    cv2.imwrite(img_name, fused_image)

                if print_images:
                    ax = axs[1, 2]
                    ax.imshow(fused_image, cmap=plt.cm.gray)
                    ax.set_title('Fused Image (Thermal x Visual)')
                    ax.axis('off')

            if print_images:
                plt.show()


if __name__ == "__main__":
    print_images = True
    write_images = True
    main()
