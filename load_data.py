import cv2
import numpy as np
import os

def load_data(save = False, path = '../Data/research_data'):
    notation_path = os.path.join(path, 'labels')
    image_path = os.path.join(path, 'image')
    actual_notation = os.path.join(path, 'notation')
    notations = []
    images = []
    act_not = []
    done = False

    dot_list = os.listdir(notation_path)
    image_list = os.listdir(image_path)
    not_list = os.listdir(actual_notation)

    sumt = 0
    img = cv2.imread(os.path.join(actual_notation, not_list[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for row in img:
        for val in row:
            if val != 0:
                sumt += 1

    print(sumt)

    for entry in dot_list:
        img = cv2.imread(os.path.join(notation_path, entry))
        # Insert any image preprocessing here!
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not done:
            pix = {}
            for row in img:
                for val in row:
                    if val not in pix:
                        pix[val] = 1
                    else:
                        pix[val] += 1

            print(pix)
            done = True

        img = img[..., np.newaxis]       
        notations.append(img)

    for entry in image_list:
        img = cv2.imread(os.path.join(image_path, entry))
        images.append(img)

    for entry in not_list:
        img = cv2.imread(os.path.join(actual_notation, entry))
        act_not.append(img)

    if save:
        images = np.array(images)
        notations = np.array(notations)

        np.save('./saves/x_raw.npy', images)
        np.save('./saves/y_raw.npy', notations)
        np.save('./saves/dots.npy', act_not)

    else:
        return np.array(images), np.array(notations)

if __name__ == '__main__':
    load_data(True)

