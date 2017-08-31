#import tensorflow as tf
import numpy as np
import cv2
#import cv2.cv as cv
import matplotlib.pyplot as plt
import random




#change the root directory
root = '../tiny-imagenet-200/train/'

def resize_frame(frame, size=(128,171), verbose = False):


    if frame.shape != (240,320,3):
        print frame.shape


    height, width = size[0], size[1]
    new_img = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)

    if verbose:
        print frame.shape
        cv2.imwrite('/s/chopin/k/grad/dkpatil/PycharmProjects/Projects/intermediate-studies/images/train/resized image.png', new_img)

    return new_img


def crop(img, (topleft_h, topleft_w), size = (112,112), verbose = False):
    #crop 112x112 from the reshaped frame
    h, w, c = img.shape

    cropped = img[topleft_h:topleft_h + size[0], topleft_w:topleft_w + size[1], :]  # tf.image.crop_to_bounding_box(img, topleft_h, topleft_w, 224, 224)

    if verbose:
        print 'topleft_h, topleft_w: ', topleft_h, topleft_w
        print cropped.shape
        cv2.imwrite('/s/chopin/k/grad/dkpatil/PycharmProjects/Projects/intermediate-studies/images/train/croped image.png' , cropped)
    return cropped


def flip_image(img, flag, verbose=False):
    #flip the 112x112 frame from crop function

    if flag:
        img_flipped = cv2.flip(img, 1)
        if verbose:
            cv2.imwrite('/s/chopin/k/grad/dkpatil/PycharmProjects/Projects/intermediate-studies/images/train/flipped image.png' , img_flipped)
            print img_flipped.shape
        return img_flipped
    else:
        if verbose:
            cv2.imwrite('/s/chopin/k/grad/dkpatil/PycharmProjects/Projects/intermediate-studies/images/train/unflipped image.png' , img)
            print img.shape
        return img


def proc_video(path, return_all = False, clip_limit = 16, size4crop = (112, 112), test_flag = False, verbose = False):

    def process_clip():
        #if verbose:
            #fourcc = cv2.VideoWriter_fourcc(*'XVID')
            #out = cv2.VideoWriter(('/s/chopin/k/grad/dkpatil/PycharmProjects/Projects/intermediate-studies/images/train/' + str(clip_no) + '.avi'), fourcc, 30, frameSize=size4crop)

        clip = []
        flip_flag = random.randint(0, 1)

        resize_height = random.randint(114, 144)
        resize_width = int(resize_height * 1.333)
        h, w = resize_height, resize_width

        topleft_h = random.randint(0, h - size4crop[0] - 1)
        topleft_w = random.randint(0, w - size4crop[1] - 1)


        for frm_no in range(clip_limit):
            ret, frame = vid.read()
            if verbose:
				print(ret)
            if ret:
                if test_flag:
                    resized_test = resize_frame(frame, size=(112, 149))
                    cropped_test = crop(resized_test, (0, 18), size=size4crop)
                    clip.append(cropped_test)
                else:
                    resized = resize_frame(frame, size=(resize_height, resize_width), verbose=verbose)
                    cropped = crop(resized, (topleft_h, topleft_w), size=size4crop, verbose=verbose)
                    flipped = flip_image(cropped, flip_flag, verbose)
                    clip.append(flipped)
            #if verbose:
                #out.write(flipped)
        #if verbose:
            #out.release()
        if len(clip) == clip_limit:
            return clip
        else:
            return None

    vid = cv2.VideoCapture(path)
    processed = []
    if vid.isOpened:
        if verbose:
			ret, frame = vid.read()
			print(ret)
			print(frame)
        total_frame = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        if verbose:
			print('total frame number %d' %total_frame)
        #h, w = size4resize
        clip_nums = range(int(total_frame / clip_limit))

        if return_all == False:
            target_clip = random.choice(clip_nums)
            for clip_no in range(int(total_frame / clip_limit)):
                if clip_no == target_clip:
                    #print 'clip in consideration is: ', target_clip
                    return process_clip()
                else:
                    for _ in range(clip_limit):
                        vid.read()
        else:
            for clip_no in range(int(total_frame / clip_limit)):
                clip = process_clip()
                if clip != None:
                    processed.append(clip)
            return processed
    else:
		print('video not opened')
    return None
'''
if __name__ == '__main__':
    path = 'v_ApplyEyeMakeup_g01_c01.avi'

    proc_video(path, verbose=True)
    # vid = cv2.VideoCapture(path)
    # if vid.isOpened:
    #     print vid.get(cv2.CAP_PROP_FRAME_COUNT)
        # frame = vid.read()[1]
        # new_img = resize_frame(frame, verbose=True)
        #
        # topleft_h = random.randint(0, h - size[0] - 1)
        # topleft_w = random.randint(0, w - size[1] - 1)
        # cropped_img = crop(new_img, verbose=True)
        # print cropped_img.shape


'''
