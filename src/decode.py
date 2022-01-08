import cv2
import numpy as np
import matplotlib.pyplot as plt

vidcap = cv2.VideoCapture('cat_without_light.mp4')

length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

totalNumberOfFrame = length
centerHeight = int(height/2)
centerWidth = int(width/2)

correct_output = np.tile(np.array([1, 0]), 70)
output_value = []

d_pixel_cut = 100
d_pixel_static = 10
d_histo_cut = 1
d_histo_static = 0.1

cut = 2
static = 0
gradual = 1

ROW_NUM = 1
COL_NUM = 1

correct_bit = []
current_bit = 1

def read_window(vidcap, window_length=6):
    frame_window_list = []
    for _ in range(window_length):
        _, image = vidcap.read()
        frame_window_list.append(image)

    return np.array(frame_window_list)

def get_difference(previous_frame, current_frame):
    previous_frame = previous_frame.astype(np.int32)
    current_frame = current_frame.astype(np.int32)

    d_pixel = np.sum(np.abs(previous_frame - current_frame))/(previous_frame.shape[0]*previous_frame.shape[1])
    old_bins = np.zeros(766)
    new_bins = np.zeros(766)
    rgb_sum_previous_frame = np.sum(previous_frame, axis = -1)
    rgb_sum_current_frame = np.sum(previous_frame, axis = -1)
    
    for i in range(previous_frame.shape[0]): # height
        for j in range(previous_frame.shape[1]): # width
            old_bins[rgb_sum_previous_frame[i][j]] += 1 
            new_bins[rgb_sum_current_frame[i][j]] += 1

    d_histo = np.nansum(np.power((old_bins - new_bins), 2)/np.maximum(old_bins, new_bins))

    return d_pixel, d_histo

currentFrameWindow = read_window(vidcap) # [6, 2160, 4096, 3]
currentSampleFrame = currentFrameWindow[0] # [2160, 4096, 3]

print('decode value : ')

for i in range(6, totalNumberOfFrame-5, 6):
    nextFrameWindow = read_window(vidcap)
    nextSampleFrame = nextFrameWindow[0]

    # resized_current_image = resize_image(currentSampleFrame)
    # resized_next_image = resize_image(nextSampleFrame)

    # get difference
    resized_current_frame = cv2.resize(currentSampleFrame, (int(width/4), int(height/4)))
    resized_next_frame = cv2.resize(nextSampleFrame, (int(width/4), int(height/4)))
    d_pixel, d_histo = get_difference(resized_current_frame, resized_next_frame)

    # determine frame scene
    if d_pixel > d_pixel_cut and d_histo > d_histo_cut: # cut scene
        frameScene = cut
        currentSampleFrame = nextSampleFrame # assign the new frame to previous frame for next window
        currentFrameWindow = nextFrameWindow # assign the next window to current window
        continue
    elif d_pixel < d_pixel_static and d_histo < d_histo_static: # static scene
        frameScene = static
    else:
        frameScene = gradual

    #  sum RGB
    currentFrameWindow = np.sum(currentFrameWindow, axis=3)

    # display pixel value
    # fprintf("--------------------\npixel value on %d th frame\n", i-6)
    # fprintf("%d %d %d %d %d %d\n", currentFrameWindow(360, 640, 1), currentFrameWindow(360, 640, 2), currentFrameWindow(360, 640, 3), currentFrameWindow(360, 640, 4), currentFrameWindow(360, 640, 5), currentFrameWindow(360, 640, 6))

    # cut into grids
    row_num = ROW_NUM
    col_num = COL_NUM
    for j in range(row_num):
        for k in range(col_num):
            centerHeight = int((j+0.5)*height/row_num)
            centerWidth = int((k+0.5)*width/col_num)

            # pixelIntensity = (squeeze(currentFrameWindow(centerHeight, centerWidth, : ))).'
            pixel_intensity = currentFrameWindow[:, centerHeight, centerWidth]
            alphaValue = 1 - pixel_intensity/pixel_intensity[1]
            
            # extended_alpha = np.tile(alphaValue, 5)
            demodulation = np.fft.fft(alphaValue, 30)
            # Pyy = demodulation * np.conjugate(demodulation) / 30 # normalization
            Pyy = abs(demodulation) / 30

            # freq = np.arange(0, 15)*60/30
            # plt.plot(freq, Pyy[:15])
            # plt.show()

            if Pyy[11] > Pyy[16]:
                print("0")
                output_value.append(0)
            else:
                print("1")
                output_value.append(1)
    
    currentSampleFrame = nextSampleFrame # assign the new frame to previous frame for next window
    currentFrameWindow = nextFrameWindow # assign the next window to current window

    correct_bit.append(current_bit)
    current_bit = 1 - current_bit

print("error rate = ", np.sum(np.abs(np.array(output_value) - np.array(correct_bit)))/len(correct_bit))
    


