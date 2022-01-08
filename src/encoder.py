import cv2
import numpy as np
import math

def encode_frame(frame, alpha, offset_h, offset_w, height, width):
    hc, wc = int(height/2), int(width/2)
    for h in range(offset_h, offset_h+height):
        for w in range(offset_w, offset_w+width):
            frame[h,w,:] = np.dot(frame[h,w,:],(1-alpha/math.sqrt(2*math.pi)*math.exp(-((wc-w)**2+(hc-h)**2)/2)))
    return frame

def encode(window, bit, alpha, offset_h, offset_w, height, width):
    cnt = [1, 3, 5] if bit == 1 else [2, 5]
    for i in cnt:
        window[i] = encode_frame(window[i], alpha, offset_h, offset_w, height, width)
    return window

if __name__ == '__main__':
    path = './resized.mp4'
    video = cv2.VideoCapture(path)

    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    numFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    encoded = cv2.VideoWriter('output.mp4', fourcc, 30.0, (width,  height))

    ### Parameter ###
    grid_height_cnt = 2     # count of grids vertically
    grid_width_cnt = 2      # count of grids horizontally
    block_size = 4          # evaluate color intensity every ? pixels
    #################

    bit = 1

    grid_height = int(height/grid_height_cnt)
    grid_width = int(width/grid_width_cnt)

    frame_window = []
    _, frame = video.read()
    frame_window.append(frame)
    for f in range(6, numFrames, 6):
        # log
        print('[{}/{}]'.format(int(f/6), int(numFrames/6)))

        for i in range(6):
            _, frame = video.read()
            frame_window.append(frame)

        for g in range(0, grid_height_cnt*grid_width_cnt):
            tmp, hist1, hist2 = np.int32(0), np.zeros(1+255*3, dtype=np.int32), np.zeros(1+255*3, dtype=np.int32)
            mean = 0
            height_sample, width_sample = int(grid_height/block_size), int(grid_width/block_size)
            gh, gw = int(g/grid_height_cnt), g%grid_height_cnt
            for h in range(height_sample):
                for w in range(width_sample):
                    intensity1 = np.sum(frame_window[0][gh*grid_height+h*block_size, gw*grid_width+w*block_size, :], dtype=np.int32)
                    intensity2 = np.sum(frame_window[-1][gh*grid_height+h*block_size, gw*grid_width+w*block_size, :], dtype=np.int32)
                    tmp += abs(intensity1 - intensity2)
                    hist1[intensity1] += 1
                    hist2[intensity2] += 1
                    mean = mean + intensity2
            tmp /= height_sample * width_sample
            mean /= height_sample * width_sample

            tmp2 = 0
            for i in range(1+255*3):
                if hist1[i] + hist2[i] > 0:
                    tmp2 += ((hist1[i] - hist2[i])**2 / max(hist1[i], hist2[i]))

            delta_alpha = 0.35 if mean <= 50 else 0.3 if mean <= 100 else 0.25 if mean <= 150 else 0.2

            # static scene
            if tmp < 10 and tmp2 < 0.1:
                frame_window = encode(window=frame_window, bit=bit, alpha=delta_alpha, offset_h=gh*grid_height, offset_w=gw*grid_width, height=grid_height, width=grid_width)
            # gradual scene
            elif tmp < 100 or tmp2 < 1:
                frame_window = encode(window=frame_window, bit=bit, alpha=delta_alpha*2, offset_h=gh*grid_height, offset_w=gw*grid_width, height=grid_height, width=grid_width)
            # cut scene
            else:
                pass

        for i in range(6):
            encoded.write(frame_window[i])

        frame_window = [frame_window[-1]]
        bit = 1 - bit
    
    video.release()
    encoded.release()
        