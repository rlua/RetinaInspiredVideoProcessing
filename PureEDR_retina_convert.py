import os
import argparse

import scipy.misc
import numpy as np
import skvideo.io
import scipy
import imageio

from timeit import default_timer as timer

#RCL
#from ReichardtDS8 import *

def rescale(matrix, scale_min=0, scale_max=255):
    """
    Rescale matrix in a given range, element wise
    """
    matrix = matrix.astype(np.float32)
    matrix = (matrix - matrix.min()) * ((scale_max - scale_min) / 
        np.ptp(matrix)) + scale_min
    return matrix

def retina(video, alpha, mu_on, mu_off):
    """
    video is a numpy array, where the first dimention is time T 
    """
    # decay constant must be smaller than 1
    assert alpha <= 1 and alpha >= 0

    ema = np.zeros(video.shape, dtype=float)

    # normalize video pixel value to 0 ~ 1
    video = (video.astype(np.float32)) + 1e-3

    # EMA filtering
    ema[0, :, :, :] = video[0, :, :, :]

    for t in range(1, video.shape[0]):
        ema[t, :, :, :] = (1 - alpha) * ema[t - 1, :, :, :] + \
            alpha * video[t, :, :, :]
    
    # compute relative change 
    change = np.tanh(np.log(np.divide(video, ema + 1e-5)))

    print('change shape',change.shape)
    # thresholding
    on = np.expand_dims(np.maximum(0, change - mu_on), 
        len(video.shape))
    off = np.expand_dims(np.maximum(0, - (change - mu_off)),
        len(video.shape))

    print('on shape',on.shape)
    print('off shape',off.shape)
    return np.concatenate((off, on), axis=len(video.shape))

#Don't call expand_dims
#Don't convert to float and add 1e-3
#Remove print statements
def retina2(video, alpha, mu_on, mu_off):
    """
    video is a numpy array, where the first dimention is time T
    """
    # decay constant must be smaller than 1
    assert alpha <= 1 and alpha >= 0

    ema = np.zeros(video.shape, dtype=float)

    # normalize video pixel value to 0 ~ 1
    #video = (video.astype(np.float32)) + 1e-3

    # EMA filtering
    ema[0, :, :, :] = video[0, :, :, :]

    for t in range(1, video.shape[0]):
        ema[t, :, :, :] = (1 - alpha) * ema[t - 1, :, :, :] + \
            alpha * video[t, :, :, :]

    # compute relative change
    #change = np.tanh(np.log(np.divide(video, ema + 1e-5)))
    change = np.tanh(np.log(np.divide(video, ema + 1e-5)))

    #print('change shape',change.shape)
    # thresholding
    on = np.maximum(0, change - mu_on)
    off = np.maximum(0, - (change - mu_off))

    #print('on shape',on.shape)
    #print('off shape',off.shape)
    #return np.concatenate((off, on), axis=len(video.shape)-1)
    return off, on


def compute_local_motion_feature(video):
    """
    Compute local motion gradient, ignore this
    """

    # pad t, h, w with same value
    video = np.pad(video, [[1,1] for _ in range(3)] + [[0, 0]], mode='wrap')
    

    vertical_filter = np.expand_dims(np.array(
        [[[-1,-1,-1],[0,0,0],[1,1,1]],[[1,1,1],[0,0,0],[-1,-1,-1]]]), len(video.shape))
    horizon_filter = np.expand_dims(np.array(
        [[[1,0,-1],[1,0,-1],[1,0,-1]],[[-1,0,1],[-1,0,1,],[-1,0,1]]]), len(video.shape))

    # filtering using v_filter and h_filter
    v_map = np.sign(scipy.signal.convolve(video, vertical_filter,
        mode="valid"))
    h_map = np.sign(scipy.signal.convolve(video, horizon_filter,
        mode="valid"))

    return (v_map>0).astype(np.uint8), (v_map<0).astype(np.uint8),\
        (h_map>0).astype(np.uint8), (h_map<0).astype(np.uint8),

def write_video(path, video, frame_rate=5):
    writer = imageio.get_writer(path, fps=frame_rate)
    for frame in video:
        writer.append_data(frame)

def write_frame(path, video, fmt="jpg"):
    for i,frame in enumerate(video):
        off = frame[:,:,0]  
        on = frame[:,:,1]  
        scipy.misc.imsave(os.path.join(path,str(i).zfill(6)+"_off."+fmt), \
            off)
        scipy.misc.imsave(os.path.join(path,str(i).zfill(6)+"_on."+fmt), \
            on)

def make_retina_output(args):
    """
    Produce pre-normalization retina output
    """
    video = skvideo.io.vread(args.input_path, as_grey=True)
    video_rgb = skvideo.io.vread(args.input_path, as_grey=False)

    #For long videos from Caleb Kemere
    video = video[:100]
    video_rgb = video_rgb[:100]

    print('skvideo.io.vread returned video with shape',video.shape)
    #print('skvideo.io.vread video type: ',type(video[0,0,0,0]),video[0,0,0,0])
    #print('skvideo.io.vread returned video_rgb with shape',video_rgb.shape)

    # 
    #video = (video.astype(np.float32))/255.0
    video = (video.astype(np.float32)) + 1e-3
    #video = 2*video-1
    #print(video)

    ##RCL
    #start = timer()
    #vp1, vm1 = Reichardt_vertical_2channels_Vectorized(video,timeDelay=1)
    #vp3, vm3 = Reichardt_horizontal_2channels_Vectorized(video,timeDelay=1)
    #vp2, vm2 = Reichardt_diagonal1_2channels_Vectorized(video,timeDelay=1)
    #vp4, vm4 = Reichardt_diagonal2_2channels_Vectorized(video,timeDelay=1)
    #
    #vp1, vm1, vp2, vm2, vp3, vm3, vp4, vm4 = Reichardt8(video)
    #end = timer()
    #print('Reichardt elapsed time: ', end - start)

    #Build what directions to show here
    #retina_output = np.concatenate((vp1, vp2, vp3), axis=len(video.shape)) 
    #retina_output = np.concatenate((vp3, vp3, vm3), axis=(len(video.shape)-1)) 

    # save retina output of video
    #start = timer()
    retina_output = retina(video, args.alpha, args.mu_on, args.mu_off)
    #off, on = retina2(video, args.alpha, args.mu_on, args.mu_off)
    #end = timer()
    #print('EDR elapsed time: ', end - start)
    #retina_output = np.concatenate((off, on), axis=len(video.shape)-1)

    #print retina_output.shape
    print('retina_output shape',retina_output.shape)

    # rescale video for better contrast in visualization
    retina_output = np.clip(rescale(retina_output), 0, 255).\
        astype(np.uint8)
 
    # put on channel on red, off channel on green
    #RCL color assignments were flipped, on is green, off is red.
    #Below, B color channel is added (with default value 0) by padding
    retina_output = np.squeeze(np.pad(retina_output,
        [(0, 0) for i in range(len(retina_output.shape) - 1)] + [(0, 1)],
         mode='constant'))
    print('retina_output shape after adding third (B) color channel',retina_output.shape)
    #retina_output = np.squeeze(retina_output)

    # write retina output to file
    if args.out_path:
        #Default
        #write_video(args.out_path, retina_output, frame_rate=args.frame_rate)
        #Side-by-side of Reichardt-treated video and original
        #write_video(args.out_path, np.concatenate((retina_output[:,:,:int(video_rgb.shape[2]/2)],video_rgb[:,:,:int(video_rgb.shape[2]/2)]),axis=2), frame_rate=args.frame_rate)
        write_video(args.out_path, np.concatenate((retina_output[:,:,:],video_rgb[:,:,:]),axis=2), frame_rate=args.frame_rate)

    # compute local motion feature from EDR
    #v_map_p, v_map_n, h_map_p, h_map_n =\
    #    compute_local_motion_feature(retina_output_rgb)
    
    # write lmf to file
    #skvideo.io.vwrite(args.local_motion_feature_path, 
    #        255*np.concatenate((v_map_p,v_map_n, h_map_p,h_map_n), axis=2))

    # write retina frame to file
    if args.frame_folder:
        if not os.exists(args.frame_folder):
            os.mkdir(args.frame_folder)
        write_frame(args.frame_folder, retina_output_rgb)
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('-in', "--input_path", type=str, 
        help="Input path of video.")   
    parser.add_argument('-out', "--out_path", type=str, 
        help="Out path of video.", default=None)   
    parser.add_argument('-lmf_out', "--local_motion_feature_path", 
        type=str, help="Out path of local motion feature.")   
 
    parser.add_argument('-mon', "--mu_on", type=float, 
        help="EDR on channel threshold.", default=0.05)
    parser.add_argument('-moff', "--mu_off", type=float, 
        help="EDR off channel threshold.", default=-0.1)
    parser.add_argument('-al', "--alpha", type=float, help="decay constant",
        default=0.5)
    parser.add_argument('-fps',"--frame_rate", type=int, help="output frame rate", default=12)
    parser.add_argument('-fr', "--frame_folder", type=str, help="edr as frame_folder", default=None)


    #from timeit import default_timer as timer

    start = timer()
    make_retina_output(parser.parse_args())
    end = timer()
    print('make_retina_output elapsed time: ', end - start)

