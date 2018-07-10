import numpy as np

def Reichardt8(video):
    '''
       Returns a tuple of Reichardt-Hassenstein correlators in 8 directions
    '''
    vp1, vm1 = Reichardt_vertical_2channels_Vectorized(video) #Directions 1, -1
    vp3, vm3 = Reichardt_horizontal_2channels_Vectorized(video) #Directions 3, -3
    vp2, vm2 = Reichardt_diagonal1_2channels_Vectorized(video) #Directions 2, -2
    vp4, vm4 = Reichardt_diagonal2_2channels_Vectorized(video) #Directions 4, -4

    return vp1, vm1, vp2, vm2, vp3, vm3, vp4, vm4

#timeDelay is unused in the Vectorized method, but may be useful later
def Reichardt_vertical_2channels_Vectorized(video,timeDelay=1):
    '''
       Reichardt-Hassenstein inspired video processing
       Put negative values into another tensor and then concat for the two (e.g. red and green) channels
    '''

    # normalize video pixel value to 0 ~ 1 (I don't think this normalizes to that range...)
    #video = (video.astype(np.float32)) + 1e-3

    vc_shift_vert_by1back=np.roll(video,1,axis=1)
    vc_shift_time_by1forw=np.roll(video,-1,axis=0)
    vc_shift_vert_by1back_time_by1forw=np.roll(vc_shift_vert_by1back,-1,axis=0)
    vc= vc_shift_vert_by1back*vc_shift_time_by1forw - vc_shift_vert_by1back_time_by1forw*video 

    vc_neg=vc.clip(max=0)
    vc_neg=-1*vc_neg
    vc=vc.clip(0)

    #Expand dims may not be necessary
    #vc=np.expand_dims(vc,len(video.shape))
    #vc_neg=np.expand_dims(vc_neg,len(video.shape))

    #Merge on the last dimension (but still separate indices)
    #return np.concatenate((vc_neg, vc), axis=len(video.shape))
    return vc, vc_neg


def Reichardt_diagonal1_2channels_Vectorized(video,timeDelay=1):
    '''
       Reichardt-Hassenstein inspired video processing
       Put negative values into another tensor and then concat for the two (e.g. red and green) channels
    '''

    # normalize video pixel value to 0 ~ 1 (I don't think this normalizes to that range...)
    #video = (video.astype(np.float32)) + 1e-3
    
    vc_shift_diag_by1back=np.roll(video,(1,1),axis=(1,2))
    vc_shift_time_by1forw=np.roll(video,-1,axis=0)
    vc_shift_diag_by1back_time_by1forw=np.roll(vc_shift_diag_by1back,-1,axis=0)
    vc= vc_shift_diag_by1back*vc_shift_time_by1forw - vc_shift_diag_by1back_time_by1forw*video

    vc_neg=vc.clip(max=0)
    vc_neg=-1*vc_neg
    vc=vc.clip(0)

    #vc=np.expand_dims(vc,len(video.shape))
    #vc_neg=np.expand_dims(vc_neg,len(video.shape))

    #Merge on the last dimension (but still separate indices)
    #return np.concatenate((vc_neg, vc), axis=len(video.shape))
    return vc, vc_neg


def Reichardt_horizontal_2channels_Vectorized(video,timeDelay=1):
    '''
       Reichardt-Hassenstein inspired video processing
       Put negative values into another tensor and then concat for the two (e.g. red and green) channels
    '''

    # normalize video pixel value to 0 ~ 1 (I don't think this normalizes to that range...)
    #video = (video.astype(np.float32)) + 1e-3

    vc_shift_horz_by1back=np.roll(video,1,axis=2)
    vc_shift_time_by1forw=np.roll(video,-1,axis=0)
    vc_shift_horz_by1back_time_by1forw=np.roll(vc_shift_horz_by1back,-1,axis=0)
    vc= vc_shift_horz_by1back*vc_shift_time_by1forw - vc_shift_horz_by1back_time_by1forw*video

    vc_neg=vc.clip(max=0)
    vc_neg=-1*vc_neg
    vc=vc.clip(0)

    #vc=np.expand_dims(vc,len(video.shape))
    #vc_neg=np.expand_dims(vc_neg,len(video.shape))

    #Merge on the last dimension (but still separate indices)
    #return np.concatenate((vc_neg, vc), axis=len(video.shape))
    return vc, vc_neg


def Reichardt_diagonal2_2channels_Vectorized(video,timeDelay=1):
    '''
       Reichardt-Hassenstein inspired video processing
       Put negative values into another tensor and then concat for the two (e.g. red and green) channels
    '''

    # normalize video pixel value to 0 ~ 1 (I don't think this normalizes to that range...)
    #video = (video.astype(np.float32)) + 1e-3

    vc_shift_diag_by1back=np.roll(video,(-1,1),axis=(1,2))
    vc_shift_time_by1forw=np.roll(video,-1,axis=0)
    vc_shift_diag_by1back_time_by1forw=np.roll(vc_shift_diag_by1back,-1,axis=0)
    vc= vc_shift_diag_by1back*vc_shift_time_by1forw - vc_shift_diag_by1back_time_by1forw*video

    vc_neg=vc.clip(max=0)
    vc_neg=-1*vc_neg
    vc=vc.clip(0)

    #vc=np.expand_dims(vc,len(video.shape))
    #vc_neg=np.expand_dims(vc_neg,len(video.shape))

    #Merge on the last dimension (but still separate indices)
    #return np.concatenate((vc_neg, vc), axis=len(video.shape))
    return vc, vc_neg

