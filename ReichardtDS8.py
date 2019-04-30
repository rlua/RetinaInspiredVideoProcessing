import numpy as np

#Returns two-components, each component can take positive or negative values
def Reichardt2(video):

    v1 = Reichardt_vertical_SingleChannel_Vectorized(video) #Directions 1, -1
    v3 = Reichardt_horizontal_SingleChannel_Vectorized(video) #Directions 3, -3

    return [v1, v3]

#Returns four components, each component can take positive or negative values
def Reichardt4(video):
    v1 = Reichardt_vertical_SingleChannel_Vectorized(video) #Directions 1, -1
    v3 = Reichardt_horizontal_SingleChannel_Vectorized(video) #Directions 3, -3
    v2 = Reichardt_diagonal1_SingleChannel_Vectorized(video) #Directions 2, -2
    v4 = Reichardt_diagonal2_SingleChannel_Vectorized(video) #Directions 4, -4

    return [v1, v2, v3, v4]

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

#Modified from Vertical System (VS) tangential cells network model (Trousdale et al. 2014)
#https://senselab.med.yale.edu/modeldb/showModel.cshtml?model=155727&file=/sim_vs_net/sim_vs_net.py#tabs-2
#Original description:
'''
Function:     reichardt_filter

Arguments:    movie - a movie (here, generated by rotation of an image about an axis) of dimension
                      (y_pixels,x_pixels,time_steps)
              reichard_locs - binary array of dimension (y_pixels,x_pixels), with 1 corresponding to a pixel containing
                              a Reichardt detector
              tau_hp - time constant of the high-pass filter in each half of the Reichardt detector
              tau_lp - time constant of the low-pass filter in each half of the Reichardt detector
              dt - time step of integration for the application of the Reichardt filtering
           
Output:       reich_up - Upward component of motion detected by the Reichardt filtering
              reich_down - Downward component of motion detected by the Reichardt filtering
           
Description: The reichardt_filter function filters the given movie through an array of Reichardt detectors, returning
             components corresponding to upward and downward motion. The type of detector implemented here makes use of
             a total of four filters --- two high pass and two low pass. For details, see Borst and Weber (2011) "Neural Action
             Fields for Optic Flow Based Navigation: A Simulation Study of the Fly Lobula Plate Network", 
             PLoS ONE 6(1): e16303. doi:10.1371/journal.pone.0016303.

Authors:     James Trousdale - jamest212@gmail.com
             * Adapted from code provided by Drs. Y. Elyada and A. Borst
'''

#def reichardt_filter(movie,reichardt_locs,tau_hp,tau_lp,dt):
def reichardt_filter_vertical(video,tau_hp,tau_lp,dt):
    
    dims = np.array(np.shape(video))
    
    high_pass = np.zeros(dims)
    low_pass = np.zeros(dims)
    reich = np.zeros(dims)
    
    high_pass_const = dt/(tau_hp + dt)
    low_pass_const = dt/(tau_lp + dt)
    
    # Perform high- and low-pass filtering in time of the movie
    for i in range(1,video.shape[0]):
        high_pass[i,:,:,:] = high_pass_const*video[i,:,:,:] + (1-high_pass_const)*high_pass[i-1,:,:,:]
        low_pass[i,:,:,:] = low_pass_const*video[i,:,:,:] + (1-low_pass_const)*low_pass[i-1,:,:,:]
    high_pass = video - high_pass
    
    
    # Each detector is composed of two detector subunits separated by a vertical distance of two degrees. Thus,
    # the output of a detector cross-multiplies the high-pass component of the corresponding pixel with the low pass 
    # component of the pixel two degrees above the pixel corresponding to the detectors location, and vice-versa, then
    # subtracts the two. Under this formalism, a negative response indicates generally upward motion across the pixel
    # corresponding to the detector, and a positive response indicates downward motion.
    #reich[int(2.0/y_res):,:,:] = low_pass[:-int(2.0/y_res),:,:]*high_pass[int(2.0/y_res):,:,:] \
    #                                - high_pass[:-int(2.0/y_res),:,:]*low_pass[int(2.0/y_res):,:,:]
    reich[:,1:,:,:] = low_pass[:,:-1,:,:]*high_pass[:,1:,:,:] \
                                    - high_pass[:,:-1,:,:]*low_pass[:,1:,:,:]
        
    
    # Separate the filtered movie into upward and downward components.  
    reich_up = np.copy(reich)
    reich_up[reich_up > 0] = 0
    reich_down = reich
    reich_down[reich_down < 0] = 0
    
    # Zero all pixels which do not correspond to the location of a Reichardt detector.
    #reich_up = -reich_up*repmat_3d(reichardt_locs,np.size(movie,2))
    #reich_down = reich_down*repmat_3d(reichardt_locs,np.size(movie,2))
    
    #return (reich_up,reich_down)
    return (reich_down,-reich_up)

def reichardt_filter_horizontal(video,tau_hp,tau_lp,dt):

    dims = np.array(np.shape(video))

    high_pass = np.zeros(dims)
    low_pass = np.zeros(dims)
    reich = np.zeros(dims)

    high_pass_const = dt/(tau_hp + dt)
    low_pass_const = dt/(tau_lp + dt)

    # Perform high- and low-pass filtering in time of the movie
    for i in range(1,video.shape[0]):
        high_pass[i,:,:,:] = high_pass_const*video[i,:,:,:] + (1-high_pass_const)*high_pass[i-1,:,:,:]
        low_pass[i,:,:,:] = low_pass_const*video[i,:,:,:] + (1-low_pass_const)*low_pass[i-1,:,:,:]
    high_pass = video - high_pass


    # Each detector is composed of two detector subunits separated by a vertical distance of two degrees. Thus,
    # the output of a detector cross-multiplies the high-pass component of the corresponding pixel with the low pass
    # component of the pixel two degrees above the pixel corresponding to the detectors location, and vice-versa, then
    # subtracts the two. Under this formalism, a negative response indicates generally upward motion across the pixel
    # corresponding to the detector, and a positive response indicates downward motion.
    #reich[int(2.0/y_res):,:,:] = low_pass[:-int(2.0/y_res),:,:]*high_pass[int(2.0/y_res):,:,:] \
    #                                - high_pass[:-int(2.0/y_res),:,:]*low_pass[int(2.0/y_res):,:,:]
    reich[:,:,1:,:] = low_pass[:,:,:-1,:]*high_pass[:,:,1:,:] \
                                    - high_pass[:,:,:-1,:]*low_pass[:,:,1:,:]


    # Separate the filtered movie into upward and downward components.
    reich_left = np.copy(reich)
    reich_left[reich_left > 0] = 0
    reich_right = reich
    reich_right[reich_right < 0] = 0

    # Zero all pixels which do not correspond to the location of a Reichardt detector.
    #reich_up = -reich_up*repmat_3d(reichardt_locs,np.size(movie,2))
    #reich_down = reich_down*repmat_3d(reichardt_locs,np.size(movie,2))

    #return (reich_up,reich_down)
    return (reich_right,-reich_left)


#Combine the two above, but make only one calculation for the hig_pass and low_pass exponential moving averages
def reichardt_filter_vertical_horizontal(video,tau_hp,tau_lp,dt):

    dims = np.array(np.shape(video))

    high_pass = np.zeros(dims)
    low_pass = np.zeros(dims)
    reich = np.zeros(dims)
    reich2 = np.zeros(dims)

    high_pass_const = dt/(tau_hp + dt)
    low_pass_const = dt/(tau_lp + dt)

    # Perform high- and low-pass filtering in time of the movie
    for i in range(1,video.shape[0]):
        high_pass[i,:,:,:] = high_pass_const*video[i,:,:,:] + (1-high_pass_const)*high_pass[i-1,:,:,:]
        low_pass[i,:,:,:] = low_pass_const*video[i,:,:,:] + (1-low_pass_const)*low_pass[i-1,:,:,:]
    high_pass = video - high_pass


    # Each detector is composed of two detector subunits separated by a vertical distance of two degrees. Thus,
    # the output of a detector cross-multiplies the high-pass component of the corresponding pixel with the low pass
    # component of the pixel two degrees above the pixel corresponding to the detectors location, and vice-versa, then
    # subtracts the two. Under this formalism, a negative response indicates generally upward motion across the pixel
    # corresponding to the detector, and a positive response indicates downward motion.
    #reich[int(2.0/y_res):,:,:] = low_pass[:-int(2.0/y_res),:,:]*high_pass[int(2.0/y_res):,:,:] \
    #                                - high_pass[:-int(2.0/y_res),:,:]*low_pass[int(2.0/y_res):,:,:]
    reich[:,1:,:,:] = low_pass[:,:-1,:,:]*high_pass[:,1:,:,:] \
                                    - high_pass[:,:-1,:,:]*low_pass[:,1:,:,:]

    #Left-right version
    reich2[:,:,1:,:] = low_pass[:,:,:-1,:]*high_pass[:,:,1:,:] \
                                    - high_pass[:,:,:-1,:]*low_pass[:,:,1:,:]



    # Separate the filtered movie into upward and downward components.
    reich_up = np.copy(reich)
    reich_up[reich_up > 0] = 0
    reich_down = reich
    reich_down[reich_down < 0] = 0

    # Separate the filtered movie into leftward and rightward components.
    reich_left = np.copy(reich2)
    reich_left[reich_left > 0] = 0
    reich_right = reich2
    reich_right[reich_right < 0] = 0

    return (reich_down,-reich_up,reich_right,-reich_left)

