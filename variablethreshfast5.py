
import scipy as sp
import time
import sys

import createbacteria  as cb
import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy.ndimage import label
from multiprocessing import Pool
import morphsnakes
from scipy.interpolate import InterpolatedUnivariateSpline

def gauss_kern(Img):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    h2,h1 = Img.shape    
    x, y = np.mgrid[0:h2, 0:h1]
    x = x-h2/2
    y = y-h1/2
#    sigma = 13.4
    sigma = 1.5
    g = np.exp( -( x**2 + y**2 ) / (2*sigma**2) );
    return g / g.sum()
    
def createSortindex(radius):
    circle = cb.createCircleEdge(radius)
    circindex = np.argwhere(circle)
    circindex[:, 0] = circindex[:, 0] - circindex[:, 0].mean()
    circindex[:, 1] = circindex[:, 1] - circindex[:, 1].mean()
    quadrindex = np.zeros((circindex.shape[0], 4))
    quadrindex[:, 0:2] = circindex
    quadrindex[:, 2] = np.angle(quadrindex[:,0]+1j*quadrindex[:,1],deg=True)
    sortindex =  quadrindex[np.argsort(quadrindex[:,2]), 0:2]
    return(sortindex)
#@jit('f8[:,:](f8[:,:])')
def weightFind(temparr):
#    temparr = weightarray[:,:,jj]
    uniq = np.unique(temparr)
    uniq = uniq[uniq>0]
    
    for ii in range(uniq.shape[0]):
        singlereg = temparr==uniq[ii]
        
        if( np.sum(singlereg)>5000 or np.sum(singlereg)<150  ):
            temparr[temparr==uniq[ii]] = 0
        else:
            
            params = cb.ellipseParam(np.int64(np.argwhere(singlereg==1)))  
            
# change the following parameters according to the data used              
#            if( params[1] >8 and params[1]<23 and params[0] <300 and params[0]>15): #phase data 1 
#            if( params[1] >8 and params[1]<23 and params[0] <300 and params[0]>20): #phase data 2
#            if( params[1] >12 and params[1]<30 and params[0] <300 and params[0]>35): #phase data 3
            if( params[1] >5 and params[1]<19 and params[0] <150 and params[0]>15): # fluorescent data

                indivarea = np.sum(np.float32(singlereg))
                elipsarea = params[2]                
                                                        
                indivperim = np.sum(np.float32(singlereg - cv2.erode(np.uint8(singlereg) ,np.ones((3,3))) ))                    
                elipsperim = params[3]
                                                                        
                residual = np.abs(indivarea - elipsarea ) 
                residuarearatio = np.min([indivarea,  elipsarea] )/np.max([indivarea,  elipsarea] )
                    
                residuperimratio = np.min([indivperim,  elipsperim] )/np.max([indivperim,  elipsperim] )              
                convexity = indivarea/cb.findConvexarea(singlereg)
#                weightval = residuarearatio #+ 0.5*convexity
                weightval = 0.5*residuarearatio + 0.5*convexity
                
                temparr[temparr==uniq[ii]] = weightval
    
            else:
               temparr[temparr==uniq[ii]] = 0

    return(temparr)
    
    
def renormalize(eigh1):
    et = np.copy(eigh1)
    rval = np.arange(eigh1.min(),eigh1.max(),10)
    arr = []
    for ii in np.arange(eigh1.min(),eigh1.max(),10):
        arr.append( (eigh1<ii).sum() )
    
    a = np.array(arr)
    b = np.float64(a)
    b = b/b.max()
    rthresh = rval[np.argwhere(b>0.01)[0]]
    et[et<rthresh] = rthresh
    temarr = et-et.min()
    temarr = temarr/temarr.max()
    temarr = np.uint8(temarr*255)
    return(temarr)
    

    
#%%###########################################################################
parallel = 1 # 0 for runnning single thread 1 for multithreading

THRESHWIN = 5  #select threshold window around the mean value of etemp

#orimg = cv2.imread('phasedata1.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
#orimg = cv2.imread('phasedata2.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
#orimg = cv2.imread('phasedata3.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
orimg = cv2.imread('fluodata.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)





orimg = np.float32(orimg)
orimg = orimg.max()-orimg # uncomment for fluorescent data
inputimage = np.copy(orimg)
#orimg = orimg[250:400,300:400]

startime = time.time()


#% variable thresholding 
# find the eigen value of gaussian smoothed hessian image
Img = np.copy(orimg)
gau = gauss_kern(Img)
#Img_smooth = signal.convolve(Img,g,mode='same')
Imgfft = np.fft.rfft2(Img)
gfft = np.fft.rfft2(gau)
fftimage = np.multiply(Imgfft, gfft)
Img_smooth =np.real(np.fft.ifftshift( np.fft.irfft2(fftimage)))
#myshow2(Img_smooth)
Iy, Ix = np.gradient(Img_smooth)
Ixy, Ixx = np.gradient(Ix)
Iyy, Iyx = np.gradient(Iy)

eigvallam = np.zeros((2,Ixx.shape[0],Ixx.shape[1] ))
trhessian = Ixx+Iyy
dethessian = Ixx*Iyy-Ixy*Ixy

eigvallam[0,:,:] = 0.5*(trhessian + np.sqrt(trhessian*trhessian - (4*dethessian) ))
eigvallam[1,:,:] = 0.5*(trhessian - np.sqrt(trhessian*trhessian - (4*dethessian) ))
eigh1 = eigvallam.min(0)
eigh2 = eigvallam.max(0)



#cb.myshow2(eigh1)
etemp = np.copy(eigh1)
etemp = etemp-etemp.min()
etemp = etemp/etemp.max()
etemp = np.uint8(etemp*255)
#%%




#find histogram peak to detect start and end intensities
histo, inten = np.histogram(etemp,range(256))

histo = np.float32(histo)
inten = np.float32(inten)

meanval = np.int(np.sum((histo/histo.sum())*range(255)))

peakval = inten[np.argwhere(histo == histo.max())[0][0]]

startinten = meanval - THRESHWIN
stopinten = meanval + THRESHWIN



LEVELS = np.int(stopinten-startinten)
labelimgarray = np.zeros((etemp.shape[0], etemp.shape[1], LEVELS), dtype=np.float)

for ii in range(LEVELS):
    labelimgarray[:,:,ii], ncc = label((etemp>=(ii+startinten)),np.ones((3,3))) 
#    cc[ii]=ncc
    
weightarray = []
for ii in range(LEVELS):
    weightarray.append(np.float64(labelimgarray[:,:,ii]))
    

#resultarray = []

#for ii in range(LEVELS):
#    resultarray.append(weightFind(np.float64(labelimgarray[:,:,ii])))


#for ii in range(LEVELS):
#    labelimgarray[:,:,ii], ncc = label((etemp>=(ii)),np.ones((3,3))) 
#    cc[ii]=ncc
 

#
##weightFind(temparr)
#l
resultarray = []

if (parallel ==1):
    pool = Pool(4)
    resultarray = pool.map(weightFind,weightarray)
    pool.close()

#fastweightFind = jit(f8[:,:](f8[:,:]))(weightFind)
else:
    for ii in range(LEVELS):
        resultarray.append(weightFind(weightarray[ii]))
        print ii

maximg = np.max(resultarray,0)
maximg = sp.ndimage.binary_fill_holes(maximg>0.75)
outimg , count = label(maximg,np.ones((3,3)))

#cb.myshow2(outimg)
#cb.myshow2(cb.overlayImage(orimg,outimg))

cv2.imwrite('CBAout.tif',cb.overlayImage(orimg,outimg))

print 'CBA segmentation time in seconds :', time.time()-startime




#%%###########################################################################
# optional filtering and smoothing part

uniq = np.unique(outimg)[np.unique(outimg)>0]
feat = np.zeros((uniq.shape[0],5))

jj = 0
for ii in uniq:
    
    params = cb.ellipseParam(np.int64(np.argwhere(outimg == ii)))  
    feat[jj, 0] = params[0] 
    feat[jj, 1] = params[1]
    feat[jj, 2] = params[2]
    feat[jj, 3] = params[3]
    feat[jj, 4] = np.median(orimg[outimg == ii])
#    feat[jj,5] = np.argwhere(outimg == ii).sum()
    jj += 1

#    r = feat[:,0]/feat[:,1]

normfeat= feat/feat.max(0)
#%

#% gaussian fit to find upper cut of intensity
featval = normfeat[:, 4]    
h, c = np.histogram(featval,20)
nh = h*1.0/h.max()

ius = InterpolatedUnivariateSpline(c[:-1], nh)

maxval = ius(c).max()
maxvalloc =  np.argwhere(ius(c)==ius(c).max())[0][0]

startval = c[maxvalloc]
endval = c.max()
for jj in range(3):
    forwardint = np.linspace(startval, endval,1000)
    for ii in forwardint:
        if(ius(ii)<= maxval/2 ):
#                print ii
            endval = ii
            break

forlim = ii    
startval = c[maxvalloc]
endval = c.min()
for jj in range(3):
    reverseint = np.linspace(c[maxvalloc],c.min(),1000)
    for ii in reverseint:
        if(ius(ii)<= maxval/2 ):
            endval = ii
#                print ii
            break
    
revlim = ii    

meanlim = ((c[maxvalloc]-revlim) + (forlim-c[maxvalloc]))/2

fulwidhalfmax = forlim-revlim
estimsigma = fulwidhalfmax/(2*np.sqrt(2*np.log(2)) )

uppercut = c[maxvalloc] +4*estimsigma    #for phase data3

#uppercut = c[maxvalloc] + 5*meanlim    # for fluo data

tempout = np.zeros((outimg.shape[0],outimg.shape[1] ))    
jj = 0
for ii in uniq:
    tempout[outimg==ii] = featval[jj]
    jj +=1


filterout = outimg*( tempout<uppercut)    
cv2.imwrite('CBAfilter.tif',cb.overlayImage(orimg,filterout))
print 'CBA segmentation + filtering time in seconds :', time.time()-startime

#print time.time()-startime


#%% levelset smoothing
#https://github.com/pmneila/morphsnakes
#smoothing boundary  
uniq = np.unique(filterout)
uniq = uniq[uniq>0]

img = etemp/etemp.max()
smoothout = np.zeros_like(filterout)    

for cc in uniq:
# g(I)
    gI = morphsnakes.gborders(img, alpha=1000, sigma=4.248)
    
    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=25.01, balloon=-1)
    mgac.levelset = (filterout==cc)*1 
#    mgac.levelset = (filterout>0)*1 
    
    # Visual evolution.
    #ppl.figure()
    out = morphsnakes.evolve_visual(mgac, num_iters=10, background=img)
    smoothout = smoothout + out*cc

#smoothout,ncc = label(out,np.ones((3,3)))
#cb.myshow2(cb.overlayImage(orimg,smoothout))
cv2.imwrite('CBAfiltersmooth.tif',cb.overlayImage(orimg,smoothout))


print 'CBA segmentation + filtering + smoothing time in seconds :', time.time()-startime



