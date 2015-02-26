''''
img is the classified image
'''


import numpy as np
import cv2
from matplotlib import pyplot as plt
import skimage.morphology as skmorph
from scipy.ndimage import label
import scipy as sp
import gaussianRectangle
from sklearn.cluster import k_means_
import itertools
import sklearn

def findAreaDistribution(img):
    '''
    findAreaDistribution(img)--> sizearray, containing area of all objects
    '''
    uniq = np.unique(img)
    uniq = uniq[uniq>0]
    sizearray = np.zeros_like(uniq)
    for ii, jj in enumerate(uniq):
        sizearray[ii] = np.sum(img==jj)
    return(sizearray)

def removeBoundaryLabel(labelimg):
    '''
       removeBoundaryLabel(labelimg)-> outimg boundary labels removed
    '''    
    
    boundlab = np.unique( np.concatenate(( labelimg[0,:] , labelimg[:,0] ,labelimg[-1,:], labelimg[:,-1])  ) )
    boundlab = boundlab[boundlab>0]
    for ii in boundlab:
        labelimg[labelimg==ii]=0
    return(labelimg)



def labelDilation(img, window):
    
    '''labelDilation(img, window) -->dilimage
    
    input (labeled image, window size)
    
    '''

    
    wsize = window/2
    gt = img
    dilimage = np.copy(img)
    for ii in np.arange(window, gt.shape[0]-window,1 ):
        for jj in np.arange(window, gt.shape[1]-window,1 ):
            if(np.sum(gt[ii-wsize:ii+wsize+1,jj-wsize:jj+wsize+1]) !=0 and gt[ii,jj]==0):
                dilimage[ii,jj] = findMaxOccurance(gt[ii-wsize:ii+wsize+1,jj-wsize:jj+wsize+1])
    return(dilimage)


def evaluateSegmentation2(glabel, bacteriaregion):
    '''
    evaluateSegmentation2(glabel, bacteriaregion)--> recall, precision, fmeasure, dicescore
     version2
    '''
    labelvals = np.unique(glabel)
    baclabels = np.unique(bacteriaregion)
    baclabels = baclabels[baclabels>0]
    visited = np.zeros_like(baclabels)
    
    recall = np.zeros(labelvals.shape[0])
    precision = np.zeros(labelvals.shape[0])
    accuracy = np.zeros(labelvals.shape[0])
    fmeasure = np.zeros(labelvals.shape[0])
    dicescore =np.zeros(labelvals.shape[0])
    
    for ii in np.arange(1,labelvals.shape[0]):
        temp = glabel == labelvals[ii]
        temparray = bacteriaregion[temp]
        vals = np.unique(temparray)
        countval = np.zeros((vals.shape))
        if (vals.shape[0] > 1):
            for jj in range(vals.shape[0]):
                countval[jj] = temparray.tolist().count(vals[jj])
        maxoverlaplabel = vals[countval.argmax()]
        if(maxoverlaplabel!=0):
            coverageingt = temparray.tolist().count(maxoverlaplabel)/np.float64(temparray.shape[0])
            bacloc = np.argwhere(bacteriaregion == maxoverlaplabel)
            coverageinbac = temp[bacloc[:,0],bacloc[:,1]].tolist().count(True)/np.float64(bacloc.shape[0])
            #if (coverageingt > 0.6 and coverageinbac > 0.6):
            if (coverageingt > 0.0 and coverageinbac > 0.0): 
                
                visited[np.argwhere(baclabels==maxoverlaplabel)[0][0]] = 1
                
                tp = np.float64(temparray.tolist().count(maxoverlaplabel))
                fn = np.float64(temparray.shape[0] - tp)
                fp = np.float64(bacloc.shape[0] - tp)
                recall[ii] = tp/(tp + fn)
                precision[ii] = tp/(tp + fp)
                accuracy[ii] = tp/(tp + fn + fp)
                fmeasure[ii] = (2*precision[ii]*recall[ii])/(precision[ii] + recall[ii])
                dicescore[ii] = 2*tp/(fp + fn + 2*tp)
    
    for ii in baclabels[np.argwhere(visited!=1)]:
        recall = np.append(recall,0)
        precision = np.append(precision,0)
        accuracy = np.append(accuracy,0)
        fmeasure = np.append(fmeasure,0)
        dicescore = np.append(dicescore,0)

    return(recall, precision, fmeasure, dicescore)


def findAverageimage(img1, img2):
    '''
    findAverageimage(img1, img2)--> average image
    
    '''
    row,col = img1.shape
    param1 = ellipseParam(np.argwhere(img1))
    param2 = ellipseParam(np.argwhere(img2))
    
    avgorientation = (param1[4][0] + param2[4][0])/2
    
    pos1 = np.argwhere(img1).mean(0)    
    pos2 = np.argwhere(img2).mean(0)
        
    avgpos = np.int32((pos1+pos2)/2)
        
    flatimg1 = rotateImage(img1,-param1[4])
    flatimg2 = rotateImage(img2,-param2[4])
    
    flatcoord1 = np.argwhere(flatimg1)
    newcoord1 = np.int32(flatcoord1 + (-pos1+avgpos) )
    newimg1 = np.zeros_like(img1)
    
    
    flatcoord2 = np.argwhere(flatimg2)
    newcoord2 = np.int32(flatcoord2 + (-pos2+avgpos) )
    newimg2 = np.zeros_like(img2)
    
    if((newcoord2[:,0]>=row).sum() >0 or (newcoord2[:,1]>=col).sum() or (newcoord1[:,0]>=row).sum() >0 or (newcoord1[:,1]>=col).sum() ):
        finalimg = img1
    else:
        newimg1[newcoord1[:,0], newcoord1[:,1]] = 1
        newimg2[newcoord2[:,0], newcoord2[:,1]] = 1
        
        signdist1 = cv2.distanceTransform(np.uint8(newimg1),2,3) - cv2.distanceTransform(np.uint8(( ~(newimg1==1)) ) ,2,3) 
        signdist2 = cv2.distanceTransform(np.uint8(newimg2),2,3) - cv2.distanceTransform(np.uint8(( ~(newimg2==1)) ) ,2,3) 
        
        avgsign = signdist1 + signdist2
        
        finalimg =  rotateImage(avgsign>0,(avgorientation ) )
    
    return(finalimg)



def rotateImage(img1,angle):
    '''
    rotateImage(img,angle)--> roatated image
    angle in radians
    '''
    row, col = img1.shape
    pos1 = np.argwhere(img1).mean(0)
    
    cencoord1 = np.argwhere(img1)-pos1
    
    
    rotcencoord1 = np.zeros_like(cencoord1)
    
    angleofrect = angle
    
    rotcencoord1[:, 0] = cencoord1[:, 1] * np.math.sin(angleofrect) + cencoord1[:, 0] * np.math.cos(angleofrect)
    rotcencoord1[:, 1] = cencoord1[:, 1] * np.math.cos(angleofrect) - cencoord1[:, 0] * np.math.sin(angleofrect)
    
    rotcencoord1 = np.int32(rotcencoord1 + pos1)
    
    newimg = np.zeros_like(img1)
    
        
    rotcencoord1 = rotcencoord1[rotcencoord1[:,0]<row,: ]
    rotcencoord1 = rotcencoord1[rotcencoord1[:,1]<col,: ]

    newimg[rotcencoord1[:,0] , rotcencoord1[:,1]] = 1
    
    newimg = sp.ndimage.binary_fill_holes(newimg)

    return(newimg)


def cenMoment(boundcoord, p, q):
    '''
    cenMoment(np.ndarray[DTYPE_t, ndim=2] boundcoord, p, q) -> p,q th order central moment
    '''
    return(np.sum((boundcoord[:,0]-boundcoord[:,0].mean())**p * (boundcoord[:,1]-boundcoord[:,1].mean())**q))
    

def ellipseParam( boundcoord):
    '''
    ellipseParam( boundcoord)-->params
    params = [majoraxis, minoraxis, area, circumference,orientation(in 4th quadrant)]
    '''
     
    params = np.zeros((5,1))
    m00 = cenMoment(boundcoord, 0, 0)
    m11 = cenMoment(boundcoord, 1, 1)
    m02 = cenMoment(boundcoord, 0, 2)
    m20 = cenMoment(boundcoord, 2, 0)

    eigvals, eigvec = np.linalg.eig([[m02,m11],[m11,m20]])
    
    maxindex = np.argmax(eigvals)
    minindex = np.argmin(eigvals)
    
    majoraxis = 4*(eigvals[maxindex]/m00)**.5
    minoraxis = 4*(eigvals[minindex]/m00)**.5
    
    a = majoraxis/2
    b = minoraxis/2
    
    area = np.pi*a*b
    circumference = np.pi*(3*(a+b)- np.sqrt(10*a*b + 3*(a**2 + b**2)))
    orientation = 0.5*np.math.atan2(2*m11, (m20-m02))
    
    params[0] = majoraxis
    params[1] = minoraxis
    params[2] = area
    params[3] = circumference
    params[4] = orientation
#    orientation = 0.5*np.math.atan2((m20-m02),2*m11)  
    
#    orientation =  np.math.degrees(np.arctan(2*m11/(m20-m02))/2)
    return(params)

def findMaxOccurance(data):
    '''
    findMaxOccurance(data)-->maxoccur
    finds the element which repeats maximum
    '''
    uniqdata = np.unique(data)
    uniqdata = uniqdata[uniqdata>0]
    sizelist = []
    for ii in range(uniqdata.shape[0]):
        sizelist.append(np.sum(data==uniqdata[ii]))
    maxoccur = uniqdata[np.argmax(sizelist)]
    return(maxoccur)



def kmeanCluster(img, clusters):
    '''
     kmeanCluster(img, clusters) --> clustout, centroids:
   
    '''    
    
    kmean = sklearn.cluster.KMeans(clusters)
    tempimg = img.reshape(img.shape[0]*img.shape[1],1)
    clusterimg = kmean.fit_predict(tempimg)
    clustout = clusterimg.reshape(img.shape)
    centroids = kmean.cluster_centers_
    return(clustout, centroids)

def mergeCells2(prevobj, currimg, jimg, nextlab):
    '''
    mergeCells2(prevobj, currimg, jimg, nextlab) --> outimg
    '''
    maxlab = currimg.max()
    overlapjoin = np.unique(prevobj*jimg)
    overlapjoin = overlapjoin[overlapjoin >0] 
    
    comb = findCombinations(nextlab.shape[0])
    countval =[]
    #merging segments based on minimum error compared to previous image    
    for ii in range(np.size(comb)):
        currobj = np.zeros_like(currimg)
        for jj in range(np.size(comb[ii])):
            currobj = currobj + (currimg == nextlab[comb[ii][jj]-1])
        countval.append( np.sum((prevobj>0)-(currobj>0)))
    
    mincount = np.argwhere(countval == np.min(countval))[0][0]
    currobj = np.zeros_like(currimg)
    
    for jj in range(np.size(comb[mincount])):
        currobj = currobj + (currimg == nextlab[comb[mincount][jj]-1])
#    coord = np.argwhere(currobj>0)
    finlab = np.unique(currobj*currimg)
    finlab = finlab[finlab>0]
    
    currlabimg = currobj*currimg
    # adding join label value to the obtained combined segment to fill inside
    for ii in range(overlapjoin.shape[0]):
        tempimg = jimg==overlapjoin[ii]
        labj = np.unique((skmorph.binary_dilation(tempimg,np.ones((3,3)))-tempimg)*currlabimg)
        labj = labj[labj>0]
        if (labj.shape[0]==2):
            currobj = currobj + (jimg==overlapjoin[ii])

    objlab = np.unique(currlabimg)            
    objlab = objlab[objlab>0]
    #correction to output image
    outimg = np.copy(currimg)
    if ((np.sum((currobj>0)-(prevobj>0))*1.0)/np.sum((prevobj>0))<0.2):
        for ii in range(objlab.shape[0]):        
            outimg[outimg == objlab[ii]] = 0
        outimg = outimg + ((currobj>0)*(maxlab+1))

    return outimg, maxlab+1


def findCombinations(count):
    '''
    findCombinations(count) --> listval, list of all combinations
    '''

    listval = []
    for ii in np.arange(1,count+1):
         aa = itertools.combinations(np.arange(1,count+1).tolist(),ii)
         for jj in range(np.math.factorial(count)/(np.math.factorial(ii)*np.math.factorial(count-ii))):
             listval.append(aa.next())
    return(listval)
    
    
def mergeCell(labimg, jimg, labvals):
    '''
     mergeCell(labimg, jimg, labvals):--> newimg, label of new merged cell
    '''
    #    labvals = [298,279]
    #    labimg = imgarray[:,:,34]
    maxlab = labimg.max()
    
    coord = np.argwhere(labimg == labvals[0])
    centroid = np.sum(coord,0)/coord.shape[0]
    
        
    combseg = (labimg==labvals[0])+(labimg==labvals[1]) +(jimg>0)
    
    labelimg, num = label(combseg,np.ones((3,3)))
    newimg = np.copy(labimg)
    if (np.sum(labelimg == labelimg[centroid[0],centroid[1]])>0 and labelimg[centroid[0],centroid[1]] ):
        newimg[newimg==labvals[0]] = 0
        newimg[newimg==labvals[1]] = 0    
        newimg = newimg + ((labelimg == labelimg[centroid[0],centroid[1]])>0 )*(maxlab+1)
    
    return(newimg, maxlab+1)

def fitEllipseImage(data):
    '''
    fitEllipse(data) -->img,ellipseparams
    return fitted ellipse on data
    '''
    tempbound = data - skmorph.binary_erosion(data,np.ones((3,3)))    
    boundcoord = np.argwhere(tempbound == 1)    
    ellipsecoord = cv2.fitEllipse(boundcoord) 
       
    center = (int(ellipsecoord[0][1]), int(ellipsecoord[0][0]))
    axes = (int(ellipsecoord[1][1]/2), int(ellipsecoord[1][0])/2)
    
    img = np.uint8(np.zeros_like(data))
    cv2.ellipse(img,center,axes, 360-int(ellipsecoord[2]),0,360,1,-1)
    return(img, ellipsecoord)   



def findConvexImage(data):
    '''findConvexImage(data) --> convex image of given object '''
    boundary = data - skmorph.binary_erosion(data,np.ones((3,3))) 
    tempimg = np.zeros((data.shape))
    convexpoints = cv2.convexHull(np.argwhere(boundary))
    revconvexpoints = np.zeros((convexpoints.shape), np.int32)
    revconvexpoints[:, 0, 0] = convexpoints[:, 0, 1]
    revconvexpoints[:, 0, 1] = convexpoints[:, 0, 0]
    cv2.fillConvexPoly(tempimg, revconvexpoints, 1)
#    convexarea = np.argwhere(tempimg).shape[0]
    return(tempimg)




def splitCell(inputimg, lbl):
    '''splitCell(inputimg,lbl)-->labeled image
    watershed segmentation to split region
    '''
    lbl = (((inputimg*lbl*3)) +(~(inputimg>0)))
    lbl = lbl.astype(np.int32)
    row, col=inputimg.shape    
    gray3channel = np.zeros((row, col,3),dtype = np.uint8 ) 
    gray3channel[:,:,0] = gray3channel[:,:,1] = gray3channel[:,:,2] = (np.uint8(~(inputimg>0))*255)
    cv2.watershed(gray3channel, lbl)
    lbl[skmorph.binary_dilation(lbl==-1,np.ones((3,3)))] = 0
    return(lbl/3)


def checkcollinear(partialsegments):
    '''
    checkcollinear(partialsegments) return 0 if not collinear and 1
    if collinear ie parallel cells lying in a straight line
    '''
#    plabel, pnum = label(partialsegments,np.ones((3,3)))
    uniqlab = np.unique(partialsegments)
    uniqlab = uniqlab[uniqlab>0]
    lab1 = partialsegments == uniqlab[0]
    lab2 = partialsegments == uniqlab[1]
    
    lab1coord = np.float32(np.argwhere(lab1==1))
    lab1coordmean = np.mean(lab1coord,0)
    lab1coord[:, 0] = lab1coord[:, 0] - lab1coordmean[0]
    lab1coord[:, 1] = lab1coord[:, 1] - lab1coordmean[1]
    
    imgrotcoordorigin = np.zeros_like(lab1coord)
    imgrotcoordorigin[:, 0] = lab1coord[:, 1] * np.math.sin(np.pi/2) + lab1coord[:, 0] * np.math.cos(np.pi/2)
    imgrotcoordorigin[:, 1] = lab1coord[:, 1] * np.math.cos(np.pi/2) - lab1coord[:, 0] * np.math.sin(np.pi/2)
    
    imgrotcoordorigin[:, 0] = imgrotcoordorigin[:, 0] + lab1coordmean[0]
    imgrotcoordorigin[:, 1] = imgrotcoordorigin[:, 1] + lab1coordmean[1]
    imgrotcoordorigin = np.int32(imgrotcoordorigin)
    
    newlab1 = np.zeros_like(lab1)
    for ii in range(imgrotcoordorigin.shape[0]):
        if(imgrotcoordorigin[ii, 0]>0 and imgrotcoordorigin[ii, 0]<partialsegments.shape[0] and imgrotcoordorigin[ii, 1]>0 and imgrotcoordorigin[ii, 1]<partialsegments.shape[1]):
            newlab1[imgrotcoordorigin[ii, 0],imgrotcoordorigin[ii, 1]] = 1
    
    overlap = (newlab1)*1 + (lab2)*1
    collinear = 0
    if(np.sum(overlap == 2) == 0):
        collinear = 1
    return (collinear)




def randomizeLabel(labelimg):    
    index = np.unique(labelimg)
    orindx = np.copy( index)
    np.random.shuffle(index[1:])
    randlabel = np.zeros((labelimg.shape), labelimg.dtype)
    for ii in np.arange(1, index.shape[0]):
        randlabel[labelimg == orindx[ii]] = index[ii]
    return(randlabel)


def createSortindex(radius):
    circle = createCircleEdge(radius)
    circindex = np.argwhere(circle)
    circindex[:, 0] = circindex[:, 0] - circindex[:, 0].mean()
    circindex[:, 1] = circindex[:, 1] - circindex[:, 1].mean()
    quadrindex = np.zeros((circindex.shape[0], 4))
    quadrindex[:, 0:2] = circindex
    quadrindex[:, 2] = np.angle(quadrindex[:,0]+1j*quadrindex[:,1],deg=True)
    sortindex =  quadrindex[np.argsort(quadrindex[:,2]), 0:2]
    return(sortindex)
    
def overlayImageFull(inputimage, randlabel):
    '''overlayImage(inputimage, randlabel)--> overlayimg full not only boundary
    '''
    tempimg = np.float32(inputimage)
    timg = (np.uint8(255*((tempimg-tempimg.min())/tempimg.max())))
    
    ovarlayimg = np.zeros((inputimage.shape[0],inputimage.shape[1],3),np.uint8)
    ovarlayimg[:,:,0] = np.copy(timg)
    ovarlayimg[:,:,1] = np.copy(timg)
    ovarlayimg[:,:,2] = np.copy(timg)
    baclabels = np.unique(randlabel)
    for ii in range(baclabels.shape[0]):
        if(baclabels[ii]!=0):
            seplabel = randlabel==baclabels[ii]
            boundary = seplabel
            ovarlayimg[:,:,0] = ovarlayimg[:,:,0]+(boundary*0) -(timg*boundary)
            ovarlayimg[:,:,1] = ovarlayimg[:,:,1]+(boundary*0) -(timg*boundary)
            ovarlayimg[:,:,2] = ovarlayimg[:,:,2]+(boundary*255) -(timg*boundary)
    return(ovarlayimg)


def overlayImage(inputimage, randlabel):
    '''overlayImage(inputimage, randlabel)--> overlayimg
    '''
    tempimg = np.float32(inputimage)
    timg = (np.uint8(127*((tempimg-tempimg.min())/tempimg.max())))
#    timg = (np.uint8(150*((tempimg-tempimg.min())/tempimg.max())))

#    timg = (np.uint8(255*((tempimg-tempimg.min())/tempimg.max())))

    ovarlayimg = np.zeros((inputimage.shape[0],inputimage.shape[1],3),np.uint8)
    ovarlayimg[:,:,0] = np.copy(timg)
    ovarlayimg[:,:,1] = np.copy(timg)
    ovarlayimg[:,:,2] = np.copy(timg)
    baclabels = np.unique(randlabel)
    for ii in range(baclabels.shape[0]):
        if(baclabels[ii]!=0):
            seplabel = randlabel==baclabels[ii]
            boundary = seplabel-skmorph.binary_erosion(seplabel,np.ones((3,3)))
#            altbound = boundary
#            altbound[0:-1:2,0:-1:2] = 0
            ovarlayimg[:,:,0] = ovarlayimg[:,:,0]+(boundary*0) -(timg*boundary)  #+altbound
            ovarlayimg[:,:,1] = ovarlayimg[:,:,1]+(boundary*255) -(timg*boundary)
            ovarlayimg[:,:,2] = ovarlayimg[:,:,2]+(boundary*0) -(timg*boundary)  # +altbound
    return(ovarlayimg)

def perpendicularLine(startpt, endpt,lengthofline):  
    '''perpendicularLine(lengthofrect, widthofrect, angleofrect )
    '''
    img = np.zeros((200,200))
    imgout = np.zeros((200,200))
    rotimg = np.zeros((200,200), np.uint8)
    #lengthofrect =  np.int32(np.sqrt((startpt[0] - endpt[0])**2 + (startpt[1] - endpt[1])**2))
    angleofrect = np.math.atan2((startpt[0] - endpt[0]), (startpt[1] - endpt[1]))
#    widthofrect = 25
    centroid = np.int32((startpt + endpt)/2)
    img[ 100 - lengthofline/2: 100 + lengthofline/2,100] = 1
    imgcoord = np.float64(np.argwhere(img))
    imgcoordorigin = np.zeros((imgcoord.shape))
    imgrotcoord = np.zeros((imgcoord.shape))
    imgrotcoordorigin = np.zeros((imgcoord.shape))
    rowmean = imgcoord[:,0].mean()
    colmean = imgcoord[:,1].mean()
    imgcoordorigin[:, 0] = imgcoord[:, 0] - rowmean
    imgcoordorigin[:, 1] = imgcoord[:, 1] - colmean
    
    imgrotcoordorigin[:, 0] = imgcoordorigin[:, 1] * np.math.sin(angleofrect) + imgcoordorigin[:, 0] * np.math.cos(angleofrect)
    imgrotcoordorigin[:, 1] = imgcoordorigin[:, 1] * np.math.cos(angleofrect) - imgcoordorigin[:, 0] * np.math.sin(angleofrect)
    imgrotcoord[:, 0] = imgrotcoordorigin[:, 0] + 100
    imgrotcoord[:, 1] = imgrotcoordorigin[:, 1] + 100
    imgrotcoord = np.int64(imgrotcoord)
    strelplus = np.ones((3,3),np.uint8)
    strelplus[0,0] = strelplus[2,0] = strelplus[0,2] = strelplus[2,2] = 0
    rotimg[imgrotcoord[:,0], imgrotcoord[:,1]] = 1
    rotimg = np.uint8(rotimg)
    rotimg = skmorph.closing(rotimg, np.ones((3, 3), np.uint8))
    
    return(rotimg)





def createline(imagesize, startpt, endpt ):  
    '''createline(imagesize, startpt, endpt ):
    '''

    img = np.zeros((imagesize[0], imagesize[1]))
    imgout = np.zeros((imagesize[0], imagesize[1]))
    rotimg = np.zeros((imagesize[0], imagesize[1]))
    lengthofrect =  np.int32(np.sqrt((startpt[0] - endpt[0])**2 + (startpt[1] - endpt[1])**2)) 
    lengthofrect = lengthofrect + 20
    angleofrect = np.math.atan2((startpt[0] - endpt[0]), (startpt[1] - endpt[1]))
    centroid = np.int32((startpt + endpt)/2)
    widthofrect = 2
    img[centroid[0] - np.int32(widthofrect/2) : centroid[0] +np.int32( widthofrect/2), centroid[1] - np.int32(lengthofrect/2): centroid[1] + np.int32(lengthofrect/2)] = 1
    
    
    imgcoord = np.float64(np.argwhere(img))
    imgcoordorigin = np.zeros((imgcoord.shape))
    imgrotcoord = np.zeros((imgcoord.shape))
    imgrotcoordorigin = np.zeros((imgcoord.shape))
    rowmean = imgcoord[:,0].mean()
    colmean = imgcoord[:,1].mean()
    imgcoordorigin[:, 0] = imgcoord[:, 0] - rowmean
    imgcoordorigin[:, 1] = imgcoord[:, 1] - colmean
    
    imgrotcoordorigin[:, 0] = imgcoordorigin[:, 1] * np.math.sin(angleofrect) + imgcoordorigin[:, 0] * np.math.cos(angleofrect)
    imgrotcoordorigin[:, 1] = imgcoordorigin[:, 1] * np.math.cos(angleofrect) - imgcoordorigin[:, 0] * np.math.sin(angleofrect)
    imgrotcoord[:, 0] = imgrotcoordorigin[:, 0] + centroid[0]
    imgrotcoord[:, 1] = imgrotcoordorigin[:, 1] + centroid[1]
    imgrotcoord = np.int32(imgrotcoord)
    strelplus = np.ones((3,3),np.uint8)
    strelplus[0,0] = strelplus[2,0] = strelplus[0,2] = strelplus[2,2] = 0
    rotimg[imgrotcoord[:,0], imgrotcoord[:,1]] = 1
    rotimg = np.uint8(rotimg)
    rotimg = skmorph.closing(rotimg, np.ones((3, 3), np.uint8))
    
    return(rotimg)


def evaluateSegmentation(glabel, bacteriaregion):
    '''evaluateSegmentation(glabel, bacteriaregion) --> (recall, precision, fmeasure, dicescore)'''
    labelvals = np.unique(glabel)
    recall = np.zeros(labelvals.shape[0])
    precision = np.zeros(labelvals.shape[0])
    accuracy = np.zeros(labelvals.shape[0])
    fmeasure = np.zeros(labelvals.shape[0])
    dicescore =np.zeros(labelvals.shape[0])
    
    for ii in np.arange(1,labelvals.shape[0]):
        temp = glabel == labelvals[ii]
        temparray = bacteriaregion[temp]
        vals = np.unique(temparray)
        countval = np.zeros((vals.shape))
        if (vals.shape[0] > 1):
            for jj in range(vals.shape[0]):
                countval[jj] = temparray.tolist().count(vals[jj])
        maxoverlaplabel = vals[countval.argmax()]
        if(maxoverlaplabel!=0):
            coverageingt = temparray.tolist().count(maxoverlaplabel)/np.float64(temparray.shape[0])
            bacloc = np.argwhere(bacteriaregion == maxoverlaplabel)
            coverageinbac = temp[bacloc[:,0],bacloc[:,1]].tolist().count(True)/np.float64(bacloc.shape[0])
            #if (coverageingt > 0.6 and coverageinbac > 0.6):
            if (coverageingt > 0.0 and coverageinbac > 0.0):    
                tp = np.float64(temparray.tolist().count(maxoverlaplabel))
                fn = np.float64(temparray.shape[0] - tp)
                fp = np.float64(bacloc.shape[0] - tp)
                recall[ii] = tp/(tp + fn)
                precision[ii] = tp/(tp + fp)
                accuracy[ii] = tp/(tp + fn + fp)
                fmeasure[ii] = (2*precision[ii]*recall[ii])/(precision[ii] + recall[ii])
                dicescore[ii] = 2*tp/(fp + fn + 2*tp)
    return(recall, precision, fmeasure, dicescore)

def fillIrregular(irrimg, newdata):
    newdata[irrimg==0] = 0
    WINSIZE = 1
    dataloc = np.argwhere(irrimg > 0)
    for ii in range(dataloc.shape[0]):
        if(newdata[dataloc[ii,0], dataloc[ii, 1]] == 0):
            win = newdata[dataloc[ii,0] -WINSIZE :dataloc[ii,0] +WINSIZE, dataloc[ii, 1] - WINSIZE :dataloc[ii, 1] + WINSIZE ]
            newdata[dataloc[ii,0], dataloc[ii, 1]] = win.max()
    return(newdata)

def separateBacteria(data):
    '''separateBacteria(data)-> separatedimage '''
    finalimg = np.copy(data)    
    RADIUS = 9
    
    skeldata = skmorph.skeletonize(data)
    skeldata = prunSkeleton(5,skeldata)
    skelcoord = np.argwhere(skeldata)
    weightarray = np.zeros(skelcoord.shape)
    lengtharray = np.zeros(skelcoord.shape[0])
    for ii in range(skelcoord.shape[0]):
        tempskel = np.copy(skeldata)
        tempskel[skelcoord[ii, 0], skelcoord[ii, 1]] = 0
        labelimg, nums = label(tempskel,np.ones((3,3)))
        #print(nums)
        if (nums == 2):
            skel1 = labelimg == 1
            skelimg1 = skeletontoImage(skel1, RADIUS)
            #skel1 = cb.prunSkeleton(5, skel1)
            weightarray[ii,0] = np.argwhere(skelimg1).shape[0]/findConvexarea(skelimg1)        
            skel2 = labelimg == 2
            skelimg2 = skeletontoImage(skel2, RADIUS)
            #skel2 = cb.prunSkeleton(5, skel2)
            weightarray[ii, 1] = np.argwhere(skelimg2).shape[0]/findConvexarea(skelimg2)  
            lengtharray[ii] = (np.argwhere(skel1).shape[0] + np.argwhere(skel2).shape[0])/(np.abs(np.argwhere(skel1).shape[0] - np.argwhere(skel2).shape[0]) + 1)
    sumval = weightarray[:, 0] + weightarray[:, 1] 
#    tot = 0.0*(lengtharray/lengtharray.max()) + 1.0 *(sumval/sumval.max())
    tot = 0.9*(lengtharray/lengtharray.max()) + 0.1 *(sumval/sumval.max())
    optimloc = np.argwhere(tot == tot.max())
    tempskel = np.copy(skeldata)
    tempskel[skelcoord[optimloc, 0], skelcoord[optimloc, 1]] = 0
    labelimg, nums = label(tempskel,np.ones((3,3)))
    if (nums == 2):    
        skelimg1 = skeletontoImage( labelimg == 1, RADIUS)
        skelimg2 = skeletontoImage( labelimg == 2, RADIUS)
#        skelimg1 = skeletontoImage(prunSkeleton(5, labelimg == 1), RADIUS)
#        skelimg2 = skeletontoImage(prunSkeleton(5, labelimg == 2), RADIUS)
        finalimg = skelimg1 + 2*skelimg2
        common = np.argwhere(finalimg == 3)
        #tempskel[common[:, 0], common[:, 1]] = 0   
        labelimg, nums = label(tempskel,np.ones((3,3)))
        skelimg1 = skeletontoImage(prunSkeleton(1, labelimg == 1), RADIUS)
        skelimg2 = skeletontoImage(prunSkeleton(1, labelimg == 2), RADIUS)
        finalimg = skelimg1 + 2*skelimg2
    #finalimg[common[:, 0], common[:, 1]] = 0
    return(finalimg, optimloc)
    
#    return(skelimg1, skelimg2)
def skeletontoImage(skeldata, radius):
    ''' convert skeleton to image '''
    #radius = 10
    circle = createCircle(radius)
    circlecoord = np.argwhere(circle)
    circlecoord = circlecoord - radius
    skelcoord = np.argwhere(skeldata)
    newimg = np.zeros(skeldata.shape)
    for ii in range(skelcoord.shape[0]):
        ncoord = skelcoord[ii] + circlecoord
        newimg[ncoord[:,0], ncoord[:,1]] = 1
    return(newimg)


def segmentIrregular(randlabel, irregular, orimg):
    ''' segmentIrregular(randlabel, irregular)'''
    newdata = np.zeros((randlabel.shape))
    for ii in range(irregular.shape[0]):
        data = randlabel == irregular[ii]
        pskeleton = prunSkeleton(25, data)
        endofskeleton = findSkeletonEndpoints(pskeleton)
        while(np.argwhere(pskeleton).shape[0]>50):
            #endofskeleton = findSkeletonEndpoints(pskeleton)
            allpts = np.argwhere(pskeleton)
            weightval = np.zeros((allpts.shape[0], 1))
            for ii in range(allpts.shape[0]):
                weightval[ii] = bacteriaFitCriteria(pskeleton, endofskeleton[0], allpts[ii], orimg)
            
            temp, pskeleton = bacteriaFit(pskeleton, endofskeleton[0], allpts[np.argwhere(weightval==weightval.max())[0][0]], orimg, data)
            newdata = newdata + temp*(ii + 1)
    return(newdata)

def bacteriaFit(pskeleton, startpt, endpt, orimg, data):
    '''bacteriaFit(pskeleton, startpt, endpt, orimg, data) --> return segmented bacteria region, newskeleton '''
    bacimg = createBacteria2(pskeleton.shape, startpt, endpt)
    
    coords = np.argwhere(bacimg)
    intensityval = orimg[coords[:, 0], coords[:, 1]]
    intensityval = intensityval.reshape((intensityval.shape[0],1))
    
    kmeanobj = k_means_.KMeans(2)
    kmeanobj.fit(intensityval)
    bacteriaregion = kmeanobj.labels_ == np.argwhere(kmeanobj.cluster_centers_ == kmeanobj.cluster_centers_.min())[0][0]
    
    arearatio = np.float64(np.argwhere(bacteriaregion).shape[0])/np.float32(bacteriaregion.shape[0])
    
    newskeleton = np.copy(pskeleton)
    bacteriacoords = coords[bacteriaregion]
    tempimg = np.zeros((pskeleton.shape))
    tempimg[bacteriacoords[:, 0], bacteriacoords[:, 1]] = 1
    newskeleton = newskeleton * np.float32(tempimg == 0)
    return(tempimg, newskeleton)

def bacteriaFitCriteria(pskeleton, startpt, endpt, orimg):
    ''' bacteriaFitCriteria(pskeleton, startpt, endpt, orimg) --> return weight value'''
    bacimg = createBacteria2(pskeleton.shape, startpt, endpt)
    
    coords = np.argwhere(bacimg)
    intensityval = orimg[coords[:, 0], coords[:, 1]]
    intensityval = intensityval.reshape((intensityval.shape[0],1))
    
    kmeanobj = k_means_.KMeans(2)
    kmeanobj.fit(intensityval)
    bacteriaregion = kmeanobj.labels_ == np.argwhere(kmeanobj.cluster_centers_ == kmeanobj.cluster_centers_.min())[0][0]
    
    arearatio = np.float64(np.argwhere(bacteriaregion).shape[0])/np.float32(bacteriaregion.shape[0])
    
    newskeleton = np.copy(pskeleton)
    bacteriacoords = coords[bacteriaregion]
    tempimg = np.zeros((pskeleton.shape))
    tempimg[bacteriacoords[:, 0], bacteriacoords[:, 1]] = 1
    skelcontrib = tempimg*pskeleton
    skeletonratio = np.float64(np.argwhere(skelcontrib).shape[0])/np.float64(np.argwhere(pskeleton).shape[0])
    sumval = arearatio + skeletonratio
    return(sumval)

def createBacteria2(imagesize, startpt, endpt ):  
    '''createBacteria2(imagesize, startpt, endpt ):
    '''
    
    img = np.zeros((imagesize[0], imagesize[1]))
    imgout = np.zeros((imagesize[0], imagesize[1]))
    rotimg = np.zeros((imagesize[0], imagesize[1]))
    lengthofrect =  np.int32(np.sqrt((startpt[0] - endpt[0])**2 + (startpt[1] - endpt[1])**2)) 
    lengthofrect = lengthofrect + 20
    angleofrect = np.math.atan2((startpt[0] - endpt[0]), (startpt[1] - endpt[1]))
    centroid = (startpt + endpt)/2
    widthofrect = 14
    img[centroid[0] - widthofrect/2 : centroid[0] + widthofrect/2, centroid[1] - lengthofrect/2: centroid[1] + lengthofrect/2] = 1
    
    gau = gauss_kern(img)
    #Img_smooth = signal.convolve(Img,g,mode='same')
    imgfft = np.fft.rfft2(img)
    gfft = np.fft.rfft2(gau)
    fftimage = np.multiply(imgfft, gfft)
    img_smooth =np.real(np.fft.ifftshift( np.fft.irfft2(fftimage)))
    imgout = img_smooth>0.3
    
    imgcoord = np.float64(np.argwhere(imgout))
    imgcoordorigin = np.zeros((imgcoord.shape))
    imgrotcoord = np.zeros((imgcoord.shape))
    imgrotcoordorigin = np.zeros((imgcoord.shape))
    rowmean = imgcoord[:,0].mean()
    colmean = imgcoord[:,1].mean()
    imgcoordorigin[:, 0] = imgcoord[:, 0] - rowmean
    imgcoordorigin[:, 1] = imgcoord[:, 1] - colmean
    
    imgrotcoordorigin[:, 0] = imgcoordorigin[:, 1] * np.math.sin(angleofrect) + imgcoordorigin[:, 0] * np.math.cos(angleofrect)
    imgrotcoordorigin[:, 1] = imgcoordorigin[:, 1] * np.math.cos(angleofrect) - imgcoordorigin[:, 0] * np.math.sin(angleofrect)
    imgrotcoord[:, 0] = imgrotcoordorigin[:, 0] + centroid[0]
    imgrotcoord[:, 1] = imgrotcoordorigin[:, 1] + centroid[1]
    imgrotcoord = np.int64(imgrotcoord)
    strelplus = np.ones((3,3),np.uint8)
    strelplus[0,0] = strelplus[2,0] = strelplus[0,2] = strelplus[2,2] = 0
    rotimg[imgrotcoord[:,0], imgrotcoord[:,1]] = 1
    rotimg = np.uint8(rotimg)
    rotimg = skmorph.closing(rotimg, np.ones((3, 3), np.uint8))
    
    return(rotimg)


def breakSkeleton(orimg, pskeleton):
    '''breakSkeleton(orimg, pskeleton)--> return separated skeleton '''
    skcoord=np.argwhere(pskeleton)
    neighsize = 10
    temp = orimg[skcoord[0][0] - neighsize : skcoord[0][0] + neighsize, skcoord[0][1] - neighsize : skcoord[0][1] + neighsize ] 
    temp = np.float32(temp)
    dy, dx = gaussDerivative(temp, 3)
    dirderiv = np.zeros((dx.shape))
    for ii in range(dx.shape[0]):
        for jj in range(dx.shape[1]):
            dirderiv[ii, jj] = np.math.atan2(dy[ii, jj], dx[ii, jj])
    meanint = np.zeros((skcoord.shape[0]))
    for ii in range(skcoord.shape[0]):
        temp = orimg[skcoord[ii][0] - neighsize : skcoord[ii][0] + neighsize, skcoord[ii][1] - neighsize : skcoord[ii][1] + neighsize ] 
        meanint[ii] = temp.mean()
    lowval = np.argwhere(meanint > meanint.max()*0.95)
    for ii in range(lowval.shape[0]):
        pskeleton[skcoord[lowval][ii][0][0], skcoord[lowval][ii][0][1]] = 0
    return(pskeleton)
    
def gaussDerivative(Img, sigma):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    h2,h1 = Img.shape    
    x, y = np.mgrid[0:h2, 0:h1]
    x = x-h2/2
    y = y-h1/2
    #    sigma = 5.5
    #    sigma = 15
    g = np.exp( -( x**2 + y**2 ) / (2*sigma**2) )
    g = g / g.sum()
    Imgfft = np.fft.rfft2(Img)
    gfft = np.fft.rfft2(g)
    fftimage = np.multiply(Imgfft, gfft)
    Img_smooth = np.real(np.fft.ifftshift( np.fft.irfft2(fftimage)))
    Iy, Ix = np.gradient(Img_smooth)
    return(Iy, Ix)


def findSkeletonEndpoints(skeldata):
    '''findSkeletonEndpoints(skeldata) --> return skeleton endpoints '''
    endpt =[]
    skelcoord = np.argwhere(skeldata)
    for ii in range(skelcoord.shape[0]):
        if(skeldata[skelcoord[ii][0]-1:skelcoord[ii][0] + 2, skelcoord[ii][1] - 1 : skelcoord[ii][1] + 2].sum() == 2):
            endpt.append([skelcoord[ii][0], skelcoord[ii][1]])
    endpt = np.array(endpt)
    return(endpt)
    
def prunSkeleton(times, data):
    '''prunSkeleton(times) --> skeldat pruned 10 pixels and regown'''
    skeldata = skmorph.skeletonize(data)
    #prune
    initskel = np.copy(skeldata)
    for ii in range(times):
        endpts = findSkeletonEndpoints(skeldata)
        skeldata[endpts[:, 0], endpts[:, 1]] = 0
    #grow
    for ii in range(times):
        endpts = findSkeletonEndpoints(skeldata)
        for ii in range(endpts.shape[0]):
            skeldata[endpts[ii][0]-1:endpts[ii][0] + 2, endpts[ii][1] - 1 : endpts[ii][1] + 2] = 1
            skeldata = initskel * skeldata
    return(skeldata)

def createCircle(radius):
    '''createCircle(radius) --> a filled circle of given radius '''
    width = 2*np.int(radius) + 1
    mask = np.zeros((width, width)) 
    mask[radius, radius] = 1
    dt = cv2.distanceTransform(np.uint8(mask == 0), 2, 3)
    #    dt = cv2.distanceTransform(np.uint8(mask == 0), cv2.cv.CV_DIFF_L2, 3)
    circle = dt < radius
    return(circle)

def createRing(innerradius, outerradius):
    '''createRing(innerradius, outerradius) --> a filled ring between inner and outer radius '''
    width = 2*np.int(outerradius) + 1
    mask = np.zeros((width, width)) 
    mask[outerradius, outerradius] = 1
    dt = cv2.distanceTransform(np.uint8(mask == 0), 2, 3)
    #    dt = cv2.distanceTransform(np.uint8(mask == 0), cv2.cv.CV_DIFF_L2, 3)
    #circle = dt > outerradius
    circle = np.bitwise_and(dt>innerradius-1, dt<outerradius)
    return(circle)

def createCircleEdge(radius):
    '''createCircle(radius) --> a filled circle of given radius '''
    width = 2*np.int(radius) + 1
    mask = np.zeros((width, width)) 
    mask[radius, radius] = 1
    dt = cv2.distanceTransform(np.uint8(mask == 0), 2, 3)
    #    dt = cv2.distanceTransform(np.uint8(mask == 0), cv2.cv.CV_DIFF_L2, 3)
    circle = np.bitwise_and(dt>radius-1, dt<radius)
    return(circle)

def findConvexarea(data):
    data = np.uint8(data)
    '''findConvexarea(data) --> convexarea of given object '''
#    boundary = data - skmorph.binary_erosion(data,np.ones((3,3))) 
    boundary = data - cv2.erode(data,np.ones((3,3))) 
    
    tempimg = np.zeros((data.shape))
    convexpoints = cv2.convexHull(np.argwhere(boundary))
    revconvexpoints = np.zeros((convexpoints.shape), np.int32)
    revconvexpoints[:, 0, 0] = convexpoints[:, 0, 1]
    revconvexpoints[:, 0, 1] = convexpoints[:, 0, 0]
    cv2.fillConvexPoly(tempimg, revconvexpoints, 1)
    convexarea = np.argwhere(tempimg).shape[0]
    return(np.float32(convexarea))

def irregularShapeDetect(randlabel):
    '''irregularShapeDetect(randlabel) --> label of convex deficient objects '''
    objectlabel = []
    convexdeficiency = []
    alllabel = np.unique(randlabel)
    for ii in range(alllabel.shape[0]):
        if(alllabel[ii] !=0 ):
            convexdeficiency.append(np.float32(np.argwhere(randlabel == alllabel[ii]).shape[0])/findConvexarea(randlabel == alllabel[ii]))
            objectlabel.append(alllabel[ii])
    convexdeficiency = np.array(convexdeficiency)
    objectlabel = np.array(objectlabel)
    deficient = np.argwhere(convexdeficiency <0.8)
    if (deficient.shape[0] >0 ):
        return(objectlabel[deficient])
    else:
        return(0)
    

def extractProperties(data, endpt):
    '''extractProperties(data, endpt)  
    return(dicescore, imgarea, imgregion)'''
    accsize = gaussianRectangle.findSum(endpt.shape[0])
    dicescore = np.zeros((accsize))
    imgregion = np.zeros((200, 200, accsize))
    coverage = np.zeros((accsize))
    areabyconvexarea = np.zeros((accsize))
    imgarea = np.zeros((accsize))
    imgcoordinate = []
    tempcounter = 0
    for ii in range(endpt.shape[0] - 1):
        for jj in np.arange(ii + 1, endpt.shape[0]):
            print(ii, jj)
            dicescore[tempcounter], imgregion[:, :, tempcounter], coverage[tempcounter], tempval = gaussianRectangle.compare(endpt[ii], endpt[jj], data)
            imgcoordinate.append(tempval)
            tempcounter += 1
    # filtering the object parts
    for ii in range(tempcounter):
        labelimg, nums = label(imgregion[:,:,ii] > 0)
        imgarea[ii] = np.argwhere(imgregion[:,:,ii]>0).shape[0]
        if(nums > 1 or imgarea[ii] < 500):
            dicescore[ii] = 0
        convarea = findConvexarea(imgregion[:,:,ii]>0)
        areabyconvexarea[ii] = np.float64(imgarea[ii])/convarea
        return(dicescore, imgarea, imgregion)

def createEndPoints(data):
    '''createEndPoints(data)
    return(endpt)'''
    endpt =[]
    skeldata = skmorph.skeletonize(data)
    skelcoord = np.argwhere(skeldata)
    for ii in range(skelcoord.shape[0]):
        if(skeldata[skelcoord[ii][0]-1:skelcoord[ii][0] + 2, skelcoord[ii][1] - 1 : skelcoord[ii][1] + 2].sum() == 2):
            endpt.append([skelcoord[ii][0], skelcoord[ii][1]])
    endpt = np.array(endpt)
    return(endpt)

def createRandLabel(img):
    '''createRandLabel(img) , img is the classification result from ilastik
    return(randlabel)'''
    img = img == img.max()
    img = sp.ndimage.filters.median_filter(img,(3,3))
    img = skmorph.remove_small_objects(img, 500)
    labelimg, ncc = label(img,np.ones((3,3)))
    index = np.arange(1,ncc+1)
    np.random.shuffle(index)
    randlabel = np.zeros((labelimg.shape), labelimg.dtype)
    for ii in np.arange(1, ncc):
        randlabel[labelimg == ii] = index[ii]
    temprandlabel = np.copy(randlabel)
    winsize = 1
    row, col = randlabel.shape
    for iteration in range(2):
        for ii in np.arange(0 + winsize, row - winsize):
            for jj in np.arange(0 + winsize, col - winsize):
                window = randlabel[ii - winsize: ii + winsize +1, jj - winsize: jj + winsize + 1]
                if (window[winsize, winsize] == 0):
                    temprandlabel[ii, jj] = window[np.unravel_index(np.argmax(window),(window.shape))]
        randlabel = np.copy(temprandlabel)            
    return(randlabel)
#%% 
def gauss_kern(Img):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    h2,h1 = Img.shape    
    x, y = np.mgrid[0:h2, 0:h1]
    x = x-h2/2
    y = y-h1/2
    sigma = 5.5
    #    sigma = 15
    g = np.exp( -( x**2 + y**2 ) / (2*sigma**2) );
    return g / g.sum()

def myshow(img):
    
    def onClick(event):
        print(img[np.int(event.ydata),np.int(event.xdata)])     
        #    plt.close('all')
    plt.figure()    
    plt.ion() 
    plt.imshow(img,cmap='gray'), plt.show()
    fig = plt.gcf()
    #    on mouse click the value at the image location is displayed in output screen
    _ = fig.canvas.mpl_connect('button_press_event', onClick)

def myshow2(img):
    
    def onClick(event):
        print(img[np.int(event.ydata),np.int(event.xdata)])     
        #    plt.close('all')
    plt.figure()    
    plt.ion() 
    plt.imshow(img),plt.show()
    fig = plt.gcf()
    #    on mouse click the value at the image location is displayed in output screen
    _ = fig.canvas.mpl_connect('button_press_event', onClick)

def createBacteria(lengthofrect, widthofrect, angleofrect, centroid ):  
    '''createBacteria(lengthofrect, widthofrect, angleofrect )
    '''
    img = np.zeros((200,200))
    imgout = np.zeros((200,200))
    rotimg = np.zeros((200,200), np.uint8)
#    lengthofrect =  np.int32(np.sqrt((startpt[0] - endpt[0])**2 + (startpt[1] - endpt[1])**2))
#    angleofrect = np.math.atan2((startpt[0] - endpt[0]), (startpt[1] - endpt[1]))
#    widthofrect = 25
    img[100 - widthofrect/2 : 100 + widthofrect/2, 100 - lengthofrect/2: 100 + lengthofrect/2] = 1
    
    gau = gauss_kern(img)
    #Img_smooth = signal.convolve(Img,g,mode='same')
    imgfft = np.fft.rfft2(img)
    gfft = np.fft.rfft2(gau)
    fftimage = np.multiply(imgfft, gfft)
    img_smooth =np.real(np.fft.ifftshift( np.fft.irfft2(fftimage)))
    imgout = img_smooth>0.3
    
    imgcoord = np.float64(np.argwhere(imgout))
    imgcoordorigin = np.zeros((imgcoord.shape))
    imgrotcoord = np.zeros((imgcoord.shape))
    imgrotcoordorigin = np.zeros((imgcoord.shape))
    rowmean = imgcoord[:,0].mean()
    colmean = imgcoord[:,1].mean()
    imgcoordorigin[:, 0] = imgcoord[:, 0] - rowmean
    imgcoordorigin[:, 1] = imgcoord[:, 1] - colmean
    
    imgrotcoordorigin[:, 0] = imgcoordorigin[:, 1] * np.math.sin(angleofrect) + imgcoordorigin[:, 0] * np.math.cos(angleofrect)
    imgrotcoordorigin[:, 1] = imgcoordorigin[:, 1] * np.math.cos(angleofrect) - imgcoordorigin[:, 0] * np.math.sin(angleofrect)
    imgrotcoord[:, 0] = imgrotcoordorigin[:, 0] + centroid[0]
    imgrotcoord[:, 1] = imgrotcoordorigin[:, 1] + centroid[1]
    imgrotcoord = np.int64(imgrotcoord)
    strelplus = np.ones((3,3),np.uint8)
    strelplus[0,0] = strelplus[2,0] = strelplus[0,2] = strelplus[2,2] = 0
    rotimg[imgrotcoord[:,0], imgrotcoord[:,1]] = 1
    rotimg = np.uint8(rotimg)
    rotimg = skmorph.closing(rotimg, np.ones((3, 3), np.uint8))
    
    return(rotimg)
