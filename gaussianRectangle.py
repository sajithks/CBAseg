# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:06:09 2013

@author: root
"""
import numpy as np



def findSum(value):
    temp = np.copy(value - 1)
    sumval = 0
    for ii in range(value):
        sumval = sumval + temp
        temp = temp - 1
    return(sumval)
    
def compare(startpt, endpt, data):
    import cv2
    from matplotlib import pyplot as plt
    import skimage.morphology as skmorph
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
            print(img[event.ydata,event.xdata])     
            #    plt.close('all')
        plt.figure()    
        plt.ion() 
        plt.imshow(img,cmap='gray'),plt.show()
        fig = plt.gcf()
        #    on mouse click the value at the image location is displayed in output screen
        _ = fig.canvas.mpl_connect('button_press_event', onClick)
    
    def myshow2(img):
        
        def onClick(event):
            print(img[event.ydata,event.xdata])     
            #    plt.close('all')
        plt.figure()    
        plt.ion() 
        plt.imshow(img),plt.show()
        fig = plt.gcf()
        #    on mouse click the value at the image location is displayed in output screen
        _ = fig.canvas.mpl_connect('button_press_event', onClick)
    
    
    img = np.zeros((200,200))
    imgout = np.zeros((200,200))
    rotimg = np.zeros((200,200), np.uint8)
    lengthofrect = np.int32(np.sqrt((startpt[0] - endpt[0])**2 + (startpt[1] - endpt[1])**2)) + 30
    angleofrect = np.math.atan2((startpt[0] - endpt[0]), (startpt[1] - endpt[1]))
    widthofrect = 25
    img[100 - widthofrect/2 : 100 + widthofrect/2, 100 - lengthofrect/2: 100 + lengthofrect/2] = 1

    
    gau = gauss_kern(img)
    #Img_smooth = signal.convolve(Img,g,mode='same')
    imgfft = np.fft.rfft2(img)
    gfft = np.fft.rfft2(gau)
    fftimage = np.multiply(imgfft, gfft)
    img_smooth =np.real(np.fft.ifftshift( np.fft.irfft2(fftimage)))
    imgout = img_smooth>0.1
    
    #imgborder = imgout - cv2.dilate(np.uint8(imgout), None, iterations=1)
    #bordercoord = np.argwhere(imgborder)
    
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
    imgrotcoord[:, 0] = imgrotcoordorigin[:, 0] + rowmean
    imgrotcoord[:, 1] = imgrotcoordorigin[:, 1] + colmean
    imgrotcoord = np.int64(imgrotcoord)
    strelplus = np.ones((3,3),np.uint8)
    strelplus[0,0] = strelplus[2,0] = strelplus[0,2] = strelplus[2,2] = 0
    #rotimg = np.uint8(rotimg)
    rotimg[imgrotcoord[:,0], imgrotcoord[:,1]] = 1
    #rotimg = cv2.dilate(rotimg, strelplus, iterations = 1)
    #rotimg = cv2.erode(rotimg, strelplus, iterations = 1)
    rotimg = np.uint8(rotimg)
    rotimg = skmorph.closing(rotimg, np.ones((3, 3), np.uint8))
    
    #enlarge the image to gather the details of crop region
    enlargimg = skmorph.dilation(rotimg, np.ones((9, 9), np.uint8))
    enlargimgcoord = np.argwhere(enlargimg)
    encoord = np.copy(enlargimgcoord)
    #normalize the coordinate
    enlargimgcoord[:, 0] = enlargimgcoord[:, 0] - enlargimgcoord[:, 0].mean()
    enlargimgcoord[:, 1] = enlargimgcoord[:, 1] - enlargimgcoord[:, 1].mean()
    #midpoint of skeleton line
    midpt = np.array( [startpt[0] + endpt[0], startpt[1] + endpt[1]])/2
    #add midpoint of skeleton to get the image coordinates of data
    
    imgpartcoord = np.zeros(enlargimgcoord.shape)
    imgpartcoord[:, 0] = enlargimgcoord[:, 0] + midpt[0]
    imgpartcoord[:, 1] = enlargimgcoord[:, 1] + midpt[1]
    imgpartcoord = np.int32(imgpartcoord)
    # image part to compare
    outcoord = np.argwhere(rotimg)
    imgpart = np.zeros((200,200))
    imgpart[encoord[:, 0], encoord[:, 1]] = data[imgpartcoord[:, 0], imgpartcoord[:, 1]]
    numerator = np.sum(imgpart[outcoord[:,0], outcoord[:,1]]) * 2 
    denomenator = np.argwhere(imgpart).shape[0] + np.argwhere(rotimg).shape[0]
    dicescore = numerator/denomenator
    coverage = np.float64(np.argwhere(imgpart).shape[0])/np.float64(enlargimgcoord.shape[0])  
    return(dicescore, imgpart, coverage, imgpartcoord)

#urotimg, srotimg, vrotimg = np.linalg.svd(rotimg)
#uimgpart, simgpart, vimgpart = np.linalg.svd(imgpart)

#evalrotimg, princomprotimg = principalComponents(rotimg)
#evalimgpart, princompimgpart = principalComponents(imgpart)






