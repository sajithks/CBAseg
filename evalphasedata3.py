#%%

import time

import createbacteria  as cb
import numpy as np
from matplotlib import pyplot as plt
import cv2
import scipy.io as sio



def fthreshValue(fmeasure,binval):
    a = np.linspace(0, 1, binval)
    fthresh = np.zeros((a.shape[0]))
    for ii in range(a.shape[0]):
        fthresh[ii] = (fmeasure > a[ii] ).tolist().count(True)
    return(fthresh/fmeasure.shape[0])

LINEWIDTH = 1
FONTSIZE = 14
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : FONTSIZE}

plt.rc('font', **font)






#%% Elf phase dataset
startime = time.time()

gt = cv2.imread('gt_phaselabel.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)

gt = cb.removeBoundaryLabel(gt)
gt = cb.labelDilation(gt,3)



cbaout = cv2.imread('cba_phase_outimg.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
cbaout = cb.removeBoundaryLabel(cbaout)

cbafilt = cv2.imread('cba_phase_filterout.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)

cbafilt = cb.removeBoundaryLabel(cbafilt)

cbasmooth = cv2.imread('cba_phase_smoothout.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)

cbasmooth = cb.removeBoundaryLabel(cbasmooth)


mamle = cv2.imread('mamle_phase.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)
#mamle = cv2.imread('/Users/sajithks/Documents/project_cell_tracking/processed/result_comparisons/segmentation/others/mamle/mamleelfphase.tif',cv2.CV_LOAD_IMAGE_UNCHANGED)

mamle = cb.removeBoundaryLabel(mamle)


plotloc = ''

matfile = sio.loadmat('phase3.mat')

metrics = matfile['metrics']

recall1, precision1, fmeasure1, dicescore1 = cb.evaluateSegmentation2(gt, cbaout)
recall2, precision2, fmeasure2, dicescore2 = cb.evaluateSegmentation2(gt, cbafilt)
recall3, precision3, fmeasure3, dicescore3 = cb.evaluateSegmentation2(gt, cbasmooth)

recall4 = metrics[:,0]
precision4 = metrics[:,1]
fmeasure4 = metrics[:,2]
dicescore4 = metrics[:,3]

recall5, precision5, fmeasure5, dicescore5 = cb.evaluateSegmentation2(gt, mamle)

bincount = 100
ft1 = fthreshValue(fmeasure1, bincount)
ft2 = fthreshValue(fmeasure2, bincount)
ft3 = fthreshValue(fmeasure3, bincount)
ft4 = fthreshValue(fmeasure4, bincount)
ft5 = fthreshValue(fmeasure5, bincount)


xval = np.linspace(0, 1, bincount)


plt.figure()#, plt.plot(xval, ft1, label='outimg','')#, xval, ft2, 'b', xval, ft3, 'r', xval, ft4, 'g'),plt.title('F-score Elf phase'),plt.legend(('line1','line2','line3','line4'),('outimg','filterout','smoothout','microbetracker')),plt.show()
line1, = plt.plot(xval, ft1, label='algorithm',color = 'k')
line2, = plt.plot(xval, ft2, label='algorithm + filtering',color = 'r', linestyle = '--')
line3, = plt.plot(xval, ft3, label='algorithm + filtering + smoothing',color = 'g', linestyle = ':' ,marker = '')
line4, = plt.plot(xval, ft4, label='Microbetracker',color = 'b', linestyle = '-')
line5, = plt.plot(xval, ft5, label='MAMLE',color = 'y', linestyle = '-')

plt.xlabel('F-score ', fontsize=30),plt.ylabel('cumulative % of cells', fontsize=30)

#plt.xlabel('F-score phase data 2', fontsize=FONTSIZE),plt.ylabel('% of total cells', fontsize=FONTSIZE)
#plt.legend(handler_map={line1: HandlerLine2D(numpoints=16)},loc='lower left')
#plt.show()
frame=plt.gca()
plt.yticks(np.arange(0, 1.25, .25),np.arange(0,125,25))
plt.xticks(np.arange(0,1.25,.5))
plt.tick_params(axis='both', which='major', labelsize=30)

#frame.axes.get_xaxis().set_visible(False)
#frame.axes.get_yaxis().set_visible(False)
plt.savefig(plotloc+'phasdat3fscorecumul.png',bbox_inches='tight')

# recall
binvals = 100
xval = np.linspace(0, 1, binvals)
#
#r1, b1 = np.histogram(recall1, xval)
#r2, b1 = np.histogram(recall2, xval)
#r3, b1 = np.histogram(recall3, xval)
#r4, b1 = np.histogram(recall4, xval)
#r5, b1 = np.histogram(recall5, xval)
#
#
#r1 = np.insert(r1,0,0)
#r2 = np.insert(r2,0,0)
#r3 = np.insert(r3,0,0)
#r4 = np.insert(r4,0,0)
#r5 = np.insert(r5,0,0)

#plt.figure(), plt.plot( xval[:-1],r1,'k', xval[:-1],r2,'b', xval[:-1],r3,'r', xval[:-1],r4,'g'),plt.title('Recall Elf phase'),plt.show()
#plt.figure()#, plt.plot(xval, ft1, label='outimg','')#, xval, ft2, 'b', xval, ft3, 'r', xval, ft4, 'g'),plt.title('F-score Elf phase'),plt.legend(('line1','line2','line3','line4'),('outimg','filterout','smoothout','microbetracker')),plt.show()
#line1, = plt.plot(xval, r1, label='outimg',color = 'k')
#line2, = plt.plot(xval, r2, label='filterout',color = 'r', linestyle = '--')
#line3, = plt.plot(xval, r3, label='smoothout',color = 'g', linestyle = ':' ,marker = '')
#line4, = plt.plot(xval, r4, label='microbetracker',color = 'b', linestyle = '-')
#line5, = plt.plot(xval, r5, label='MAMLE',color = 'y', linestyle = '-')
#
#plt.xlabel('Recall phase data 2', fontsize=FONTSIZE),plt.ylabel('Number of cells', fontsize=FONTSIZE)
##plt.legend(handler_map={line1: HandlerLine2D(numpoints=16)})
##plt.show()
#frame=plt.gca()
#frame.axes.get_xaxis().set_visible(False)
#frame.axes.get_yaxis().set_visible(False)
#plt.savefig(plotloc+'phasdat3recall.png',bbox_inches='tight')


#precision

#p1, b1 = np.histogram(precision1, xval)
#p2, b1 = np.histogram(precision2, xval)
#p3, b1 = np.histogram(precision3, xval)
#p4, b1 = np.histogram(precision4, xval)
#p5, b1 = np.histogram(precision5, xval)
#
#p1 = np.insert(p1,0,0)
#p2 = np.insert(p2,0,0)
#p3 = np.insert(p3,0,0)
#p4 = np.insert(p4,0,0)
#p5 = np.insert(p5,0,0)

#plt.figure()#, plt.plot(xval, ft1, label='outimg','')#, xval, ft2, 'b', xval, ft3, 'r', xval, ft4, 'g'),plt.title('F-score Elf phase'),plt.legend(('line1','line2','line3','line4'),('outimg','filterout','smoothout','microbetracker')),plt.show()
#line1, = plt.plot(xval, p1, label='outimg',color = 'k')
#line2, = plt.plot(xval, p2, label='filterout',color = 'r', linestyle = '--')
#line3, = plt.plot(xval, p3, label='smoothout',color = 'g', linestyle = ':' ,marker = '')
#line4, = plt.plot(xval, p4, label='microbetracker',color = 'b', linestyle = '-')
#line5, = plt.plot(xval, p5, label='MAMLE',color = 'y', linestyle = '-')
#
#plt.xlabel('Precision phase data 2', fontsize=FONTSIZE),plt.ylabel('Number of cells', fontsize=FONTSIZE)
##plt.legend(handler_map={line1: HandlerLine2D(numpoints=16)})
##plt.show()
#frame=plt.gca()
#frame.axes.get_xaxis().set_visible(False)
#frame.axes.get_yaxis().set_visible(False)
#plt.savefig(plotloc+'phasdat3prec.png',bbox_inches='tight')



#plt.figure(), plt.plot( xval[:-1],p1,'k', xval[:-1],p2,'b', xval[:-1],p3,'r', xval[:-1],p4,'g'),plt.title('Precision Elf phase'),plt.show()

# dice score
#
#d1, b1 = np.histogram(dicescore1, xval)
#d2, b1 = np.histogram(dicescore2, xval)
#d3, b1 = np.histogram(dicescore3, xval)
#d4, b1 = np.histogram(dicescore4, xval)
#d5, b1 = np.histogram(dicescore5, xval)
#
#d1 = np.insert(d1,0,0)
#d2 = np.insert(d2,0,0)
#d3 = np.insert(d3,0,0)
#d4 = np.insert(d4,0,0)
#d5 = np.insert(d5,0,0)

#plt.figure()#, plt.plot(xval, ft1, label='outimg','')#, xval, ft2, 'b', xval, ft3, 'r', xval, ft4, 'g'),plt.title('F-score Elf phase'),plt.legend(('line1','line2','line3','line4'),('outimg','filterout','smoothout','microbetracker')),plt.show()
#line1, = plt.plot(xval, d1, label='outimg',color = 'k')
#line2, = plt.plot(xval, d2, label='filterout',color = 'r', linestyle = '--')
#line3, = plt.plot(xval, d3, label='smoothout',color = 'g', linestyle = ':' ,marker = '')
#line4, = plt.plot(xval, d4, label='microbetracker',color = 'b', linestyle = '-')
#line5, = plt.plot(xval, d5, label='MAMLE',color = 'y', linestyle = '-')
#
#plt.xlabel('Dice score phase data 2', fontsize=FONTSIZE),plt.ylabel('Number of cells', fontsize=FONTSIZE)
##plt.legend(handler_map={line1: HandlerLine2D(numpoints=16)})
##plt.show()
#frame=plt.gca()
#frame.axes.get_xaxis().set_visible(False)
#frame.axes.get_yaxis().set_visible(False)
#plt.savefig(plotloc+'phasdat3fscore.png',bbox_inches='tight')


#plt.figure(), plt.plot( xval[:-1],d1,'k', xval[:-1],d2,'b', xval[:-1],d3,'r', xval[:-1],d4,'g'),plt.title('Dice score Elf phase'),plt.show()

a1 = cb.findAreaDistribution(gt)
a2 = cb.findAreaDistribution(cbaout)
a3 = cb.findAreaDistribution(cbafilt)
a4 = cb.findAreaDistribution(cbasmooth)
#a5 = cb.findAreaDistribution(microb)

microare =  sio.loadmat( '/Users/sajithks/Documents/project_cell_tracking/processed/result_comparisons/segmentation/dataset/johan_phase/microcontour/area2_3.mat') 
a5 = microare['areaarray']
a6 = cb.findAreaDistribution(mamle)


bincount=20
ah1, bina1 = np.histogram(a1,bincount)
ah2, bina2 = np.histogram(a2,bincount)
ah3, bina3 = np.histogram(a3,bincount)
ah4, bina4 = np.histogram(a4,bincount)
ah5, bina5 = np.histogram(a5,bincount)
ah6, bina6 = np.histogram(a6,bincount)

#plt.figure(),plt.plot(bina1[1:],ah1,color='k'),plt.plot(bina2[1:],ah2,color='r'),plt.plot(bina5[1:],ah5,color='g'),
#plt.show()
#a1.mean(),a2.mean(),a5.mean()

plt.figure()#, plt.plot(xval, ft1, label='outimg','')#, xval, ft2, 'b', xval, ft3, 'r', xval, ft4, 'g'),plt.title('F-score Elf phase'),plt.legend(('line1','line2','line3','line4'),('outimg','filterout','smoothout','microbetracker')),plt.show()
line1, = plt.plot(bina1[:-1], np.float32(ah1)/ah1.sum(), label='ground truth',color = 'c',linewidth = LINEWIDTH)
line2, = plt.plot(bina2[:-1],np.float32(ah2)/ah2.sum(), label='CBA',color = 'k',linewidth = LINEWIDTH)
line3, = plt.plot(bina5[:-1],np.float32(ah5)/ah5.sum(), label='Microbetracker',color = 'b', linestyle = '-',linewidth = LINEWIDTH)
line4, = plt.plot(bina6[:-1],np.float32(ah6)/ah6.sum(), label='MAMLE',color = 'y', linestyle = '-',linewidth = LINEWIDTH)

plt.xlabel('Cell area', fontsize=26),plt.ylabel('% of total cells ', fontsize=30)

#plt.xlabel('Cell area phase data 2', fontsize=FONTSIZE),plt.ylabel('Number of cells ', fontsize=FONTSIZE)
#plt.legend(handler_map={line1: HandlerLine2D(numpoints=16)},loc='upper right')
#plt.show()
frame=plt.gca()
plt.yticks(np.arange(0, .3, .1),np.arange(0, 30, 10))
plt.xticks(np.arange(0, 3500, 1000))
plt.tick_params(axis='both', which='major', labelsize=30)

#frame.axes.get_xaxis().set_visible(False)
#frame.axes.get_yaxis().set_visible(False)
plt.savefig(plotloc+'phasdat3areadist.png',bbox_inches='tight')

