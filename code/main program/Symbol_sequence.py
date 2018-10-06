import cv2
import numpy as np
import matplotlib.pyplot as plt
# from skimage.feature import hog
# from skimage import data, exposure
w = -1
h = -1
def main ():
    number_of_sliding_window = 50
    path = "D:\Work\Project\\training_set\Cut_img\\"
    file = "001.jpg"
    img = cv2.cvtColor(cv2.imread(path+file),cv2.COLOR_BGR2GRAY)

    # t,b,f,l = find_size_slide(img)
    top,bottom,left,right = find_size_slide(img)
    width = int(((bottom-top)/3)*1.5)
    w = width
    h = bottom - top
    # print (t,b,f,l,w)
    step = ((right-int(width/2))-(left+int(width/2)))/100
    while left + width < right:

        crop_img = img[top:bottom, left:left+width]
        # crop_img = deskew(crop_img,width,bottom-top)
        # find_hog(crop_img)
        # cv2.imshow("test",crop_img)
        cv2.waitKey()
        left += int(2*width/3)
    crop_img2 = img[top:bottom, left:left+width]
    deskewed = [list(map(deskew,crop_img))]
    hogdata = [list(map(hog,deskewed))]
    trainData = np.float32(hogdata).reshape(-1,64)
    responses = np.repeat(np.arange(10),250)[:,np.newaxis]

    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)

    svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    svm.save('svm_data.dat')
    # cv2.imshow("test2",crop_img)
    

# def find_hog(img):
#     cell_size = (8, 8)  # h x w in pixels
#     block_size = (2, 2)  # h x w in cells
#     nbins = 9  # number of orientation bins

#     # winSize is the size of the image cropped to an multiple of the cell size
#     hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
#                                     img.shape[0] // cell_size[0] * cell_size[0]),
#                             _blockSize=(block_size[1] * cell_size[1],
#                                         block_size[0] * cell_size[0]),
#                             _blockStride=(cell_size[1], cell_size[0]),
#                             _cellSize=(cell_size[1], cell_size[0]),
#                             _nbins=nbins)

#     n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
#     hog_feats = hog.compute(img)\
#                 .reshape(n_cells[1] - block_size[1] + 1,
#                             n_cells[0] - block_size[0] + 1,
#                             block_size[0], block_size[1], nbins) \
#                 .transpose((1, 0, 2, 3, 4))  # index blocks by rows first
#     # hog_feats now contains the gradient amplitudes for each direction,
#     # for each cell of its group for each group. Indexing is by rows then columns.

#     gradients = np.zeros((n_cells[0], n_cells[1], nbins))

#     # count cells (border cells appear less often across overlapping groups)
#     cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

#     for off_y in range(block_size[0]):
#         for off_x in range(block_size[1]):
#             gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
#                     off_x:n_cells[1] - block_size[1] + off_x + 1] += \
#                 hog_feats[:, :, off_y, off_x, :]
#             cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
#                     off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

#     # Average gradients
#     gradients /= cell_count
#     # Preview
#     plt.figure()
#     plt.imshow(img, cmap='gray')
#     plt.show()

#     bin = 5  # angle is 360 / nbins * direction
#     plt.pcolor(gradients[:, :, bin])
#     plt.gca().invert_yaxis()
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.colorbar()
#     plt.show()
#     cv2.waitKey()

def find_size_slide(img):
    mask = cv2.inRange(img,(0),(180))
    kernel = np.ones((5,2), np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    kernel = np.ones((12,12), np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    temp = mask.copy()
    contourmask , contours, hierarchy = cv2.findContours(temp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    top = []
    left = []
    right = []
    bottom = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        top.append(y)
        left.append(x)
        right.append(x+w)
        bottom.append(y+h)
    top.sort()
    left.sort()
    right.sort()
    bottom.sort()
    return top[0],bottom[len(bottom)-1],left[0],right[len(right)-1]

def hog(img):
    bin_n = 16
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

def deskew(img):
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*h*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(w, h),flags=affine_flags)
    return img

# def extract_sliding_window(img):  
    
main()