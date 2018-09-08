import cv2
import os
path = 'D:\Work\Project[261491 & 261492]\Test_tensorflow\\test\\'
patha = 'D:\Work\Project[261491 & 261492]\Test_tensorflow\\a\\'
pathb = 'D:\Work\Project[261491 & 261492]\Test_tensorflow\\b\\'
count = 1
for file in os.listdir(patha):
    print (file)
    # img = cv2.imread(file)
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite(patha+'a_'+str(count)+'.bmp',img)
    os.rename(patha+file,path+str(count)+'.bmp')
    count += 1