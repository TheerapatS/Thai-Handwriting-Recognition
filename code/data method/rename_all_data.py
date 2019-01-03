import os


path = "D:\Work\Project\\training_set\\68PersonsBmpChar\\"


for fo in os.listdir(path):
	path_fo = path + str(fo) + "\\"
	i = 1
	for file in os.listdir(path_fo):
			os.rename(path_fo + file,path_fo + "a" + str(i)+".bmp")
			i = i + 1 
        
for fo in os.listdir(path):
	path_fo = path + str(fo) + "\\"
	i = 1
	for file in os.listdir(path_fo):
			os.rename(path_fo + file,path_fo + str(i)+".bmp")
			i = i + 1 

for fo in os.listdir(path):
	path_fo = path + str(fo) + "\\"
	for filr in os.listdir(path_fo):
		img = cv2.cvtColor(cv2.imread(path_fo),cv2.COLOR_BGR2GRAY)


def find_bdbox(img):
    mask = cv2.inRange(img,(0),(range_color_char))
    temp = mask.copy()
    __, contours, __ = cv2.findContours(temp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
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
    return [top[0],bottom[len(bottom)-1],left[0],right[len(right)-1]]
