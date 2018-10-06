import cv2

def main ():
    path = "D:\Work\Project\\training_set\Cut_img\\"
    file = "001.jpg"
    img = cv2.imread(path+file,cv2.IMREAD_GRAYSCALE)
    find_size_slide(img)

def find_size_slide(img):
    top = -1
    bt = -1
    p = len(img)-1
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] < 100:
                top = i
            if img[p][j] < 100:
                bt = p
        p -= 1
        if bt != -1 and top != -1:
            break
    print (top,bt)

main()