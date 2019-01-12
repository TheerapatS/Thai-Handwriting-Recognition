import os


path = "D:\Work\Project\\training_set\\68PersonsBmpChar\\"


for fo in os.listdir(path):
	path_fo = path + str(fo) + "\\"
	i = 1
	for file in os.listdir(path_fo):
			os.rename(path_fo + file,path_fo + "b" + str(i)+".bmp")
			i = i + 1 
        
for fo in os.listdir(path):
	path_fo = path + str(fo) + "\\"
	i = 1
	for file in os.listdir(path_fo):
			os.rename(path_fo + file,path_fo + str(i)+".bmp")
			i = i + 1 


