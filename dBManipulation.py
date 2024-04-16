import os

path = "C:/Study/612/Project/facedB/"
dirList = os.listdir(path)

startw = '7_Parth_'
extention = '.jpg'
counter = 1

for file in dirList:
    if file.startswith( startw ):
        newName = startw + str(counter) + extention
        print( newName )
        os.rename( file, newName )
        counter += 1
