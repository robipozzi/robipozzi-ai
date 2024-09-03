import os
import numpy as np

def manageFile():
    fileDir = "/Users/robertopozzi/temp/python"
    os.makedirs(fileDir, exist_ok=True)
    filePath = fileDir + "/demofile.txt"
    try:
        f = open(filePath, "x")
    except:
        print("File exists, read content ...")
        f = open(filePath, "r")
        fileContent = f.read()
        print(fileContent)

def createArray():
    print("##### createArray() #####")
    a = np.array(42)
    b = np.array([1, 2, 3, 4, 5])
    c = np.array([[1, 2, 3], [4, 5, 6]])
    d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    print(f"Array:\n {a} \nDimension: {a.ndim} \nType: {type(a)}")
    print("----------------------")
    print(f"Array:\n {b} \nDimension: {b.ndim} \nType: {type(b)}")
    print("----------------------")
    print(f"Array:\n {c} \nDimension: {c.ndim} \nType: {type(c)}")
    print("----------------------")
    print(f"Array:\n {d} \nDimension: {d.ndim} \nType: {type(d)}")
    print("----------------------")
    print()

def accessArray():
    print("##### accessArray() #####")
    a = np.array([[1,2,3,4,5], [6,7,8,9,10]])
    b = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

    print(f"Array:\n {a} \nDimension: {a.ndim} \nType: {type(a)}")
    print("----------------------")
    print('Access 2nd element on 1st row (as a[0,1]): ', a[0, 1])
    print(f"Shape = {a.shape}")
    print("----------------------")
    
    print(f"Array:\n {b} \nDimension: {b.ndim} \nType: {type(b)}")
    print("----------------------")
    print('Access the third element of the second array of the first array (as b[0,1,2]): ', b[0, 1, 2])
    print(f"Shape = {b.shape}")
    print("----------------------")  
    print()

def sliceArray():
    print("##### sliceArray() #####")
    a = np.array([1, 2, 3, 4, 5, 6, 7])
    print(f"Array:\n {a} \nDimension: {a.ndim} \nType: {type(a)}")
    print("----------------------")
    print('Access 2nd element (as a[1]): ', a[1])
    print(f"Shape = {a.shape}")
    print("Slicing ...")
    startIndex = 1
    endIndex = 5
    print(f"Slice with startIndex = {startIndex}, endIndex = {endIndex} --> {a[startIndex:endIndex]}")
    startIndex = 4
    print(f"Slice with startIndex = {startIndex} to the end --> {a[startIndex:]}")
    endIndex = 4
    print(f"Slice from start to endIndex = {endIndex} --> {a[:endIndex]}")
    print("----------------------")
    print()

if __name__ == "__main__":
    print("Here I am, let's start\n")
    createArray()
    accessArray()
    sliceArray()
