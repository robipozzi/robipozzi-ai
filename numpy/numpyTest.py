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

if __name__ == "__main__":
    print("Here I am, let's start\n")
    createArray()
