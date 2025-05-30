import numpy as np
import torch





def Get_Pixel_Pos_Dict_EBStyle():

    '''
    Returns the Pixel_Id to Pixel_Position Dictionary for the
    EventBrowser Style :
    The bottom left pixel is the first pixel in the array.
    Increases up first, then right.
    The array has 22 rows and 20 columns for each mirror
    '''

    Pixel_Pos_Dict = {}
    
    for i in range(22):
        for j in range(20):
            Pixel_Pos_Dict[22*j+i+1] = [21-i,j]

    return Pixel_Pos_Dict
    

def Get_Pixel_Pos_Dict():

    '''
    Returns the Pixel_Id to Pixel_Position Dictionary for the
    Convention Stype
    The top left pixel is the first pixel in the array.
    Increases down first, then right.
    The array has 22 rows and 20 columns for each mirror
    '''

    Pixel_Pos_Dict = {}
    
    for i in range(22):
        for j in range(20):
            Pixel_Pos_Dict[22*j+i+1] = [i,j]

    return Pixel_Pos_Dict
    
def Find_Pixel_Id(Pixel_Pos_Dict,Pixel_Pos):

    '''
    Returns the Pixel_Id for a given Pixel_Position
    '''

    for key in Pixel_Pos_Dict:
        if Pixel_Pos_Dict[key] == Pixel_Pos:
            return key

    return -1





















if __name__ == '__main__':
    PixelPosDict = Get_Pixel_Pos_Dict()

    print('Finding Pixel Id')

    print(Find_Pixel_Id(PixelPosDict,[15,1]))