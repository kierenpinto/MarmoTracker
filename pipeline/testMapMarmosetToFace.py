from position_tools import mapMarmosetToFace
import numpy as np
def testMapMarmosetToFace():
    marmosets =[(790.0, 451.0), (164.5, 563.5), (34.5, 615.5), (67.5, 659.5)]
    faces = [(839, 357),(837, 355)]
    marmo_map = mapMarmosetToFace(faces,marmosets)
    print(marmo_map)

if __name__ == "__main__":
    testMapMarmosetToFace()