import re
from pyx import *

filename = "nodesLevel_"
depth = 2
for i in range(depth):
    with open(filename + str(i), 'r') as f:
        lines = f.readlines()
    getnumbers = re.compile(r"\d+")
    aux = getnumbers.findall(lines[0])
    region = {"x1": int(aux[0]),
              "x2": int(aux[1]),
              "y1": int(aux[2]),
              "y2": int(aux[3]),
              "level": int(aux[4])
              }
    aux = getnumbers.findall(lines[1])
    splitting = {"splitx": int(aux[0]),
                 "splity": int(aux[1]),
                 "logx": int(aux[2]),
                 "logy": int(aux[3])
                 }
    states = []
    for j in range(3, len(lines)):
        aux = getnumbers.findall(lines[j])
        states.append(
                     {"x": int(aux[0]),
                      "y": int(aux[1]),
                      "state": int(aux[2])}
                     )
    lengthx = region.get("x2") - region.get("x1")
    rectsx = splitting.get("splitx") * \
             (splitting.get("logx") ** region.get("level"))
    stepx = lengthx / rectsx
    lengthy = region.get("y2") - region.get("y1")
    rectsy = splitting.get("splity") * \
             (splitting.get("logy") ** region.get("level"))
    stepy = lengthy / rectsy
    print(str(stepx))
    print(str(stepy))
    c = canvas.canvas()
    for state in states:
        if state.get("state"):
            c.fill(path.rect(state.get("x") * stepx, 
                             state.get("y") * stepy, 
                             stepx,
                             stepy))
    c.writePDFfile(filename + str(i)) 
