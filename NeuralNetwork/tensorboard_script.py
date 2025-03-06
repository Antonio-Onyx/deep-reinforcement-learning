import math
from torch.utils.tensorboard.writer import SummaryWriter

#in terminal use the comand: tensorboard --logdir=runs
#the go to your browser an put  http://localhost:6006/ 

if __name__ == '__main__':
    writer = SummaryWriter()
    funcs = {"sin":math.sin, "cos":math.cos, "tan":math.tan}

    for angle in range(-360, 360):
        angle_rad = angle * math.pi / 180
        for name, fun in funcs.items():
            val = fun(angle_rad)
            writer.add_scalar(name, val, angle)

    writer.close()