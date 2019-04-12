def calc_radius(x, y):
    global ix, iy
    diff1 = ix - x
    diff2 = iy - y

    r = sqrt(diff1*diff1 + diff2*diff2)
    return int(r)

def draw(event, x, y, flags, param):
    global drawing, ix, iy, shape, canvas, brush

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if shape == 1:
                cv2.circle(canvas, (x, y), pencil, color, -1)
            elif shape == 2:
                cv2.circle(canvas, (x, y), brush, color, -1)
            elif shape == 3:
                cv2.circle(canvas, (x, y), eraser, (255, 255, 255), -1)
            elif shape == 5:
                cv2.rectangle(canvas, (ix, iy), (x, y), color, -1)
            elif shape == 6:
                cv2.circle(canvas, (x, y), calc_radius(x, y), color, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if shape == 1:
            cv2.circle(canvas, (x, y), pencil, color, -1)
        elif shape == 2:
            cv2.circle(canvas, (x, y), brush, color, -1)
        elif shape == 3:
            cv2.circle(canvas, (x, y), eraser, (255, 255, 255), -1)
        elif shape == 4:
            cv2.line(canvas, (ix, iy), (x, y), color, pencil)
        elif shape == 5:
            cv2.rectangle(canvas, (ix, iy), (x, y), color, -1)
        elif shape == 6:
            cv2.circle(canvas, (x, y), calc_radius(x, y), color, -1)

def display_shape():
    global shape
    if shape == 0:
        cv2.putText(obj, 'Off', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
    elif shape == 1:
        cv2.putText(obj, 'Pencil', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
    elif shape == 2:
        cv2.putText(obj, 'Brush', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
    elif shape == 3:
        cv2.putText(obj, 'Eraser', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
    elif shape == 4:
        cv2.putText(obj, 'Line', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
    elif shape == 5:
        cv2.putText(obj, 'Rectangle', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
    elif shape == 6:
        cv2.putText(obj, 'Circle', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)


def do_nothing(param):
    pass