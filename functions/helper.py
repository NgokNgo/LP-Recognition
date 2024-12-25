import math
import cv2

def changeContrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # convert from BGR to LAB 
    l_channel, a, b = cv2.split(lab) # tach cac kenh mau
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) # tao doi tuong CLAHE
    cl = clahe.apply(l_channel) # ap dung CLAHE vao kenh L
    limg = cv2.merge((cl,a,b)) # merge lai cac kenh mau
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR) # convert from LAB to BGR
    return enhanced_img

def linear_equation(x1, y1, x2, y2):
    """
    Tính hệ số góc và hệ số tự do của phương trình đường thẳng đi qua 2 điểm (x1, y1) và (x2, y2)
    """
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b

def check_point_linear(x, y, x1, y1, x2, y2):
    """
    Kiểm tra điểm (x, y) có nằm trên đường thẳng đi qua 2 điểm (x1, y1) và (x2, y2) hay không
    """
    a, b = linear_equation(x1, y1, x2, y2)
    y_pred = a*x+b
    return(math.isclose(y_pred, y, abs_tol = 3))

def read_plate(yolo_license_plate, im):
    im = changeContrast(im)
    LP_type = "1"
    results = yolo_license_plate(im)
    bb_list = results[0].boxes.xyxy.cpu()
    characters = results[0].boxes.cls.cpu()
    if len(bb_list) == 0 or len(bb_list) < 7 or len(bb_list) > 10:
        return "unknown" #len(bb_list)
    
    # get class names
    character_list = []
    class_names = yolo_license_plate.names
    for char in characters:
        class_name = class_names[int(char)]
        character_list.append(class_name)

    center_list = []
    y_mean = 0
    y_sum = 0
    # find center of each bounding box
    for index, bb in enumerate(bb_list):
        x_c = (bb[0]+bb[2])/2
        y_c = (bb[1]+bb[3])/2
        y_sum += y_c
        center_list.append([x_c, y_c, character_list[index]]) 
    # find 2 point to draw line
    l_point = center_list[0]
    r_point = center_list[0]
    for cp in center_list:
        # tìm điểm trái nhất và phải nhất
        if cp[0] < l_point[0]:
            l_point = cp
        if cp[0] > r_point[0]:
            r_point = cp
    for ct in center_list:
        if l_point[0] != r_point[0]:
            # nếu có 1 điểm không nằm trên đường thẳng thì là biển số 2 dòng
            if (check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]) == False):
                LP_type = "2"

    y_mean = int(int(y_sum) / len(bb_list)) # đường giữa 2 dòng
    # size = results.pandas().s

    # 1 line plates and 2 line plates
    line_1 = []
    line_2 = []
    license_plate = ""
    if LP_type == "2":
        for c in center_list:
            if int(c[1]) > y_mean:
                line_2.append(c)
            else:
                line_1.append(c)
        # sort the list of center points by x coordinate
        for l1 in sorted(line_1, key = lambda x: x[0]):
            license_plate += str(l1[2])
        license_plate += "-"
        for l2 in sorted(line_2, key = lambda x: x[0]):
            license_plate += str(l2[2])
    else:
        for l in sorted(center_list, key = lambda x: x[0]):
            license_plate += str(l[2])
    return license_plate