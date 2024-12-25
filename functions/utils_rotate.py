import cv2
import numpy as np
import math

def changeContrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # convert from BGR to LAB 
    l_channel, a, b = cv2.split(lab) # tach cac kenh mau
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) # tao doi tuong CLAHE
    cl = clahe.apply(l_channel) # ap dung CLAHE vao kenh L
    limg = cv2.merge((cl,a,b)) # merge lai cac kenh mau
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR) # convert from LAB to BGR
    return enhanced_img

def compute_skew(src_img, center_thres):
    # kiểm tra số kênh màu của ảnh
    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('upsupported image type')

    # xử lí ảnh để tìm đường thẳng
    img = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(img,  threshold1 = 30,  threshold2 = 100, apertureSize = 3, L2gradient = True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 3.0, maxLineGap=h/3.0)
    if lines is None:
        return 1

    # tìm đường thẳng có tọa độ y nhỏ nhất
    min_line = 100
    min_line_pos = 0 # id của đường thẳng có tọa độ y nhỏ nhất
    for i in range (len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            center_point = [((x1+x2)/2), ((y1+y2)/2)] # tìm tọa độ trung điểm của đoạn thẳng
            
            # nếu center_thres = 1 thì chỉ xét các đoạn thẳng nằm ở giữa ảnh
            # nghĩa là loại bỏ các đoạn thẳng biên của ảnh
            if center_thres == 1: 
                if center_point[1] < 7:
                    continue

            # nếu y nhỏ hơn min_line thì cập nhật min_line và min_line_pos
            if center_point[1] < min_line:
                min_line = center_point[1]
                min_line_pos = i

    angle = 0.0
    cnt = 0
    for x1, y1, x2, y2 in lines[min_line_pos]:
        ang = np.arctan2(y2 - y1, x2 - x1) # tính góc nghiêng 
        if math.fabs(ang) <= 45: # excluding extreme rotations
            angle += ang
            cnt += 1
    if cnt == 0:
        return 0.0
    return (angle / cnt)*180/math.pi # trả về góc nghiêng trung bình tính bằng radian

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2) # tìm tâm ảnh
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0) # tạo ma trận xoay ảnh
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR) # xoay ảnh
    return result


def deskew(src_img, change_cons, center_thres):
    # kiem tra shape cua anh dau vao w,h co nho hon 120 khong
    if src_img.shape[0] < 120 or src_img.shape[1] < 120:
        extraa = max(120 - src_img.shape[0], 120 - src_img.shape[1])
        # resize anh de co chieu dai va chieu rong lon hon 120
        src_img = cv2.resize(src_img, (src_img.shape[1] + extraa, src_img.shape[0] + extraa))


    if change_cons == 1:
        return rotate_image(src_img, compute_skew(changeContrast(src_img), center_thres))
    else:
        return rotate_image(src_img, compute_skew(src_img, center_thres))