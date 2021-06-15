import cv2
import numpy as np
import math
from PIL import Image, ImageFilter
import joblib
import tkinter.messagebox

def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。
 
def trans_muban(path):
 
    file_path = path
    img_gray=cv2.imread(file_path,0)[10:490, 10:290]   # 灰度图读取，用于计算gamma值

    img = cv2.imread(file_path)[10:490, 10:290]    # 原图读取    

    mean = np.mean(img_gray)
    gamma_val = math.log10(0.5)/math.log10(mean/255)    # 公式计算gamma
     
    image_gamma_correct = gamma_trans(img, gamma_val)   # gamma变换


    kernel_size = (9, 9)
    sigma = 1;

    imgG = cv2.GaussianBlur(image_gamma_correct, kernel_size, sigma)
    return imgG,img_gray,img


def trans_jiance(path):
 
    file_path = path
    file_path_jiance = cv2.cvtColor(file_path,cv2.COLOR_BGR2GRAY)

    img_gray=file_path_jiance[10:490, 10:290]   # 灰度图读取，用于计算gamma值

    img = file_path[10:490, 10:290]    # 原图读取    

    mean = np.mean(img_gray)
    gamma_val = math.log10(0.5)/math.log10(mean/255)    # 公式计算gamma
     
    image_gamma_correct = gamma_trans(img, gamma_val)   # gamma变换


    kernel_size = (9, 9)
    sigma = 1;

    imgG = cv2.GaussianBlur(image_gamma_correct, kernel_size, sigma)
    return imgG,img_gray,img





def kaishi(jpg1,jpg2):
    imgg,img_muban,img_yq=trans_muban(jpg1)
    #imgg,img_muban=trans('blankg.jpg')
    gray = cv2.cvtColor(imgg,cv2.COLOR_BGR2GRAY)  
 #   cv2.imshow('imagmuban', img_yq) 5.16


    imgg1,imgy,img_yangq=trans_jiance(jpg2)
    #imgg1,imgy=trans('blankgq4.jpg')
    gray1 = cv2.cvtColor(imgg1,cv2.COLOR_BGR2GRAY)  
 #   cv2.imshow('imagjiance', img_yangq) 5.16

    err = cv2.absdiff(gray,gray1)
    ret, binary2 = cv2.threshold(err,50,255,cv2.THRESH_BINARY) 
 #   cv2.imshow('e',binary2) 5.16 



    # edged = cv2.Canny(binary2,50,155)                  #边缘检测

    # cv2.imshow("gray22", edged) 


    a, cnts, b = cv2.findContours(binary2, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    cont_chosen=[]
    quedians=[]
    for contour in cnts:
        if cv2.contourArea(contour)>450: #9


            cont_chosen.append(contour)


    #cv2.drawContours(img_yangq, cont_chosen, -1, (0, 0 , 255), 1)


    imgblank = np.zeros((490,290,3), np.uint8)    # 使用Numpy创建一张白纸
    imgblank.fill(255)
    length=[]

    count_pix=0
    pix=0
    nu=1
    clf=joblib.load('tree2.pkl')

    count_huaheng=0
    count_moji=0
    count_louyin=0
    quality_result=''

    for c in cont_chosen:


        rect = cv2.minAreaRect(c)

        box = cv2.boxPoints(rect)
        # 将坐标规范化为整数
        box = np.int0(box)
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1 - 1
        width = x2 - x1 - 1
        print(hight,width)
        crop_img= binary2[y1:y2, x1:x2]
        for i in range(hight):
            for j in range(width):
                try:
                    if crop_img[i,j]==255:#0白色
                        pix=pix+imgy[y1+i,x1+j]
                        # print(imgy[y1+i,x1+j])
                        count_pix=count_pix+1
                except IndexError:
                    pass

        pix_aver=pix//count_pix
        #print(pix_aver)


        if pix_aver<200:

            area=cv2.contourArea(c)
            length=cv2.arcLength(c, True)

            circularity=round((4*3.1415926*area)/(length**2),2)
            print(nu)
            print('圆度：%0.2f'%circularity)


            w0=rect[1][0]
            h0=rect[1][1]
            w=max(w0,h0)
            h=min(w0,h0)
            ratio=round(w/h,2)
            print('宽高比',ratio)
            AL=round(area/w,2)
            print('面积长必：',AL)


            M = cv2.moments(c)
            distance=0
            std_dev2=0

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
                
                
            # calculate x,y coordinate of center
            cv2.circle(img_yangq, (cX, cY), 2, (0, 255, 0), -1)
            cv2.putText(img_yangq, "%d"%nu, (cX, cY),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            nu=nu+1
            centroid=[cX,cY]
            p1=np.array(centroid)
            for i in range(len(c)):
                p2=np.array(c[i][0])
                p3=p2-p1
                distance=distance+math.hypot(p3[0],p3[1])
            ave_dis=distance/len(c)
            for i in range(len(c)):
                p2=np.array(c[i][0])
                p3=p2-p1
                distance=math.hypot(p3[0],p3[1])
                std_dev2=std_dev2+(ave_dis-distance)**2
            std_dev=round((std_dev2/len(c))**0.5,2)

            print('离散度：',std_dev)
            quedian=[circularity,ratio,AL,std_dev]

            test_x=[np.array(quedian).flatten()]
            pred_test_y = clf.predict(test_x)
            if pred_test_y==0:
                count_huaheng=count_huaheng+1
                print('\nwarning:存在划伤\n')
            if pred_test_y==1:
                count_moji=count_moji+1
                print('\nwarning:存在墨迹\n')

            quedians.append(quedian)
        else:
            M = cv2.moments(c)
            distance=0
            std_dev2=0

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
                
                
            # calculate x,y coordinate of center
            cv2.circle(img_yangq, (cX, cY), 2, (0, 255, 0), -1)
            cv2.putText(img_yangq, "%d"%nu, (cX, cY),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            count_louyin=count_louyin+1
            print(nu,'\nwarning:存在漏印\n')
            
            nu=nu+1

    if count_huaheng!=0:
        quality_result=quality_result+'存在%d处划痕\n'%count_huaheng

    if count_moji!=0:
        quality_result=quality_result+'存在%d处墨迹\n'%count_moji

    if count_louyin!=0:
        quality_result=quality_result+'存在%d处漏印\n'%count_louyin



    if quality_result!='':    
        cv2.imshow("contour", img_yangq) 
        tkinter.messagebox.showinfo('错误',quality_result)

