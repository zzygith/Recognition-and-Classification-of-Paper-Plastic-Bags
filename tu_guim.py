from tkinter import *
import cv2
import os
from PIL import Image, ImageTk
from tkinter import ttk
import tkinter.messagebox
from matplotlib import pyplot as plt
import numpy as np
import math
from imutils.perspective import four_point_transform
import imutils
import joblib
import time
from tkinter import filedialog as tkFileDialog

import z_liang_tu_bu_tong as z


class APP:
    def __init__(self):
        self.camera = None   # 摄像头
        self.phone="http://admin:admin@192.168.31.16:8081/"
        self.root = Tk()
        self.root.title('Package')
        self.root.geometry('%dx%d' % (800, 600))
        self.createFirstPage()
        mainloop()

    def createFirstPage(self):
        self.page1 = Frame(self.root)
        self.page1.pack()
        Label(self.page1, text='欢迎使用包装分拣系统', font=('粗体', 20)).pack()
        image = Image.open("xiaohui2.jpg") 
        photo = ImageTk.PhotoImage(image = image)
        self.data1 = Label(self.page1,  width=800,height=500,image = photo)
        self.data1.image = photo
        self.data1.pack(padx=5, pady=5)


        path='F:/WOLF/deepWND/seedbag/'
        queryPath=path+'test/' #图库路径

        isExists=os.path.exists(queryPath)
 
        if not isExists:
            os.makedirs(queryPath) 
        filename=queryPath+'taiguo.jpg'
        img_muban=cv2.imread('trymuban.jpg')
        cv2.imwrite(filename, img_muban)


        self.button11 = Button(self.page1, width=18, height=2, text="开始分拣", bg='red', font=("宋", 12),
                               relief='raise',command = self.createSecondPage)
        self.button11.pack(side=LEFT, padx=25, pady = 10)
        self.button12 = Button(self.page1, width=18, height=2, text="增删模板", bg='green', font=("宋", 12),
                               relief='raise', command = self.createSecondPage)
        self.button12.pack(side=LEFT, padx=110, pady = 10)
        self.button14 = Button(self.page1, width=18, height=2, text="退出系统", bg='gray', font=("宋", 12),
                               relief='raise',command = self.quitMain)
        self.button14.pack(side=LEFT, padx=30, pady = 10)

    def createSecondPage(self):
        #self.camera = cv2.VideoCapture(self.phone)


        self.page1.pack_forget()
        self.page2 = Frame(self.root)
        self.page2.pack()
        Label(self.page2, text='欢迎使用包装分拣系统', font=('粗体', 20)).pack()
        self.data2 = Label(self.page2)
        self.data2.pack(padx=5, pady=5)


        self.button21 = Button(self.page2, width=18, height=2, text="显示结果", bg='red', font=("宋", 12),
                               relief='raise',command = self.showResult)
        self.button21.pack(side=LEFT,padx=25,pady = 10)


        self.button22 = Button(self.page2, width=18, height=2, text="录入模板", bg='green', font=("宋", 12),
                               relief='raise',command = self.asksaveasfilename)
        self.button22.pack(side=LEFT,padx=25,pady = 10)

        self.file_opt = options = {}  
        options['defaultextension'] = '.jpg'  
        options['filetypes'] = [('all files', '.*'), ('jpg files', '.jpg')]  
        options['initialdir'] =   'F:\\WOLF\\deepWND\\seedbag\\test\\'
        options['initialfile'] = 'seedbackage.jpg'  



        self.button23 = Button(self.page2, width=18, height=2, text="删除模板", bg='blue', font=("宋", 12),
                               relief='raise',command = self.askopenfilename)
        self.button23.pack(side=LEFT,padx=25,pady = 10)




        self.button24 = Button(self.page2, width=18, height=2, text="返回", bg='gray', font=("宋", 12),
                               relief='raise',command = self.backFirst)
        self.button24.pack(side=LEFT,padx=25,pady = 10)



        #创建SIFT特征提取器
        self.sift = cv2.xfeatures2d.SIFT_create() 
        #创建FLANN匹配对象
        FLANN_INDEX_KDTREE=0
        indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
        searchParams=dict(checks=50)
        self.flann=cv2.FlannBasedMatcher(indexParams,searchParams)



        self.video_loop(self.data2)


    def getMatchNum(self,matches,ratio):
    #返回特征点匹配数量和匹配掩码
        matchesMask=[[0,0] for i in range(len(matches))]
        matchNum=0
        for i,(m,n) in enumerate(matches):
            if m.distance<ratio*n.distance: #将距离比率小于ratio的匹配点删选出来
                matchesMask[i]=[1,0]
                matchNum+=1
        return (matchNum,matchesMask)



    def video_loop(self, panela):

        success=1
        img = cv2.imread('tryyangpins.jpg')


        if success:


            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
            ret, binary = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)  
            nin, contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


            allarea=[]
            boxes=[]
            cont_chosen=[]
            box_chosen=[]
            for c in contours:
                rect = cv2.minAreaRect(c)
                # 计算最小面积矩形的坐标
                box = cv2.boxPoints(rect)
                # 将坐标规范化为整数
                box = np.int0(box)
                boxes.append(box)
                area = cv2.contourArea(box)
                allarea.append(area)

            maxarea=max(allarea)
            win=0

            for area in allarea:
                if area > maxarea*0.7 and area>1000:

                    #cv2.drawContours(img, [boxes[allarea.index(area)]], 0, (0, 255 , 0), 2)
                    cont_chosen.append(contours[allarea.index(area)])
                    box_chosen.append(boxes[allarea.index(area)])

            docCnt =[]

            if len(cont_chosen) > 0:
                cont_chosen =sorted(cont_chosen,key=cv2.contourArea,reverse=True)

                for c in cont_chosen:

                    peri = cv2.arcLength(c,True)                   # 轮廓按大小降序排序
                    approx = cv2.approxPolyDP(c,0.02 * peri,True)  # 获取近似的轮廓
                    if len(approx) ==4:                            # 近似轮廓有四个顶点
                        #docCnt = approx
                        docCnt.append(approx)

            image=img
            cont_chosen=box_chosen

            i=0
            self.all_results=[]

            for d in docCnt:
                result_img = four_point_transform(image, d.reshape(4,2)) # 对原始图像进行四点透视变换

                self.all_results.append(result_img)

                #cv2.imshow("result_img%d"%i, result_img)


                i=i+1

            cv2.drawContours(image, cont_chosen, -1, (0, 0 , 255), 2)
            img=image


            cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
            cv2image=cv2.resize(cv2image, (300,450)) 
            current_image = Image.fromarray(cv2image)  # 将图像转换成Image对象
            imgtk = ImageTk.PhotoImage(image=current_image)
            panela.imgtk = imgtk
            panela.config(image=imgtk)
            self.root.after(10, lambda: self.video_loop(panela))




    def showResult(self):

        path='F:/WOLF/deepWND/seedbag/'
        queryPath=path+'test/' #图库路径
        comparisonImageList={} #记录比较结果



        for result_img in self.all_results:

            result_img=cv2.resize(result_img,(300,500),interpolation=cv2.INTER_CUBIC)


            # num = num+1
            # filename = "F:/WOLF/deepWND/seedbag/p/nframes.jpg"
            # cv2.imwrite(filename,result_img)


            sampleImage = cv2.cvtColor(result_img,cv2.COLOR_BGR2GRAY)


            kp1, des1 = self.sift.detectAndCompute(sampleImage, None) #提取样本图片的特征
            for parent,dirnames,filenames in os.walk(queryPath):
                for p in filenames:
                    p=queryPath+p
                    queryImage=cv2.imread(p,0)
                    kp2, des2 = self.sift.detectAndCompute(queryImage, None) #提取比对图片的特征
                    matches=self.flann.knnMatch(des1,des2,k=2) #匹配特征点，为了删选匹配点，指定k为2，这样对样本图的每个特征点，返回两个匹配
                    (matchNum,matchesMask)=self.getMatchNum(matches,0.9) #通过比率条件，计算出匹配程度
                    if len(matches)==0:
                        matchRatio=999999 
                    else:
                        matchRatio=int(matchNum*100/len(matches))
                    comparisonImageList[p]=matchRatio

            max_prices = max(zip(comparisonImageList.values(),comparisonImageList.keys()))

            if max_prices[1]=='F:/WOLF/deepWND/seedbag/test/guiyang.jpg':
                max_price='贵阳白棒豆'
            elif max_prices[1]=='F:/WOLF/deepWND/seedbag/test/taiguo.jpg':
                max_price='泰国无筋架豆王'
            elif max_prices[1]=='F:/WOLF/deepWND/seedbag/test/lvfeicui.jpg':
                max_price='绿翡翠四季豆'
            elif max_prices[1]=='F:/WOLF/deepWND/seedbag/test/zhonglv.jpg':
                max_price='中绿一号'
            elif max_prices[1]=='F:/WOLF/deepWND/seedbag/test/zidou.jpg':
                max_price='摘不败秋紫豆'  

            tkinter.messagebox.showinfo('结果','分类： '+max_price)
            z.kaishi(max_prices[1],result_img)
            
            # print(max_price,'相似度:', '%s%%'%max_prices[0])

            # cv2.drawContours(image, cont_chosen, -1, (0, 0 , 255), 2)
            # img=image



    def asksaveasfilename(self):  

        """Returns an opened file in write mode. 
        This time the dialog just returns a filename and the file is opened by your own code. 
        """  

        # get filename 
        for result_img in self.all_results:

            result_img=cv2.resize(result_img,(300,500),interpolation=cv2.INTER_CUBIC)


            # num = num+1
            # filename = "F:/WOLF/deepWND/seedbag/p/nframes.jpg"
            # cv2.imwrite(filename,result_img)

            filename = tkFileDialog.asksaveasfilename(**self.file_opt) 
            cv2.imwrite(filename,result_img)
            tkinter.messagebox.showinfo('结果','成功添加模板')

        # open file on your own  
        # if filename:  
        #     return open(filename, 'w')  



    def askopenfilename(self):  

        """Returns an opened file in read mode. 
        This time the dialog just returns a filename and the file is opened by your own code. 
        """  

        # get filename  
        filename = tkFileDialog.askopenfilename(**self.file_opt) 
        os.remove(filename)
        tkinter.messagebox.showinfo('结果','成功删除模板')



    def backFirst(self):
        self.page2.pack_forget()
        self.page1.pack()
        # 释放摄像头资源
        #self.camera.release()
        cv2.destroyAllWindows()

    def backMain(self):
        self.root.geometry('900x600')
        self.page3.pack_forget()
        self.page1.pack()

    def quitMain(self):
        sys.exit(0)


if __name__ == '__main__':

    demo = APP()
