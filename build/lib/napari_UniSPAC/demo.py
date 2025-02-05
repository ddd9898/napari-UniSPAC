import sys
from qtpy.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QTextEdit, QCheckBox,QFrame, QSlider,QVBoxLayout
from qtpy.QtGui import QPixmap, QPainter, QPen, QColor, QWheelEvent
from qtpy.QtCore import Qt,QPoint

import zarr
import math
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from skimage.measure import label
from models.segNeuro import segEM2d, segEM3d
from PIL import Image
from scipy.stats import multivariate_normal
# from waterz import get_segmentation

# from utils.dataloader import Dataset_2D_Zebrafinch_Train,collate_fn_2D_Zebrafinch_Train

class CustomSlider(QSlider):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        # Emit a custom signal or call a function
        self.parent().rawSlider_released(event)



class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        #UI setting 
        self.mode = 0  # 初始模式为0
        self.image_idx = 0
        self.max_image_idx = 32
        self.points = []  # 存储点击的点坐标和颜色
        
        ##Model setting
        # self.device = 'cpu'
        self.device = 'cuda'
        self.segNeuro2d_model = 'segEM2d(hemi+fib25)_Best_in_val.model'
        self.segNeuro3d_model = 'segEM3d_trace(hemi+fib25)_Best_in_val.model'
        
        self.crop_size = 512
        self.slices_tracking = 32

        self.initUI()
        self.initModels()

    def initUI(self):
        self.setWindowTitle('Interactive Neuron Segmentation Tool')
        self.setGeometry(100, 100, 1400, 760)

        ##raw interface
        self.raw_slice_show = QLabel(self)
        self.raw_slice_show.setGeometry(10, 10, 600, 600)
        self.raw_slice_show.mousePressEvent = self.mousePressEvent
        # self.raw_slice_show.mouseMoveEvent = self.mouseMoveEvent 
        self.raw_slice_show.setMouseTracking(True)
        self.setMouseTracking(True)


        ##raw slider
        self.raw_slice_slider = CustomSlider(Qt.Horizontal,self)
        self.raw_slice_slider.setGeometry(500, 620, 100, 20)
        self.raw_slice_slider.setMinimum(0)
        self.raw_slice_slider.setMaximum(200)
        # self.raw_slice_slider.setSingleStep(1)
        self.raw_slice_slider.setValue(0)
        # self.raw_slice_slider.setTickPosition(QSlider.TicksBelow)
        self.raw_slice_slider.valueChanged.connect(self.rawSlider_valueChange)

        # Label to display the flag value
        self.raw_slice_label = QLabel('Slice: 0', self)
        self.raw_slice_label.setGeometry(420, 612, 60, 30)  # 设置标签的位置和大小


        self.loadImage()

        ##interfence button
        self.inference_btn = QPushButton('Inference', self)
        self.inference_btn.setGeometry(10, 615, 100, 30)
        self.inference_btn.clicked.connect(self.inference)

        ##track button
        self.track_btn = QPushButton('Trace', self)
        self.track_btn.setGeometry(230, 615, 100, 30)
        self.track_btn.clicked.connect(self.track)

        ##Output image interface
        self.seg_show = QLabel(self)
        self.seg_show.setGeometry(650, 10, 600, 600)
        self.seg_show.setFrameShape(QFrame.Box)
        # self.seg_show.mousePressEvent = self.mousePressEvent
        self.segPixmap = QPixmap(600,600)
        self.segPixmap.fill(Qt.white)
        self.affinityPixmap = QPixmap(600,600)
        self.affinityPixmap.fill(Qt.white)
        self.seg_show.setPixmap(self.segPixmap.scaled(self.seg_show.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))



        ##Checkbox
        self.mode_checkbox = QCheckBox('ACRLSD',self)
        self.mode_checkbox.setGeometry(120, 615, 100, 30)
        self.mode_checkbox.stateChanged.connect(self.changeMode)


        ##Output text interface
        self.output_text = QTextEdit(self)
        self.output_text.setGeometry(10, 650, 600, 100)


    
    def initModels(self):
        # set device
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = self.device
        print(device)
        self.output_text.append("device:{}".format(device))


        ##Load 2d model
        model_path = './checkpoints/' + self.segNeuro2d_model 
        # self.segNeuro2d_model = segneuro2d()
        self.segNeuro2d_model = segEM2d()
        weights = torch.load(model_path,map_location=torch.device('cpu'))
        self.segNeuro2d_model.load_state_dict(weights)
        self.segNeuro2d_model = self.segNeuro2d_model.to(device)
        self.output_text.append("Pretrained 2d model loaded!")
        print("Pretrained 2d model loaded!")



        ##Load 3d model
        model_path = './checkpoints/' + self.segNeuro3d_model 
        self.segNeuro3d_model = segEM3d()
        weights = torch.load(model_path,map_location=torch.device('cpu'))
        model_dict = self.segNeuro3d_model.state_dict()
        save_dict = {k:v for k,v in weights.items() if k in model_dict.keys()}
        model_dict.update(save_dict)
        self.segNeuro3d_model.load_state_dict(model_dict) #weights
        self.segNeuro3d_model = self.segNeuro3d_model.to(device)
        self.output_text.append("Pretrained 3d model loaded!")
        print("Pretrained 3d model loaded!")
        


    def loadImage(self):
        self.raw = zarr.open('./data/hemibrain_test_roi_1/raw', mode='r')
        
        ###
        # print("Loading all data to memory...")
        # self.raw = np.array(self.raw)
        print("Loading demo data to memory...")
        self.raw = np.array(self.raw[:101])
        self.raw_slice_slider.setMaximum(self.raw.shape[0]-1)

        idx = self.image_idx
        test_patch = np.array(self.raw[idx,200:-200,200:-200])
        pad_size_x = 8-(test_patch.shape[0]%8)
        pad_size_y = 8-(test_patch.shape[1]%8)
        test_patch = np.pad(test_patch,((0,pad_size_x),(0,pad_size_y)),'constant',constant_values = (0,0)) 

        self.pixmap = Image.fromarray(test_patch[:self.crop_size,:self.crop_size]).toqpixmap()
        print("Original shape = {},crop to ({},{})".format(test_patch.shape,self.crop_size,self.crop_size))
        self.raw_slice_show.setPixmap(self.pixmap.scaled(self.raw_slice_show.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def mousePressEvent(self, event):
        x = event.x()
        y = event.y()
        if x>10 and x<610 and y>10 and y<610:
            if event.button() == Qt.LeftButton:
                self.points.append((x, y, QColor("green"),1))
            elif event.button() == Qt.RightButton:
                self.points.append((x, y, QColor("red"),0))
        self.update()

    def mouseMoveEvent(self, event):
        x = event.x()
        y = event.y()
        if x>10 and x<610 and y>10 and y<610:
            if self.mode == 0:
                self.seg_show.setPixmap(self.segPixmap.scaled(self.seg_show.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.seg_show.setPixmap(self.affinityPixmap.scaled(self.seg_show.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            painter = QPainter(self.seg_show.pixmap())
            pen = QPen()
            pen.setWidth(8)
            pen.setColor( QColor("blue"))
            painter.setPen(pen)
            # print("({},{})".format(x,y))
            painter.drawPoint(QPoint(x, y))
            painter.end()
        self.update()


        
    def keyPressEvent(self, event):
        if event.key()== Qt.Key_Q and len(self.points)>0:
            self.points.pop()
        elif event.key()== Qt.Key_E:
            self.points = list()
        # print(len(self.points))
        self.raw_slice_show.setPixmap(self.pixmap.scaled(self.raw_slice_show.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.paintEvent(event)

        

    def changeMode(self, state):
        if state == Qt.Checked:
            self.mode = 1
            self.seg_show.setPixmap(self.affinityPixmap.scaled(self.seg_show.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.mode = 0
            self.seg_show.setPixmap(self.segPixmap.scaled(self.seg_show.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def paintEvent(self, event):
        painter = QPainter(self.raw_slice_show.pixmap())
        pen = QPen()
        pen.setWidth(8)

        for point in self.points:
            pen.setColor(point[2])
            painter.setPen(pen)
            # painter.setBrush(point[2])
            
            painter.drawPoint(QPoint(point[0], point[1]))

    def rawSlider_valueChange(self,value):
        self.image_idx = value
        self.raw_slice_label.setText(f'Slice: {value}')

    def rawSlider_released(self,event):
        # print("Relseased")
        idx = self.image_idx
        test_patch = np.array(self.raw[idx,200:-200,200:-200])
        # test_patch = np.array(self.raw[200+idx,200:712,200:712])
        pad_size_x = 8-(test_patch.shape[0]%8)
        pad_size_y = 8-(test_patch.shape[1]%8)
        test_patch = np.pad(test_patch,((0,pad_size_x),(0,pad_size_y)),'constant',constant_values = (0,0)) 

        self.pixmap = Image.fromarray(test_patch[:self.crop_size,:self.crop_size]).toqpixmap()
        print("Original shape = {},crop to ({},{})".format(test_patch.shape,self.crop_size,self.crop_size))
        self.raw_slice_show.setPixmap(self.pixmap.scaled(self.raw_slice_show.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))




    ##Inference-related functions
    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data)).astype(np.float32)
    def generate_gaussian_matrix(self, Points_pos,Points_lab, H, W, theta=10):
        total_matrix = np.zeros((H,W))

        record_list = list()
        for n,(X,Y) in enumerate(Points_pos):
            if (X,Y) not in record_list:
                record_list.append((X,Y))
            else:
                continue

            # 生成坐标网格
            x, y = np.meshgrid(np.arange(W), np.arange(H))
            pos = np.dstack((x, y))

            # 以 (X, Y) 为中心，theta 为标准差生成高斯分布
            rv = multivariate_normal(mean=[X, Y], cov=[[theta, 0], [0, theta]])

            # 计算高斯分布在每个像素上的值
            matrix = rv.pdf(pos)
            ##normalize
            matrix = matrix * (1/np.max(matrix))

            total_matrix = total_matrix + matrix*(Points_lab[n]*2-1)
            # total_matrix = total_matrix + matrix

        if np.max(Points_lab) == 0:
            total_matrix = total_matrix*2 + 1

        return total_matrix
    
    def create_lut(self, labels):
        ##labels标记了图像中所有的连通域，维度对应图像的维度
        max_label = np.max(labels)
        ## 下面这个lut是给每个连通域随机设定rgb值
        np.random.seed(100)
        lut = np.random.randint(
                low=0,
                high=255,
                size=(int(max_label + 1), 3),
                dtype=np.uint8)
        lut[0] = 0
        ## 除了rgb外，增加一个通道，lut的维度为：连通域数量*4
        lut = np.append(
                lut,
                np.zeros(
                    (int(max_label + 1), 1),
                    dtype=np.uint8) + 255,
                axis=1)

        lut[0] = 0 ##第0个连通域为背景，所有通道值设为0
        colored_labels = lut[labels] ## 获得所有像素点的各通道值，维度为：图像维度*通道数（4）

        return colored_labels

    def inference(self):
        ##Load image
        idx = self.image_idx
        raw_slice0 = np.array(self.raw[idx,200:-200,200:-200])
        # raw_slice0 = np.array(self.raw[200+idx,200:712,200:712])
        pad_size_x = 8-(raw_slice0.shape[0]%8)
        pad_size_y = 8-(raw_slice0.shape[1]%8)
        raw_slice0 = np.pad(raw_slice0,((0,pad_size_x),(0,pad_size_y)),'constant',constant_values = (0,0)) 
        raw_slice0 = self.normalize(raw_slice0[:self.crop_size,:self.crop_size])

        # if self.mode == 1:
        #     ##Inference
        #     raw = torch.tensor(raw_slice0 ,dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(self.device)
        #     y_lsds,y_affinity = self.segNeuro2d_model.model_affinity(raw)
        #     y_affinity = y_affinity.detach().cpu().numpy()

        #     ######计算ACRLSD的性能
        #     threshold = [0.35] #higher thresholds will merge more, lower thresholds will split more
        #     ##1: Segment upper
        #     print('Start watershed process...')
        #     # watershed assumes 3d arrays, create fake channel dim (call these watershed affs - ws_affs)
        #     # affs shape: 3, h, w
        #     # waterz agglomerate requires 4d affs (c, d, h, w) - add fake z dim
        #     ws_affs = (1-y_affinity).transpose(1,0,2,3)
        #     ws_affs = np.stack([
        #             ws_affs[0],
        #             ws_affs[1]]
        #         )

        #     seg_acrlsd = get_segmentation(ws_affs, threshold)
        #     print(seg_acrlsd.shape)

        # else:
        Points_pos = list()
        Points_lab = list()
        self.output_text.append("Inference once,the prompts are as follows:")
        print("推理一次,prompt信息如下:")
        for point in self.points:
            x_img = int((point[0]-10)/600*self.crop_size)
            y_img = int((point[1]-10)/600*self.crop_size)

            # print("({},{}):{}".format(point[0],point[1],point[3]))
            self.output_text.append("({},{}):{}".format(x_img,y_img,point[3]))

            Points_pos.append([x_img,y_img])
            Points_lab.append(point[3])
        Points_pos = np.array(Points_pos)
        Points_lab = np.array(Points_lab)

        ##Inference
        if len(self.points)==0:
            point_map = torch.ones(raw_slice0.shape,dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            point_map = self.generate_gaussian_matrix(Points_pos,Points_lab,raw_slice0.shape[0],raw_slice0.shape[1],theta=30)
            point_map = torch.tensor(point_map ,dtype=torch.float32).unsqueeze(0).to(self.device)
        self.point_map = point_map

        raw = torch.tensor(raw_slice0 ,dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(self.device)
        y_mask,y_lsds,y_affinity = self.segNeuro2d_model(raw,self.point_map)
        # y_mask,y_affinity = self.segNeuro2d_model(raw,self.point_map)
        y_mask = (y_mask>0.5)+0
        segmentation = label((y_mask.squeeze().detach().cpu().numpy())[:,:])
        segmentation_color = self.create_lut(segmentation)
        # segmentation_color[:,:,0] = (raw_slice0*255).astype(np.uint8)


        ## 
        y_affinity =  np.sum(y_affinity.detach().cpu().numpy().squeeze(),axis=0)
        y_affinity = ((y_affinity-np.min(y_affinity))/(np.max(y_affinity)-np.min(y_affinity))*255).astype(np.uint8)
        print(np.min(y_affinity))
        trans = Image.fromarray(y_affinity)
        self.affinityPixmap = trans.toqpixmap()


        # pixmap = QPixmap(self.seg_show_path)
        trans = Image.fromarray(segmentation_color)
        self.segPixmap = trans.toqpixmap()
        
        
        if self.mode == 0:
            self.seg_show.setPixmap(self.segPixmap.scaled(self.seg_show.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.seg_show.setPixmap(self.affinityPixmap.scaled(self.seg_show.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def track(self):
        ##Load image
        self.raw_3d = np.zeros((self.crop_size,self.crop_size,self.slices_tracking),dtype=np.float32)
        # for idx in range(self.slices_tracking):
        if (self.image_idx + self.slices_tracking) >= self.raw.shape[0]:
            self.image_idx = 0
            self.output_text.append("Error tracing. Out of raw shape! Slice reset to 0")
        for idx in range(self.image_idx, self.image_idx + self.slices_tracking):
            raw_slice0 = np.array(self.raw[idx,200:-200,200:-200])
            pad_size_x = 8-(raw_slice0.shape[0]%8)
            pad_size_y = 8-(raw_slice0.shape[1]%8)
            raw_slice0 = np.pad(raw_slice0,((0,pad_size_x),(0,pad_size_y)),'constant',constant_values = (0,0)) 
            self.raw_3d[:,:,idx-self.image_idx] = raw_slice0[:self.crop_size,:self.crop_size]
        self.raw_3d = self.normalize(self.raw_3d)

        raw = torch.as_tensor(self.raw_3d,dtype=torch.float, device= self.device) 
        raw = raw.unsqueeze(0).unsqueeze(0)

        print("raw shape = {}".format(raw.shape))
        print("self.point_map shape = {}".format(self.point_map.shape))
        y_mask,_,_,_ =  self.segNeuro3d_model(raw,self.point_map)
        # y_mask,_,_ =  self.segNeuro3d_model(raw,self.point_map)
        y_mask = (y_mask>0.5)+0
        y_mask = y_mask.squeeze().detach().cpu().numpy().transpose(2,0,1)
        segmentation = label(y_mask)
        segmentation_color = self.create_lut(segmentation)

        import imageio
        print(segmentation_color.shape)
        # np.save('./output/y_seg3d_color.npy',y_seg3d_color)
        imageio.volwrite('./output/roi1_slices0-31.tiff',segmentation_color)
        self.output_text.append("Tracing ends!")






if __name__ == '__main__':
    app = QApplication(sys.argv)

    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())

    