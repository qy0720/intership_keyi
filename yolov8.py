import onnxruntime
import transform
import numpy as np
import os
import ops
import torch
import cv2
import psutil
import onnxruntime
import numpy
import json
from pycocotools.cocoeval import COCOeval  # noqa

from PyQt5.QtCore import QCoreApplication
import PyQt5.QtGui
app = QCoreApplication([])

class ModelRunKit:
    def __init__(self, onnx_model_path,new_shape=(640,640)) -> None:
        self.session = onnxruntime.InferenceSession(onnx_model_path)
        N, C, H, W = self.session.get_inputs()[0].shape

        self.orig_imgs = None # image = nd.array HWC
        self.model = None
        self.new_shape = new_shape
        self.device = 'cuda'
        self.conf = 0.001
        self.iou = 0.7
        self.agnostic_nms = False
        self.max_det = 300
        self.classes = None
          
    
    
    def pre_transform(self,shape,img):
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        ratio = r, r
        # new_unpad = 640,480
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        # dw = 0, 120
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]  # wh padding
        
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border
        
        return img
        
    def preprocess(self,image):
                # 输入的image 为array
        shape = image.shape[:2]
        self.orig_imgs = [image] # image = list list[0]=nd.array HWC
        im = self.pre_transform(shape,image)
        im = np.expand_dims(im,axis=0)
        
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)

        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)
        img = im.to(self.device)
        # img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img = img.float()
        img /= 255  # 0 - 255 to 0.0 - 1.0        # process 之后要拿到 N C H W 
        self.img = img
        return self.img

    def postprocess(self,preds):
        preds = ops.non_max_suppression(preds,
                                        self.conf,
                                        self.iou,
                                        agnostic=self.agnostic_nms,
                                        max_det=self.max_det,
                                        classes=self.classes)
        
        # results = []
        for i, pred in enumerate(preds):
            orig_img = self.orig_imgs[i] if isinstance(self.orig_imgs, list) else self.orig_imgs
            if not isinstance(self.orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(self.img.shape[2:], pred[:, :4], orig_img.shape)
            # path = '/home/xz/work/ultralytics/ultralytics'
            # img_path = path[i] if isinstance(path, list) else path
            # results.append(Results(orig_img=orig_img, path=img_path, names='a.txt', boxes=pred))
        
        return preds[0]
    
    def visualize(self,image_path,save_path,preds):
        images = os.path.join(image_path)
        image = cv2.imread(images)
  
        pred = preds
        
        pred_result = []
        
        for i in range(len(pred)):
            d_pred = {}
            bbox = []
            bbox.append(float(pred[i][0]))
            bbox.append(float(pred[i][1]))
            bbox.append(float(pred[i][2])-float(pred[i][0]))
            bbox.append(float(pred[i][3])-float(pred[i][1]))
            
            # 寻找image_id
            base_name = image_path.split('/')[-1]
            # load_coco_json
            with open('/home/xz/work/ultralytics/onnx_model_tools/coco.json', 'r') as f:
                images_ids = json.load(f)
            images_ids = images_ids['images']
            id_list = list(filter(lambda x:x['file_name']==base_name,images_ids))
            d_pred['image_id'] = id_list[0]['id']
            
            d_pred['bbox'] = bbox
            d_pred['score'] = float(pred[i][4])
            d_pred['category_id'] = int(pred[i][5])
            
            pred_result.append(d_pred)
            
        for j in range(preds.shape[0]):
            x1,x2,x3,x4 = int(preds[j][0]),int(preds[j][1]),int(preds[j][2]),int(preds[j][3])
            cv2.rectangle(image,(x1,x2),(x3,x4),(0,255,0),2)
            font = cv2.FONT_HERSHEY_DUPLEX  # 设置字体
            
        # 图片对象、文本、位置像素、字体、字体大小、颜色、字体粗细
            imgzi = cv2.putText(image, str(preds[j][5]), (x1, x2), font, 2, (254, 67, 101), 5)
        
        cv2.imwrite(save_path,image)
        
        return pred_result
        
        
    def load_model(self,onnx_model_file):
        self.model = onnxruntime.InferenceSession(onnx_model_file, providers=['CUDAExecutionProvider'])
        
    
    def inference(self,img):
        onnx_input_name = self.model.get_inputs()[0].name
        onnx_output_name = self.model.get_outputs()[0].name
        img = img.detach().cpu().numpy()
        onnx_pred_y = self.model.run([onnx_output_name], {onnx_input_name: img})
        
        return onnx_pred_y
        
    def get_model_result(self, image):
        input_data = self.preprocess(image)
        input = self.session.get_inputs()[0]
        output = self.session.run(None, {input.name: input_data})
        
        # PREDS = N  84 8400
        return output

    # def inference(self, image):
    #     output = self.get_model_result(image)
    #     self.postprocess(output)
        
    def evaluate(self,anno_json,pred_json):
        from pycocotools.coco import COCO  # noqa
        anno = COCO(str(anno_json))
        pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)

        eval = COCOeval(anno, pred, 'bbox')
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        pass




def main():
    kit = ModelRunKit('/home/xz/work/ultralytics/onnx_model_tools/best.onnx')
    pred_results = []
    
    base_path = '/home/xz/work/ultralytics/datasets/gameball/images_val.txt'
    f=open(base_path, encoding='gbk')

    images_list=[]
    
    for line in f:
        images_list.append(line.strip())
    
    
    for i in images_list:
        img = cv2.imread(os.path.join('/home/xz/work/ultralytics/datasets/gameball',i))
        kit.load_model('/home/xz/work/ultralytics/onnx_model_tools/best.onnx')
        img = kit.preprocess(img)
        
        preds = kit.inference(img)
        prediction = torch.from_numpy(preds[0])
        
        final_output = kit.postprocess(prediction)
        visual_output = final_output.cpu().numpy()
        image_path = os.path.join('/home/xz/work/ultralytics/datasets/gameball',i)
        save_path = os.path.join('/home/xz/work/ultralytics/datasets/gameball/test/predict',i.split('/')[-1])
        
        pred_result = kit.visualize(image_path,save_path,visual_output)
        
        pred_results = pred_results+ pred_result
    with open('/home/xz/work/ultralytics/onnx_model_tools/pred_json.json',"w") as f:
        json.dump(pred_results,f)
        print("已生成news_json.json文件...")
    anno_json = '/home/xz/work/ultralytics/onnx_model_tools/coco.json'
    pred_json = '/home/xz/work/ultralytics/onnx_model_tools/pred_json.json'
    
    kit.evaluate(anno_json,pred_json)


if __name__ == '__main__':
    main()

    
    
