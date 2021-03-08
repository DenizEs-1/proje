import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized




########################### önceki istenilende kurdugum clas yapısı, bu yaptıklarımda kullanmadım, ilerde isime yaramazsa silicem############################################
class resultss:
    listcordinate=[]
    listname=[]
    
    def __init__(self):
        pass

    def deniz(self,name,cordinate):
        self.name=name
        self.listname.append(name)
        self.listcordinate.append(cordinate)
        
        
    def Show(self):
        print(self.listname)
        print(self.listcordinate)
        


def detect(save_img=False):
    adet_sayisi=0
    cikissüresi=0
    p1=resultss()
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                sınıf=[]
                kordinat=[]
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        
                        label = f'{names[int(cls)]} {conf:.2f}'
                        clas = f'{names[int(cls)]}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                      
                        maps= torch.tensor(xyxy).view(1, 4)[0]
                        konum=maps.numpy()
                        sınıf.append(clas)
                        kordinat.append(konum)
                       # maps=str(maps).strip("tensor")
                       # p1.deniz(clas, konum)
                
                
     ####################### ilk cikisi bulup kordinatlarini icin #########################
                ilkcikis=0 # ilk cikisin ilk x degerini tutacak
                ilkcikisikincix=0 # ilk cikisin ikinci x degerini tutacak
                
                if len(sınıf)==4:              
                    if (sınıf[0] == "CIKIS" and sınıf[2] =="CIKIS"):
                        if(kordinat[0][0]<kordinat[2][0]):
                            ilkcikis = kordinat[0][0]
                            ilkcikisikincix=kordinat[0][2]
                        else:
                            ilkcikis = kordinat[2][0]
                            ilkcikisikincix=kordinat[2][2]
                        
                    elif (sınıf[1] == "CIKIS" and sınıf[2] =="CIKIS"):
                        if(kordinat[1][0]<kordinat[2][0]):
                            ilkcikis = kordinat[1][0]
                            ilkcikisikincix=kordinat[1][2]
                        else:
                            ilkcikis = kordinat[2][0]
                            ilkcikisikincix=kordinat[2][2]
                            
                    elif (sınıf[0] == "CIKIS" and sınıf[1] =="CIKIS"):
                        if(kordinat[0][0]<kordinat[1][0]):
                            ilkcikis = kordinat[0][0]
                            ilkcikisikincix=kordinat[0][2]
                        else:
                            ilkcikis = kordinat[1][0]
                            ilkcikisikincix=kordinat[1][2]
                            
                    elif (sınıf[0] == "CIKIS" and sınıf[3] =="CIKIS"):
                        if(kordinat[0][0]<kordinat[1][0]):
                            ilkcikis = kordinat[0][0]
                            ilkcikisikincix=kordinat[0][2]
                        else:
                            ilkcikis = kordinat[3][0]
                            ilkcikisikincix=kordinat[3][2]
                            
                    elif (sınıf[1] == "CIKIS" and sınıf[3] =="CIKIS"):
                        if(kordinat[1][0]<kordinat[3][0]):
                            ilkcikis = kordinat[1][0]
                            ilkcikisikincix=kordinat[1][2]
                        else:
                            ilkcikis = kordinat[3][0]
                            ilkcikisikincix=kordinat[3][2]
                    elif (sınıf[2] == "CIKIS" and sınıf[3] =="CIKIS"):
                        if(kordinat[2][0]<kordinat[3][0]):
                            ilkcikis = kordinat[2][0]
                            ilkcikisikincix=kordinat[2][2]
                        else:
                            ilkcikis = kordinat[3][0]
                            ilkcikisikincix=kordinat[3][2]

                elif len(sınıf)==3:              
                    if (sınıf[0] == "CIKIS" and sınıf[2] =="CIKIS"):
                        if(kordinat[0][0]<kordinat[2][0]):
                            ilkcikis = kordinat[0][0]
                            ilkcikisikincix=kordinat[0][2]
                        else:
                            ilkcikis = kordinat[2][0]
                            ilkcikisikincix=kordinat[2][2]
                        
                    elif (sınıf[1] == "CIKIS" and sınıf[2] =="CIKIS"):
                        if(kordinat[1][0]<kordinat[2][0]):
                            ilkcikis = kordinat[1][0]
                            ilkcikisikincix=kordinat[1][2]
                        else:
                            ilkcikis = kordinat[2][0]
                            ilkcikisikincix=kordinat[2][2]
                    elif (sınıf[0] == "CIKIS" and sınıf[1] =="CIKIS"):
                        if(kordinat[0][0]<kordinat[1][0]):
                            ilkcikis = kordinat[0][0]
                            ilkcikisikincix=kordinat[0][2]
                        else:
                            ilkcikis = kordinat[1][0]
                            ilkcikisikincix=kordinat[1][2]
                                                    
                elif len(sınıf)==2:
                    if (sınıf[0] == "CIKIS"):
                        ilkcikis = kordinat[0][0]
                        ilkcikisikincix=kordinat[0][2]
                    elif (sınıf[1] == "CIKIS"):
                        ilkcikis = kordinat[1][0]
                        ilkcikisikincix=kordinat[1][2]
                        
                elif (len(sınıf)==1 and sınıf[0] == "CIKIS"):
                    ilkcikis = kordinat[0][0]
                    ilkcikisikincix=kordinat[0][2]
                

           
                
      ###################düsüp düsmedigini kontrol et ###################          
            
                if (ilkcikis<=668 ): #sistem en sola dayandiginda ilk cikisin 1.x degeri 667-668
                    cv2.putText(im0,"son noktaya dayali", (10,700),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
                    if(ilkcikisikincix>=703 and cikissüresi<22): #2.kez cikisindaki ortalama frame sayisi => 22
                        #cv2.putText(im0,"ilk kez cikti", (700,70),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3)
                        cikissüresi+=1 # ilk cikista kac frame boyunca dısarda onu hesaplamak icin => 13-20 frame
                        cv2.putText(im0,str(cikissüresi), (10,400),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3)
                        
                        
                if(cikissüresi>=22 and ilkcikisikincix>=703 and ilkcikis<=668 ): # 2.cıkısta kalıbın düsüp düsmedigine bakılan yer
                    cikissüresi+=1
                    #cv2.putText(im0,"2. kez cikti", (650,70),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3)
                    cv2.putText(im0,str(cikissüresi), (10,400),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3) 
                    if("KALIP" in sınıf):                              
                        cv2.putText(im0,"kalip dusmemis", (10,70),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3)
                    elif("BOS" in sınıf):                              
                        cv2.putText(im0,"kalip dusmus", (10,70),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3)  
                if (ilkcikis>=800 and cikissüresi >= 24): #sistem geri dönerken=> adet sayısı hesapnıyor, cıkılı kaldıgı süre sifirlaniyor
                    cikissüresi=0
                    adet_sayisi+=1
                    cv2.putText(im0,str(adet_sayisi), (10,70),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3)
                    
                cv2.putText(im0,str(ilkcikis), (800,700),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3) # ilk cikisin 1. x degerini görmek icin
                cv2.putText(im0,str(ilkcikisikincix), (800,900),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3)  # ilk cikisin 2. x degerini görmek icin
                                
                         
                print(sınıf) #sınıf elemanlarını görmek icin
                
                sınıf.clear()  # diger resme gecince tutulan sınıf ve kordinatlar silinsin diye
                kordinat.clear()
                        
                 

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if True:
                cv2.imshow("result", im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    
    
    
    
  
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='last.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()
    
    
    
    

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

