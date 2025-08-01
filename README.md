# sanyukta183-Object_detection_in_autonomous_vehicle_using_KITTI_and_BDD100K_dataset

## YOLO V5s Bdd100k training

The following documents is necessary for my project:

- models/custom_yolov5s.yaml
- models/uc_data.yaml
- data/bdd100k.names
- weights/yolov5s.pt
- yolov5s_training_bdd100k.ipynb
- Bdd_preprocessing.ipynb

The process documents of training with pre-trained weights located in the `runs/exp0_yolov5s_bdd_prew`

The process documents of training from scratch located in the `yolov5s_bdd100k/runs/exp1_yolov5s_bdd`

The yolov5s model arcitecture image is `yolov5s_bdd.png`

The 4k test.video was shown in the bilibili website: **https://www.bilibili.com/video/BV1sz4y1Q7wi/**

I provide the preprocessed Bdd100k dataset: **https://1drv.ms/u/s!An7G4eYRvZzthI5HCnVaEGvrdiDWAw?e=v6C4US**



## YOLO V5 Originial Readme 

This repository represents Ultralytics open-source research into future object detection methods, and incorporates our lessons learned and best practices evolved over training thousands of models on custom client datasets with our previous YOLO repository https://github.com/ultralytics/yolov3. **All code and models are under active development, and are subject to modification or deletion without notice.**.

GPU Speed measures end-to-end time per image averaged over 5000 COCO val2017 images using a V100 GPU with batch size 8, and includes image preprocessing, PyTorch FP16 inference, postprocessing and NMS.



## Pretrained Checkpoints

| Model | AP<sup>val</sup> | AP<sup>test</sup> | AP<sub>50</sub> | Speed<sub>GPU</sub> | FPS<sub>GPU</sub> || params | FLOPS |
|---------- |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| [YOLOv5s](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J)    | 36.6     | 36.6     | 55.8     | **2.1ms** | **476** || 7.5M   | 13.2B
| [YOLOv5m](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J)    | 43.4     | 43.4     | 62.4     | 3.0ms     | 333     || 21.8M  | 39.4B
| [YOLOv5l](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J)    | 46.6     | 46.7     | 65.4     | 3.9ms     | 256     || 47.8M  | 88.1B
| [YOLOv5x](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J)    | **48.4** | **48.4** | **66.9** | 6.1ms     | 164     || 89.0M  | 166.4B
| [YOLOv3-SPP](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J)  | 45.6     | 45.5     | 65.2     | 4.5ms     | 222     || 63.0M  | 118.0B


** AP<sup>test</sup> denotes COCO [test-dev2017](http://cocodataset.org/#upload) server results, all other AP results in the table denote val2017 accuracy.  
** All AP numbers are for single-model single-scale without ensemble or test-time augmentation. Reproduce by `python test.py --img 736 --conf 0.001`  
** Speed<sub>GPU</sub> measures end-to-end time per image averaged over 5000 COCO val2017 images using a GCP [n1-standard-16](https://cloud.google.com/compute/docs/machine-types#n1_standard_machine_types) instance with one V100 GPU, and includes image preprocessing, PyTorch FP16 image inference at --batch-size 32 --img-size 640, postprocessing and NMS. Average NMS time included in this chart is 1-2ms/img.  Reproduce by `python test.py --img 640 --conf 0.1`  
** All checkpoints are trained to 300 epochs with default settings and hyperparameters (no autoaugmentation). 


## Requirements

Python 3.7 or later with all `requirements.txt` dependencies installed, including `torch >= 1.5`. To install run:
```bash
$ pip install -U -r requirements.txt
```


## Tutorials

* [Notebook](https://github.com/ultralytics/yolov5/blob/master/tutorial.ipynb) <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
* [Kaggle](https://www.kaggle.com/ultralytics/yolov5-tutorial)
* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)
* [Google Cloud Quickstart](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart)

## Inference

Inference can be run on most common media formats. Model [checkpoints](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J) are downloaded automatically if available. Results are saved to `./inference/output`.
```bash
$ python detect.py --source file.jpg  # image 
                            file.mp4  # video
                            ./dir  # directory
                            0  # webcam
                            rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                            http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream
```

To run inference on examples in the `./inference/images` folder:

```bash
$ python detect.py --source ./inference/images/ --weights yolov5s.pt --conf 0.4

Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.4, device='', fourcc='mp4v', half=False, img_size=640, iou_thres=0.5, output='inference/output', save_txt=False, source='./inference/images/', view_img=False, weights='yolov5s.pt')
Using CUDA device0 _CudaDeviceProperties(name='Tesla P100-PCIE-16GB', total_memory=16280MB)

Downloading https://drive.google.com/uc?export=download&id=1R5T6rIyy3lLwgFXNms8whc-387H0tMQO as yolov5s.pt... Done (2.6s)

image 1/2 inference/images/bus.jpg: 640x512 3 persons, 1 buss, Done. (0.009s)
image 2/2 inference/images/zidane.jpg: 384x640 2 persons, 2 ties, Done. (0.009s)
Results saved to /content/yolov5/inference/output
```

<img src="https://user-images.githubusercontent.com/26833433/83082816-59e54880-a039-11ea-8abe-ab90cc1ec4b0.jpeg" width="500">  

## Reproduce Our Training

Download [COCO](https://github.com/ultralytics/yolov5/blob/master/data/get_coco2017.sh), install [Apex](https://github.com/NVIDIA/apex) and run command below. Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).
```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                48
                                         yolov5l                                32
                                         yolov5x                                16
```
<img src="https://user-images.githubusercontent.com/26833433/84186698-c4d54d00-aa45-11ea-9bde-c632c1230ccd.png" width="900">


## Reproduce Our Environment

To access an up-to-date working environment (with all dependencies including CUDA/CUDNN, Python and PyTorch preinstalled), consider a:

- **Google Cloud** Deep Learning VM with $300 free credit offer: See our [GCP Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart) 
- **Google Colab Notebook** with 12 hours of free GPU time. <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
- **Docker Image** https://hub.docker.com/r/ultralytics/yolov5. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart)

#Project By 

**Sanyukta Patki**
