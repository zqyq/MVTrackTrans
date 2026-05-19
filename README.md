# Multi-view Crowd Tracking Transformer with View-Ground Interactions Under Large Real-World Scenes, Qi Zhang, Jixuan Chen, Kaiyi Zhang, Xinquan Yu, Antoni B. Chan, and Hui Huang, CVPR 2026
![MVTrackTrans](https://github.com/zqyq/MVTrackTrans/blob/main/PipeLine.png)
## Abstract
Multi-view crowd tracking estimates each person's tracking trajectories on the ground of the scene. Recent research works mainly rely on CNNs-based multi-view crowd tracking architectures, and most of them are evaluated and compared on relatively small datasets, such as Wildtrack and MultiviewX. Since these two datasets are collected in small scenes and only contain tens of frames in the evaluation stage, it is difficult for the current methods to be applied to real-world applications where scene size and occlusion are more complicated. In this paper, we propose a Transformer-based multi-view crowd tracking model, \textit{MVTrackTrans}, which adopts interactions between camera views and the ground plane for enhanced multi-view tracking performance. Besides, for better evaluation, we collect and label two large real-world multi-view tracking datasets, MVCrowdTrack and CityTrack, which contain a much larger scene size over a longer time period. Compared with existing methods on the two large and new datasets, the proposed MVTrackTrans model achieves better performance, demonstrating the advantages of the model design in dealing with large scenes.

## Video
[video](https://private-user-images.githubusercontent.com/86394027/364318268-12d33440-e586-451f-87cc-d68d8f36a618.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjU0NDMyNzIsIm5iZiI6MTcyNTQ0Mjk3MiwicGF0aCI6Ii84NjM5NDAyNy8zNjQzMTgyNjgtMTJkMzM0NDAtZTU4Ni00NTFmLTg3Y2MtZDY4ZDhmMzZhNjE4Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA5MDQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwOTA0VDA5NDI1MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTM5M2I1N2ZhNDRjZjg5ODliMDdjOTAzMGE5Zjk3NjY1Y2ZlM2E5ZGVjZDE0NzA4ZmEzY2ViYjZjZjFhMzFkNjgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.4SadldmRJxctAfMHtKrCR5lLGm-k4bsKawxfe_021Hk)

## Poster 
CVPR 2026 poster:
![Poster](https://github.com/zqyq/MVTrackTrans/blob/main/poster_fig.png "Poster")

## Paper, Code, and Datasets
You can download the dataset CityTrack at 

Baidu: this [link](https://pan.baidu.com/s/1pIMExmYV-ttQd8x2Ky3bRA) code: 1390 and the dataset MVCrowdTrack [link](https://pan.baidu.com/s/170ae9vUmoPX_yDlPzKQ3cw) code: 2460

Dropbox: Paper PDF, Code, and Datasets [link](https://www.dropbox.com/scl/fo/3jjeq5jkw4gmg488tj6pv/APCR-ifra2RxjOuOWdlq1NA?rlkey=uqt65idvb7jzczuqpxyc3tm6u&st=teiy3zq6&dl=0) code: iMUSELab

## Dependencies
- python
- pytorch & torchvision
- numpy
- matplotlib
- pillow
- opencv-python
- kornia
- tqdm
- shutil
- Deformable-DETR [link](https://github.com/fundamentalvision/Deformable-DETR)
- DCNv2 [link](https://github.com/CharlesShang/DCNv2)
- accelerate

## Data Preparation
In the code implementation, the root path of the four main datasets are defined as ```/mnt/d/dataset```.

```
Datasets
|__CityTrack
    |__...
|__MVCrowdTrack
    |__...
|__Wildtrack
    |__...
|__MultiviewX
    |__...
```
### Folder structure
```
MVCrowdTrack
├── annotations_positions
│   ├── 00000.json
│   ├── 00001.json
│   └── ...
├── calibrations
│   ├── extrinsic
│   │   ├── extr_Camera1.xml
│   │   ├── extr_Camera2.xml
│   │   └── ...
│   └── intrinsic
│       ├── intr_Camera1.xml
│       ├── intr_Camera2.xml
│       └── ...
├── gt.txt
├── Image_subsets
│   ├── C1
│   ├── C2
│   └── ...
 ```

## Training
For training, first navigate to the corresponding directory, then specify the GPUs for multi-GPU training:
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py with deformable
```
## Testing
For testing, run:
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch test.py with deformable
```

### Cite our paper:
   If you feel the paper, dataset or code, is useful to you, please cite our paper.
   
    @inproceedings{zhang2026MVTrackTrans,
    title={Multi-view Crowd Tracking Transformer with View-Ground Interactions Under Large Real-World Scenes},
    author={Zhang, Qi, Chen, Jixuan, Zhang, Kaiyi, Yu, Xinquan, Chan, Antoni B, and Huang, Hui},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    year={2026}
    }
