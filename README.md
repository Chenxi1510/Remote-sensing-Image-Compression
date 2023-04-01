# Remote-sensing-Image-Compression
Remote-sensing Image Compression


This repository contains the code for reproducing the results with trained models (EGA-Net), NWPU-RSC Dataset, and a novel full-reference IQA FITS.

## Requirements

Python==3.6  
torch==1.6.0  
torchvision==0.7.0

  
## Data download  
  ISPR Vaihingen Data: https://www2.isprs.org/  
  AID Dataset: https://paperswithcode.com/dataset/aid  
  NWPU-RSC Dataset: https://pan.baidu.com/s/1XismIJG5iO9aB9hv0VpofQ 
提取码：1510 

## Sample of the NWPU-RSC
  
![image](https://github.com/Chenxi1510/Remote-sensing-Image-Compression/blob/main/Image/data.JPG)


## Different Visual Tasks on RS Benchmark Dataset

| Datasets                                                            |Images | Images Size                  | Categories | Image format | Objective task   |
|-----------------------------------------------------------------------------------------------------|-------------------------------------|-------------------------------------------------------------|-----------------------------------------|-------------------------------------------|------------------------------------------------|
| :NWPU VHR\-10: | 800   | \                              | 10        | .JPG        | Detection        |ticolumn\{1\}\{c\|\}\{910\}    | \\multicolumn\{1\}\{c\|\}\{1280\*659\}                      | \\multicolumn\{1\}\{c\|\}\{2\}          | \\multicolumn\{1\}\{c\|\}\{\.PNG\}        | \\multicolumn\{1\}\{c\|\}\{Detection\}         |
| \\multicolumn\{1\}\{\|c\|\}\{FAIR1M\\cite\{FAIR1M\}\}                                               | \\multicolumn\{1\}\{c\|\}\{15266\}  | \\multicolumn\{1\}\{c\|\}\{1000\*1000$\\sim$10000\*10000\}  | \\multicolumn\{1\}\{c\|\}\{5\}          | \\multicolumn\{1\}\{c\|\}\{\}             | \\multicolumn\{1\}\{c\|\}\{recognition\}       |
| \\multicolumn\{1\}\{\|c\|\}\{Inria Aerial Image Labeling \\cite\{InriaAerial\}\}                    | \\multicolumn\{1\}\{c\|\}\{360\}    | \\multicolumn\{1\}\{c\|\}\{5000\*5000\}                     | \\multicolumn\{1\}\{c\|\}\{2\}          | \\multicolumn\{1\}\{c\|\}\{\.Geotiff\}    | \\multicolumn\{1\}\{c\|\}\{Detection\}         |
| \\multicolumn\{1\}\{\|c\|\}\{RSOD\\cite\{RSOD1,RSOD2\}\}                                            | \\multicolumn\{1\}\{c\|\}\{976\}    | \\multicolumn\{1\}\{c\|\}\{1044$\\sim$1288\*915$\\sim$992\} | \\multicolumn\{1\}\{c\|\}\{4\}          | \\multicolumn\{1\}\{c\|\}\{\.JPG\}        | \\multicolumn\{1\}\{c\|\}\{Detection\}         |
| \\multicolumn\{1\}\{\|c\|\}\{RSSCN7\\cite\{RSCNN7\}\}                                               | \\multicolumn\{1\}\{c\|\}\{2800\}   | \\multicolumn\{1\}\{c\|\}\{400\*400\}                       | \\multicolumn\{1\}\{c\|\}\{7\}          | \\multicolumn\{1\}\{c\|\}\{\.JPG\}        | \\multicolumn\{1\}\{c\|\}\{Classification\}    |
| \\multicolumn\{1\}\{\|c\|\}\{UC Merced\\cite\{UCMERCED\}\}                                          | \\multicolumn\{1\}\{c\|\}\{2100\}   | \\multicolumn\{1\}\{c\|\}\{256\*256\}                       | \\multicolumn\{1\}\{c\|\}\{21\}         | \\multicolumn\{1\}\{c\|\}\{\.PNG\}        | \\multicolumn\{1\}\{c\|\}\{Classification\}    |
| \\multicolumn\{1\}\{\|c\|\}\{WHU\-RS19\\cite\{WHURS19\} \}                                          | \\multicolumn\{1\}\{c\|\}\{1005\}   | \\multicolumn\{1\}\{c\|\}\{600\*600\}                       | \\multicolumn\{1\}\{c\|\}\{19\}         | \\multicolumn\{1\}\{c\|\}\{\.TIFF\}       | \\multicolumn\{1\}\{c\|\}\{Classification\}    |
| \\multicolumn\{1\}\{\|c\|\}\{RSC11\\cite\{2015RS\}\}                                                | \\multicolumn\{1\}\{c\|\}\{1232\}   | \\multicolumn\{1\}\{c\|\}\{500\*500\}                       | \\multicolumn\{1\}\{c\|\}\{11\}         | \\multicolumn\{1\}\{c\|\}\{\.TIFF\}       | \\multicolumn\{1\}\{c\|\}\{Classification\}    |
| \\multicolumn\{1\}\{\|c\|\}\{SIRI\-Dataset\\cite\{SIRI1,SIRI2\}\}                                   | \\multicolumn\{1\}\{c\|\}\{2400\}   | \\multicolumn\{1\}\{c\|\}\{200\*200\}                       | \\multicolumn\{1\}\{c\|\}\{12\}         | \\multicolumn\{1\}\{c\|\}\{\.TIFF\}       | \\multicolumn\{1\}\{c\|\}\{Classification\}    |
| \\multicolumn\{1\}\{\|c\|\}\{AID Dataset\\cite\{AID\}\}                                             | \\multicolumn\{1\}\{c\|\}\{10000\}  | \\multicolumn\{1\}\{c\|\}\{600\*600\}                       | \\multicolumn\{1\}\{c\|\}\{30\}         | \\multicolumn\{1\}\{c\|\}\{\.JPG\}        | \\multicolumn\{1\}\{c\|\}\{Classification\}    |
| \\multicolumn\{1\}\{\|c\|\}\{NWPU\-RESISC45\\cite\{NWPURESISC45\}\}                                 | \\multicolumn\{1\}\{c\|\}\{31500\}  | \\multicolumn\{1\}\{c\|\}\{256\*256\}                       | \\multicolumn\{1\}\{c\|\}\{45\}         | \\multicolumn\{1\}\{c\|\}\{\.JPG\}        | \\multicolumn\{1\}\{c\|\}\{Classification\}    |
| \\multicolumn\{1\}\{\|c\|\}\{OPTIMAL\-31\\cite\{OPTIMAL31\}\}                                       | \\multicolumn\{1\}\{c\|\}\{1860\}   | \\multicolumn\{1\}\{c\|\}\{256\*256\}                       | \\multicolumn\{1\}\{c\|\}\{31\}         | \\multicolumn\{1\}\{c\|\}\{\.JPG\}        | \\multicolumn\{1\}\{c\|\}\{Classification\}    |
| \\multicolumn\{1\}\{\|c\|\}\{VEDAI Dataset\\cite\{VEDIA\}\}                                         | \\multicolumn\{1\}\{c\|\}\{1210\}   | \\multicolumn\{1\}\{c\|\}\{1024\*1024\}                     | \\multicolumn\{1\}\{c\|\}\{9\}          | \\multicolumn\{1\}\{c\|\}\{\.PNG\}        | \\multicolumn\{1\}\{c\|\}\{Detection\}         |
| \\multicolumn\{1\}\{\|c\|\}\{DLR 3K Vehicle \\cite\{DLR3K\}\}                                       | \\multicolumn\{1\}\{c\|\}\{20\}     | \\multicolumn\{1\}\{c\|\}\{5616\*3744\}                     | \\multicolumn\{1\}\{c\|\}\{2\}          | \\multicolumn\{1\}\{c\|\}\{\.JPG\}        | \\multicolumn\{1\}\{c\|\}\{Detection\}         |
| \\multicolumn\{1\}\{\|c\|\}\{HRSC2016\\cite\{HRSC\}\}                                               | \\multicolumn\{1\}\{c\|\}\{1070\}   | \\multicolumn\{1\}\{c\|\}\{300\*300,1500\*900\}             | \\multicolumn\{1\}\{c\|\}\{1\}          | \\multicolumn\{1\}\{c\|\}\{\.BMP\}        | \\multicolumn\{1\}\{c\|\}\{Detection\}         |
| \\multicolumn\{1\}\{\|c\|\}\{LEVIR\\cite\{LEVIR\}\}                                                 | \\multicolumn\{1\}\{c\|\}\{22000\}  | \\multicolumn\{1\}\{c\|\}\{600\*800\}                       | \\multicolumn\{1\}\{c\|\}\{3\}          | \\multicolumn\{1\}\{c\|\}\{\}             | \\multicolumn\{1\}\{c\|\}\{Detection\}         |
| \\multicolumn\{1\}\{\|c\|\}\{DOTA\\cite\{DOTA\}\}                                                   | \\multicolumn\{1\}\{c\|\}\{2806\}   | \\multicolumn\{1\}\{c\|\}\{800\*800,4000\*4000\}            | \\multicolumn\{1\}\{c\|\}\{15\}         | \\multicolumn\{1\}\{c\|\}\{\.PNG\}        | \\multicolumn\{1\}\{c\|\}\{Detection\}         |
| \\multicolumn\{1\}\{\|c\|\}\{DIOR\\cite\{DIOR\}\}                                                   | \\multicolumn\{1\}\{c\|\}\{23463\}  | \\multicolumn\{1\}\{c\|\}\{800\*800\}                       | \\multicolumn\{1\}\{c\|\}\{20\}         | \\multicolumn\{1\}\{c\|\}\{\.JPG\}        | \\multicolumn\{1\}\{c\|\}\{Detection\}         |
| \\multicolumn\{1\}\{\|c\|\}\{SIMD\\cite\{SIMD\}\}                                                   | \\multicolumn\{1\}\{c\|\}\{5000\}   | \\multicolumn\{1\}\{c\|\}\{1024\*768\}                      | \\multicolumn\{1\}\{c\|\}\{15\}         | \\multicolumn\{1\}\{c\|\}\{\.JPG\}        | \\multicolumn\{1\}\{c\|\}\{Detection\}         |
| \\multicolumn\{1\}\{\|c\|\}\{MTARSI\\cite\{MTARSI\}\}                                               | \\multicolumn\{1\}\{c\|\}\{9385\}   | \\multicolumn\{1\}\{c\|\}\{256\*256\}                       | \\multicolumn\{1\}\{c\|\}\{3\}          | \\multicolumn\{1\}\{c\|\}\{\}             | \\multicolumn\{1\}\{c\|\}\{Detection\}         |
| \\multicolumn\{1\}\{\|c\|\}\{UCM Captions\\cite\{UCMCAPTION\}\}                                     | \\multicolumn\{1\}\{c\|\}\{2100\}   | \\multicolumn\{1\}\{c\|\}\{256\*256\}                       | \\multicolumn\{1\}\{c\|\}\{5\}          | \\multicolumn\{1\}\{c\|\}\{\.PNG\}        | \\multicolumn\{1\}\{c\|\}\{Image Caption\}     |
| \\multicolumn\{1\}\{\|c\|\}\{Sydney caption\\cite\{SYDNEY\}\}                                       | \\multicolumn\{1\}\{c\|\}\{613\}    | \\multicolumn\{1\}\{c\|\}\{500\*500\}                       | \\multicolumn\{1\}\{c\|\}\{5\}          | \\multicolumn\{1\}\{c\|\}\{\.PNG\}        | \\multicolumn\{1\}\{c\|\}\{Image Caption\}     |
| \\multicolumn\{1\}\{\|c\|\}\{RSICD\\cite\{RSICD\}\}                                                 | \\multicolumn\{1\}\{c\|\}\{10921\}  | \\multicolumn\{1\}\{c\|\}\{224\*224\}                       | \\multicolumn\{1\}\{c\|\}\{5\}          | \\multicolumn\{1\}\{c\|\}\{\.PNG\}        | \\multicolumn\{1\}\{c\|\}\{Image Caption\}     |
| \\multicolumn\{1\}\{\|c\|\}\{Aerial to Map\\cite\{Aerialtomap\}\}                                   | \\multicolumn\{1\}\{c\|\}\{2794\}   | \\multicolumn\{1\}\{c\|\}\{600\*600\}                       | \\multicolumn\{1\}\{c\|\}\{2\}          | \\multicolumn\{1\}\{c\|\}\{\}             | \\multicolumn\{1\}\{c\|\}\{Style transfer\}    |
| \\multicolumn\{1\}\{\|c\|\}\{ABCD dataset\\cite\{ABCD\}\}                                           | \\multicolumn\{1\}\{c\|\}\{7506\}   | \\multicolumn\{1\}\{c\|\}\{160\*160\}                       | \\multicolumn\{1\}\{c\|\}\{1\}          | \\multicolumn\{1\}\{c\|\}\{\}             | \\multicolumn\{1\}\{c\|\}\{Change detection\}  |
| \\multicolumn\{1\}\{\|c\|\}\{WHU Building dataset\\cite\{WHUBUILDING\}\}                            | \\multicolumn\{1\}\{c\|\}\{2\}      | \\multicolumn\{1\}\{c\|\}\{32207\*15354\}                   | \\multicolumn\{1\}\{c\|\}\{1\}          | \\multicolumn\{1\}\{c\|\}\{\}             | \\multicolumn\{1\}\{c\|\}\{Change detection\}  |
| \\multicolumn\{1\}\{\|c\|\}\{Season\-varying\\cite\{SEASON\}\}                                      | \\multicolumn\{1\}\{c\|\}\{16000\}  | \\multicolumn\{1\}\{c\|\}\{256\*256\}                       | \\multicolumn\{1\}\{c\|\}\{1\}          | \\multicolumn\{1\}\{c\|\}\{\}             | \\multicolumn\{1\}\{c\|\}\{Change detection\}  |
| \\multicolumn\{1\}\{\|c\|\}\{SYSU\-CD\\cite\{SYSU\}\}                                               | \\multicolumn\{1\}\{c\|\}\{40000\}  | \\multicolumn\{1\}\{c\|\}\{256\*256\}                       | \\multicolumn\{1\}\{c\|\}\{1\}          | \\multicolumn\{1\}\{c\|\}\{\}             | \\multicolumn\{1\}\{c\|\}\{Change detection\}  |
| \\multicolumn\{1\}\{\|c\|\}\{PatternNet dataset\\cite\{Patternnet\} \}                              | \\multicolumn\{1\}\{c\|\}\{30400\}  | \\multicolumn\{1\}\{c\|\}\{256\*256\}                       | \\multicolumn\{1\}\{c\|\}\{38\}         | \\multicolumn\{1\}\{c\|\}\{\}             | \\multicolumn\{1\}\{c\|\}\{Change detection\}  |
| \\multicolumn\{1\}\{\|c\|\}\{NWPU\-RSC\}                                                            | \\multicolumn\{1\}\{c\|\}\{300000\} | \\multicolumn\{1\}\{c\|\}\{1024\*1024\}                     | \\multicolumn\{1\}\{c\|\}\{35\}         | \\multicolumn\{1\}\{c\|\}\{\.PNG\}        | \\multicolumn\{1\}\{c\|\}\{Image Compression\} |



## Visual Results 
### Experimental results on the AID dataset
![image](https://github.com/Chenxi1510/Remote-sensing-Image-Compression/blob/main/Image/AID.png)

Experimental results of different methods on the AID dataset. In each example, the right-most columns represent ground truths for input image. In the other columns, the first row shows the reconstructed results by different methods, and the second row shows the magnified details of the red rectangle.


### Experimental results on the NWPU-RSC dataset
![image](https://github.com/Chenxi1510/Remote-sensing-Image-Compression/blob/main/Image/NWPU-RSC.png)

### Experimental results on the ISPR Vaihingen Dataset
![image](https://github.com/Chenxi1510/Remote-sensing-Image-Compression/blob/main/Image/ISPRresult.png)
Rate-distortion curves of different compression algorithms on the ISPR Vaihingen Dataset. (a) PSNR,  (b) M-SSIM, (c) VIETIM and (d) LPIPS  on the ISPR dataset.



##  Change detection results in different compression methods
![image](https://github.com/Chenxi1510/Remote-sensing-Image-Compression/blob/main/Image/change_detection.JPG)
Visualization of change detection results.(a) Orginal Images, (b) BPG, (c) JPEG2000, (d) GMM, (e) C2F, (f) EGA-Net (lo), (g) EGA-Net (me) and (h) EGA-Net (hi).



