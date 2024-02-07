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

![image](https://github.com/Chenxi1510/Remote-sensing-Image-Compression/blob/main/Image/DIFFERNET.png)


## Data Sources of the NWPU-RSC
![image](https://github.com/Chenxi1510/Remote-sensing-Image-Compression/blob/main/Image/SOURCE.png)



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





### Citation

If you find our project is useful for your research, please cite:
```
@ARTICLE{10247080,
  author={Han, Pengfei and Zhao, Bin and Li, Xuelong},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Edge-Guided Remote-Sensing Image Compression}, 
  year={2023},
  volume={61},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2023.3314012}}

```



