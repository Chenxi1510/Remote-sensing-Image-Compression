close;
clear  all:

GT_list = 'F:\评价指标测试代码\ISPR\';        % 原始图像库路径
Enconded_images_list='F:\评价指标测试代码\ISPR\';              % 重构图像库路径
A=[];
Input= dir([GT_list '*.png']); % 遍历原始图像库路径下所有png格式文件
Output= dir([Enconded_images_list '*.png']);         % 遍历重构图像库路径下所有png格式文件
length(GT);
for i = 1:length(Input)              % 遍历结构体就可以一一处理图片了
    GT = imread([GT_list Input(i).name]);     %读取原始图像库的每张图片
    Enconded_images = imread([Enconded_images_list Output(i).name]);            %读取重建图像库图片
    FITS=VEIE(GT,Enconded_images);
    A(i)=FITS;  
    sprintf('%d,%d',i,A(i));
end






�
