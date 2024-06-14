# Hyperspectral-Image-Classification
学校的大作业，大创打基础练手项目。最大的感想是从0开始完成一个项目还是非常困难的，故把我这个低水平代码放上来供大家参考。
#使用说明
1.数据集下载链接：https://pan.baidu.com/s/1jMvtSPEVoFp07M4mOOyo3Q 提取码：gt6d，包含IndianPines，Houston，Pavia
2.配置好项目所需要的python环境，将数据集下载好放置在根目录的datasets文件夹下。
3.使用python自带的可视化方法会存在精度的问题，对于Houston和Pavia这种比较大的数据集会边缘模糊，建议将最终的结果导出成mat文件，与所对应的colormap一并拿到matlab当中使用如下代码可视化：

%%  结果展示
result=uint8(result)
colored_image=ind2rgb(result,colormap);
imshow(colored_image);
title('Result');

# 特别感谢
项目中的数据集和一些代码均来自于这个开源的代码https://github.com/danfenghong/IEEE_TGRS_SpectralFormer
初学者看这里的代码会有一些头大，但是还是为我提供了非常好的学习素材，特此鸣谢！

# 联系
代码前后由于时间跨度比较长，并不能保证里面的注释完全正确。
如有任何问题，欢迎与我联系：22331049@bjtu.edu.cn
