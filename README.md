# WeatherProof: Leveraging Language Guidance for Semantic Segmentation in Adverse Weather
Blake Gella<sup>1*</sup>, Howard Zhang<sup>1*</sup>, Rishi Upadhyay<sup>1</sup>, Tiffany Chang<sup>1</sup>, Nathan Wei<sup>1</sup>, Matthew Waliman<sup>1</sup>, Yunhao Ba<sup>1</sup>, Celso de Melo<sup>3</sup>, Alex Wong<sup>2</sup>, Achuta Kadambi<sup>1</sup>
University of California, Los Angeles<sup>1</sup>, Yale University<sup>2</sup>, US Army Research Laboratory<sup>3</sup>

## Project Webpage
[https://visual.ee.ucla.edu/weatherproof_clip.htm/](https://visual.ee.ucla.edu/weatherproof_clip.htm/)

## Abstract
We propose a method to infer semantic segmentation maps from images captured under adverse weather conditions. We begin by examining existing models on images degraded by weather conditions such as rain, fog, or snow, and found that they exhibit a large performance drop as compared to those captured under clear weather. To control for changes in scene structures, we propose WeatherProof, the first semantic segmentation dataset with accurate clear and adverse weather image pairs that share an underlying scene. Through this dataset, we analyze the error modes in existing models and found that they were sensitive to the highly complex combination of different weather effects induced on the image during capture. To improve robustness, we propose a way to use language as guidance by identifying contributions of adverse weather conditions and injecting that as “side information”. Models trained using our language guidance exhibit performance gains by up to 10.2% in mIoU on WeatherProof, up to 8.44% in mIoU on the widely used ACDC dataset compared to standard training techniques, and up to 6.21% in mIoU on the ACDC dataset as compared to previous SOTA methods.

## Citation

```
@article{gella2023weatherproof,
  title={WeatherProof: A Paired-Dataset Approach to Semantic Segmentation in Adverse Weather},
  author={Gella, Blake and Zhang, Howard and Upadhyay, Rishi and Chang, Tiffany and Waliman, Matthew and Ba, Yunhao and Wong, Alex and Kadambi, Achuta},
  journal={arXiv preprint arXiv:2312.09534},
  year={2023}
}
```

## Dataset
The dataset can be found [here](https://drive.google.com/file/d/1tAZE8uo5wYvJYPZxKc7Mo-ye3D3-3Ld-/view).

## Disclaimer
Please only use the code and dataset for research purposes.

## Contact
Howard Zhang</br>
UCLA, Electrical and Computer Engineering Department</br>
hwdz15508@g.ucla.edu
