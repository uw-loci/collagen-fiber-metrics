# Collagen fiber extraction and analysis in cancer tissue microenvironment

## Installation
Install [anaconda/miniconda](https://docs.conda.io/en/latest/miniconda.html)  
```
  $ conda env create --name collagen --file env.yml
  $ conda activate collagen
```
If there are issues with OpenCV  
```
  $ pip install opencv-contrib-python
```
Install ridge-detection package  
```
  $ pip install ridge-detection
```

## CenterLine class
This class handles the conversion between a centerline mask and a dictionary that contains the coordinates of individual centerlines, as well as the fiber property computation.   
For example, create a `CenterLine` object using a binary centerline mask:  
```python
    from centerline import CenterLine
    centerline = CenterLine(centerline_image=io.imread("examples/example_centerline.png"), associate_image=io.imread("examples/example_image.tif"))
```
`centerline_image` is a binary centerline mask, `associate_image` is an optional image of collagen fiber to which the binary mask corresponds.  
Compute the fiber properties:  
```python
    centerline.compute_fiber_feats() 
    print(dict(list(centerline.feats.items())[:-1]))
```
Create a colorized overlay of fiber centerline instances on the collagen fiber image:  
```python
    centerline_res.create_overlay()
```

<div align="center">
  <img src="thumbnails/output.png" width="1000px" />
</div>

Other ways to create a `CenterLine` object, check notebook [centerline-basics.ipynb](centerline-basics.ipynb).   
### Read ctFIRE results or use ridge detection
Check notebook [centerline-baselines.ipynb](centerline-baselines.ipynb)  

## FiberExtractor class
This class handles the computation of a fiber centerline mask from a neural network.  
Process a collagen fiber image:  
```python
    from fiber_extraction import FiberExtractor, UNet
    from skimage import io, img_as_uint
    net = UNet(1, 1, 16, True).eval()
    net.load_state_dict(torch.load('weights/netG.pt'))
    fiber_extractor = FiberExtractor(net)
    im_arr = img_as_uint(io.imread('examples/test_input.png'))
    result = fiber_extractor.compute(im_arr)
```
Compute the normalization range for 16-bit image.
```python
    fiber_extractor.normalization_range(file_list=file_list)
```
`file_list` is a list of directories to the image files. This function computes the range in the 16-bit image set to be stretched to the range of `(0, 65535)`.  
