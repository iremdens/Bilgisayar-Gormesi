
kayisi - v1 2024-11-10 1:50pm
==============================

This dataset was exported via roboflow.com on November 10, 2024 at 1:52 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 22 images.
Kayisi are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Random exposure adjustment of between -19 and +19 percent
* Salt and pepper noise was applied to 4.97 percent of pixels

The following transformations were applied to the bounding boxes of each image:
* Random Gaussian blur of between 0 and 1.9 pixels


