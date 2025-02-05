# napari-UniSPAC
The napari plugin for UniSPAC [A Unified Segmentation framework for Proofreading and Annotation in Connectomics]. UniSPAC provides interactive 3D neuron segmentation. Neuron segmentation, proofreading and tracking can be done with just mouse clicks, which is much more efficient than existing tools.

## Requirements

A system with enough GPU memory (recommended 24GB) and pytorch installed.

## Installation

Step 1: install napari via pip:

```shell
pip install -U 'napari[all]'
```

Step 2: install `napari-UniSPAC`

```shell
git clone https://github.com/ddd9898/napari-UniSPAC.git
cd napari-UniSPAC
pip install -e .
```

Step 3: run napari:

```shell
napari
```

You can familiarise yourself with how UniSPAC's napari plugin operates by labeling  `test_roi1_sub_z0-100.tiff`, which is an example of Drosophila vEM images.
