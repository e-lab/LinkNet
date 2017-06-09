# Dataset loader description

In case of CamVid dataset, first the video files need to be sampled using [prepCamVid.lua](prepCamVid.lua) script.


## Folder/file structure for each dataset:

1. CamVid:

    ```
    CamVid/
    ├── 1
    │   ├── input/01TP_extract.avi
    │   └── label/
    │
    ├── 2
    │   ├── input/0005VD.MXF
    │   └── label/
    │
    ├── 3
    │   ├── input/0006R0.MXF
    │   └── label/
    │
    └── 4
        ├── input/0016E5.MXF
        └── label/
    ```

2. Cityscapes:

    ```
    Cityscapes/
    └── leftImg8bit
        ├── train
        └── val
    ```
