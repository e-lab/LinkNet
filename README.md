# LinkNet

This repository contains our Torch7 implementation of the network developed by us at e-Lab.
You can go to out [blogpost](https://codeac29.github.io/projects/linknet/) for this network for further details.

## Dependencies:

+ [Torch7](https://github.com/torch/distro) : you can follow our installation step specified [here](https://github.com/e-lab/tutorials/blob/master/Setup-an-Ubuntu-GPU-box.md)
+ [VideoDecoder](https://github.com/e-lab/torch-toolbox/tree/master/Video-decoder) : video decoder for torch that utilizes avcodec library.
+ [Profiler](https://github.com/e-lab/Torch7-profiling) : use it to calculate # of paramaters, operations and forward pass time of any network trained using torch.

Currently the network can be trained on two datasets:

| Datasets | Input Resolution | # of classes |
|:--------:|:----------------:|:------------:|
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) (cv) | 768x576 | 11 |
| [Cityscapes](https://www.cityscapes-dataset.com/) (cs) | 1024x512 | 19 |

To download both datasets, follow the link provided above.
Both the datasets are first of all resized by the training script and if you want then you can cache this resized data using `--cachepath` option.
In case of CamVid dataset, the available video data is first split into train/validate/test set.
This is done using [prepCamVid.lua](data/prepCamVid.lua) file.
[dataDistributionCV.txt](misc/dataDistributionCV.txt) contains the detail about splitting of CamVid dataset.
These things are automatically run before training of the network.

## Files/folders and their usage:

* [run.lua](run.lua)    : main file
* [opts.lua](opts.lua)  : contains all the input options used by the tranining script
* [data](data)          : data loaders for loading datasets
* [models]                : all the model architectures are defined here
* [train.lua](train.lua) : loading of models and error calculation
* [test.lua](test.lua)  : calculate testing error and save confusion matrices

There are three model files present in `models` folder:

* [model.lua](models/model.lua) : our LinkNet architecture
* [model-res-dec.lua](models/model-res-dec.lua) : LinkNet with residual connection in each of the decoder blocks.
  This slightly improves the result but we had to use `bilinear interpolation` in residual connection because of which we were not able to run our trained model on TX1.
* [nobypass.lua](models/nobypass.lua) : this architecture does not use any link between encoder and decoder.
  You can use this model to verify if connecting encoder and decoder modules actually improve performance.

A sample command to train network is given below:

```
th main.lua --datapath /Datasets/Cityscapes/ --cachepath /dataCache/cityscapes/ --dataset cs --model models/model.lua --save /Models/cityscapes/ --saveTrainConf --saveAll --plot
```

### License

This software is released under a creative commons license which allows for personal and research use only.
For a commercial license please contact the authors.
You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/
