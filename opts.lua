--------------------------------------------------------------------------------
-- Contains options required by run.lua
--
-- Written by: Abhishek Chaurasia
-- Dated:      6th June, 2016
--------------------------------------------------------------------------------

local opts = {}

lapp = require 'pl.lapp'
function opts.parse(arg)
   local opt = lapp [[
   Command line options:
   Training Related:
   -r,--learningRate       (default 5e-4)        learning rate
   -d,--learningRateDecay  (default 1e-7)        learning rate decay (in # samples)
   -w,--weightDecay        (default 2e-4)        L2 penalty on the weights
   -m,--momentum           (default 0.9)         momentum
   -b,--batchSize          (default 8)           batch size
   --maxepoch              (default 300)         maximum number of training epochs
   --plot                                        plot training/testing error
   --showPlot                                    display the plots
   --lrDecayEvery          (default 100)         Decay learning rate every X epoch by 1e-1

   Device Related:
   -t,--threads            (default 8)           number of threads
   -i,--devid              (default 1)           device ID (if using CUDA)
   --nGPU                  (default 4)           number of GPUs you want to train on
   --save                  (default /media/)     save trained model here

   Dataset Related:
   --channels              (default 3)
   --datapath              (default /media/)     dataset location
   --dataset               (default cs)          dataset type: cv(CamVid)/cvs(CamVidSeg)/cs(cityscapes)/su(SUN)/rp(representation)
   --cachepath             (default /media/)     cache directory to save the loaded dataset
   --imHeight              (default 512)         image height  (576 cv/512 cs)
   --imWidth               (default 1024)        image width   (768 cv/1024 cs)

   Model Related:
   --model                 (default models/model.lua)
                           Path of model definition
   --pretrained            (default /media/HDD1/Models/pretrained/resnet-18.t7)
                           pretrained encoder for which you want to train your decoder

   Saving/Displaying Information:
   --saveTrainConf                               Save training confusion matrix
   --saveAll                                     Save all models and confusion matrices
   --printNorm                                   For visualize norm factor while training
 ]]

   return opt
end

return opts
