--------------------------------------------------------------------------------
-- Decoder for ResNet-18
--
-- Abhishek Chaurasia
-- September 2016
--------------------------------------------------------------------------------

torch.setdefaulttensortype('torch.FloatTensor')
require 'cudnn'
local nn = require 'nn'

--------------------------------------------------------------------------------
local histClasses = opt.datahistClasses
local classes = opt.dataClasses

--------------------------------------------------------------------------------
-- Initialization functions
local function ConvInit(v)
   local n = v.kW*v.kH*v.nOutputPlane
   v.weight:normal(0, math.sqrt(2/n))
end

local function BNInit(v)
   v.weight:fill(1)
   v.bias:zero()
end

print('\n\27[31m\27[4mConstructing Neural Network\27[0m')

print('Using pretrained ResNet-18')
-- loading model
local oModel = torch.load(opt.pretrained)

-- Getting rid of classifier
oModel:remove(11)
oModel:remove(10)
oModel:remove(9)
-- Last layer is size 512x8x8

-- Function and variable definition
local iChannels = 64
local Convolution = cudnn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

-- Handles decoder
local function decode(iFeatures, oFeatures, stride, adjS)
   local mainBlock = nn.Sequential()
   mainBlock:add(Convolution(iFeatures, iFeatures/4, 1, 1, 1, 1, 0, 0))
   mainBlock:add(SBatchNorm(iFeatures/4, 1e-3))
   mainBlock:add(nn.ReLU(true))
   mainBlock:add(nn.SpatialFullConvolution(iFeatures/4, iFeatures/4, 3, 3, stride, stride, 1, 1, adjS, adjS))
   mainBlock:add(SBatchNorm(iFeatures/4, 1e-3))
   mainBlock:add(nn.ReLU(true))
   mainBlock:add(Convolution(iFeatures/4, oFeatures, 1, 1, 1, 1, 0, 0))
   mainBlock:add(SBatchNorm(oFeatures, 1e-3))
   mainBlock:add(nn.ReLU(true))

   -- Initialization
   for i = 1, 2 do
      ConvInit(mainBlock:get(1))
      ConvInit(mainBlock:get(4))
      ConvInit(mainBlock:get(7))

      BNInit(mainBlock:get(2))
      BNInit(mainBlock:get(5))
      BNInit(mainBlock:get(8))
   end

   return mainBlock
end

-- Creates count residual blocks with specified number of features
local function layer(layerN, features)
   iChannels = features
   local s = nn.Sequential()
   for i=1, 2 do
      s:add(oModel:get(4+layerN):get(i))
   end
   return s
end

-- Creates bypass modules for decoders
local function bypass2dec(features, layers, stride, adjS)
   local container = nn.Sequential()
   local accum = nn.ConcatTable()
   local prim = nn.Sequential()     -- Container for encoder
   local oFeatures = iChannels

   accum:add(prim):add(nn.Identity())

   -- Add the bottleneck modules
   prim:add(layer(layers, features))
   if layers == 4 then
      --DECODER
      prim:add(decode(features, oFeatures, 2, 1))
      return container:add(accum):add(nn.CAddTable()):add(nn.ReLU(true))
   end

   -- Move on to next bottleneck
   prim:add(bypass2dec(2*features, layers+1, 2, 1))

   -- Add decoder module
   prim:add(decode(features, oFeatures, stride, adjS))
   return container:add(accum):add(nn.CAddTable()):add(nn.ReLU(true))
end

-- Model definition starts
local model
if paths.filep(opt.save .. '/all/model-last.net') then
   model = torch.load(opt.save .. '/all/model-last.net')
else
   model = nn.Sequential()

   model:add(oModel:get(1))
   model:add(oModel:get(2))
   model:add(oModel:get(3))

   model:add(oModel:get(4))
   model:add(bypass2dec(64, 1, 1, 0))

   -- Decoder section without bypassed information
   model:add(nn.SpatialFullConvolution(64, 32, 3, 3, 2, 2, 1, 1, 1, 1))
   model:add(SBatchNorm(32))
   model:add(ReLU(true))
   -- 64x128x128
   model:add(Convolution(32, 32, 3, 3, 1, 1, 1, 1))
   model:add(SBatchNorm(32, 1e-3))
   model:add(ReLU(true))
   -- 32x128x128
   model:add(nn.SpatialFullConvolution(32, #classes, 2, 2, 2, 2, 0, 0, 0, 0))
   -- Model definition ends here

   -- Initialize convolutions and batch norm existing in later stage of decoder
   for i = 1, 2 do
      ConvInit(model:get(#model))
      ConvInit(model:get(#model))
      ConvInit(model:get(#model-3))
      ConvInit(model:get(#model-3))
      ConvInit(model:get(#model-6))
      ConvInit(model:get(#model-6))

      BNInit(model:get(#model-2))
      BNInit(model:get(#model-2))
      BNInit(model:get(#model-5))
      BNInit(model:get(#model-5))
   end
end

if cutorch.getDeviceCount() > 1 then
   local gpu_list = {}
   for i = 1,cutorch.getDeviceCount() do gpu_list[i] = i end
   model = nn.DataParallelTable(1, true, false):add(model:cuda(), gpu_list)
   print('\27[32m' .. opt.nGPU .. " GPUs being used\27[0m")
end

-- Loss: NLL
print('Defining loss function...')
local classWeights = torch.pow(torch.log(1.02 + histClasses/histClasses:max()), -1)
-- classWeights[1] = 0

loss = cudnn.SpatialCrossEntropyCriterion(classWeights)

model:cuda()
loss:cuda()

----------------------------------------------------------------------
-- return package:
return {
   model = model,
   loss = loss,
}
