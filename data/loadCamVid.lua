----------------------------------------------------------------------
-- Sample CamVid videos to get train/test images
-- and corresponding labels
--
-- Abhishek Chaurasia,
-- November 2016
----------------------------------------------------------------------
require 'image'
require 'xlua'
----------------------------------------------------------------------
local trsize = 367
local tesize = 233
local vasize = 101
local dirRoot  = opt.datapath

local red      = '\27[31m'
local green    = '\27[32m'
local uline    = '\27[4m'
local resetCol = '\27[0m'
----------------------------------------------------------------------

-- Classes in Alphabetical Order
local classes = {'Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car', 'CarLuggagePram', 'Child',
                 'ColumnPole', 'Fence', 'LaneMkgsDriv', 'LaneMkgsNonDriv', 'MiscText', 'MotorcycleScooter', 'OtherMoving', 'ParkingBlock',
                 'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk', 'SignSymbol', 'Sky', 'SUVPickupTruck', 'TrafficCone',
                 'TrafficLight', 'Train', 'Tree', 'TruckBus', 'Tunnel', 'VegetationMisc', 'Void', 'Wall'}

local conClasses = {'Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car', 'CarLuggagePram', 'Child',
                    'ColumnPole', 'Fence', 'LaneMkgsDriv', 'LaneMkgsNonDriv', 'MiscText', 'MotorcycleScooter', 'OtherMoving', 'ParkingBlock',
                    'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk', 'SignSymbol', 'Sky', 'SUVPickupTruck', 'TrafficCone',
                    'TrafficLight', 'Train', 'Tree', 'TruckBus', 'Tunnel', 'VegetationMisc', 'Void', 'Wall'}

----------------------------------------------------------------------
local classMap = {[25]={1},   -- Animal
                  [50]={3},   -- Archway
                  [11]={11},  -- Bicyclist
                  [9] ={3},   -- Bridge
                  [32]={3},   -- Building
                  [18]={9},   -- Car
                  [19]={10},  -- CartLuggagePram
                  [57]={10},  -- Child
                  [62]={8},   -- ColumnPole
                  [22]={4},   -- Fence
                  [35]={2},   -- LaneMkgsDiv
                  [49]={2},   -- LaneMkgsNonDiv
                  [41]={7},   -- MiscText
                  [51]={11},  -- MotorcycleScooter
                  [37]={9},   -- OtherMoving
                  [30]={6},   -- ParkingBlock
                  [20]={10},  -- Pedestrian
                  [38]={2},   -- Road
                  [43]={2},   -- RoadShoulder
                  [3] ={6},   -- Sidewalk
                  [58]={7},   -- SignSymbol
                  [42]={12},  -- Sky
                  [27]={9},   -- SUVPickupTruck
                  [1] ={8},   -- TrafficCone
                  [5] ={7},   -- TrafficLight
                  [54]={9},   -- Train
                  [40]={5},   -- Tree
                  [59]={9},   -- TruckBus
                  [17]={3},  -- Tunnel
                  [60]={5},   -- VegetationMisc
                  [0] ={1},   -- Void
                  [28]={4}}   -- Wall

classes = {'Unlabeled',    --1
           'Road' ,        --2
           'Building',     --3
           'Fence',        --4
           'Tree',         --5
           'Sidewalk',     --6
           'SignSymbol',   --7
           'ColumnPole',   --8
           'Car',          --9
           'Pedestrian',   --10
           'Bicyclist',    --11
           'Sky'}          --12
conClasses = {
              'Road' ,        --2
              'Building',     --3
              'Fence',        --4
              'Tree',         --5
              'Sidewalk',     --6
              'SignSymbol',   --7
              'ColumnPole',   --8
              'Car',          --9
              'Pedestrian',   --10
              'Bicyclist',    --11
              'Sky'}          --12


print '\n\27[31m\27[4mLoading CamVid dataset\27[0m\27[31m ...\27[0m'
print('# of classes: ' .. #classes ..', classes: ', classes)

--------------------------------------------------------------------------------
-- Initialize data structures:
--------------------------------------------------------------------------------
local trainData = {
      data      = torch.FloatTensor(trsize, opt.channels, opt.imHeight , opt.imWidth),
      labels    = torch.FloatTensor(trsize, opt.imHeight , opt.imWidth),
      preverror = 1e10, -- a really huge number
      size      = function() return trsize end
}
local testData  = {
      data      = torch.FloatTensor(tesize, opt.channels, opt.imHeight , opt.imWidth),
      labels    = torch.FloatTensor(tesize, opt.imHeight , opt.imWidth),
      preverror = 1e10, -- a really huge number
      size      = function() return tesize end
}
local valData   = {
      data      = torch.FloatTensor(vasize, opt.channels, opt.imHeight , opt.imWidth),
      labels    = torch.FloatTensor(vasize, opt.imHeight , opt.imWidth),
      preverror = 1e10, -- a really huge number
      size      = function() return vasize end
}

local function rescaleRemapData(sourceData, destData)
   local dataSize = destData:size()
   for i = 1, dataSize do
      xlua.progress(i, dataSize)
      destData.data[i] = image.scale(sourceData.data[i], opt.imWidth, opt.imHeight)
      tempLabel = sourceData.labels[i]

      -- Remap labels based on new class mapping
      tempLabel:apply(
      function(x)
         return classMap[x][1]
      end)
      destData.labels[i] = image.scale(tempLabel, opt.imWidth, opt.imHeight)
   end
end
-----------------------------------------------------------------------------------
-- Main section
-----------------------------------------------------------------------------------
local loadedFromCache = false
local dirName = opt.imHeight .. '_' .. opt.imWidth
local cacheDir = paths.concat(opt.cachepath, dirName)
local camvidCachePath = paths.concat(cacheDir, 'trainData.t7')

if not paths.dirp(cacheDir) then paths.mkdir(cacheDir) end

if opt.cachepath ~= "none" and paths.filep(camvidCachePath) then
   print('\27[32mData cache found at: \27[0m\27[4m' .. cacheDir .. '\27[0m')
   trainData = torch.load(camvidCachePath)
   camvidCachePath = paths.concat(cacheDir, 'testData.t7')
   testData = torch.load(camvidCachePath)
   loadedFromCache = true
   collectgarbage()
else
   -- Check if a compatible version of CamVid data is present or not
   if not paths.filep(opt.cachepath .. '/trainTestVal.t7') then
      paths.dofile('prepCamVid.lua')
      collectgarbage()
   end

   print('Loading dataset as tensor from: ' .. uline .. opt.cachepath .. '/trainTestVal.t7' .. resetCol)
   local dataCache = torch.load(opt.cachepath .. '/trainTestVal.t7')
   local tempLabel = torch.Tensor(720, 960)
   print(red .. 'Rescaling training set' .. resetCol)
   rescaleRemapData(dataCache.trainData, trainData)

   print(red .. 'Rescaling testing set' .. resetCol)
   rescaleRemapData(dataCache.testData, testData)

   print(red .. 'Rescaling validation set' .. resetCol)
   rescaleRemapData(dataCache.valData, valData)

   collectgarbage()
end

print(string.format("%s# of training   data:%s %d", green, resetCol, trainData:size()))
print(string.format("%s# of validation data:%s %d", green, resetCol, valData:size()))
print(string.format("Frame res is: %dx%dx%d",
      trainData.data:size(2), trainData.data:size(3),trainData.data:size(4)))
-----------------------------------------------------------------------------------
if opt.cachepath ~= "none" and not loadedFromCache then
   print('==> Saving data to cache: ' .. cacheDir)
   -- saving training data
   camvidCachePath = paths.concat(cacheDir, 'trainData.t7')
   torch.save(camvidCachePath, trainData)
   -- saving testing data
   camvidCachePath = paths.concat(cacheDir, 'testData.t7')
   torch.save(camvidCachePath, testData)
   -- saving validation data
   camvidCachePath = paths.concat(cacheDir, 'valData.t7')
   torch.save(camvidCachePath, valData)

   collectgarbage()
end

-----------------------------------------------------------------------------------
print '==> Normalizing data'

-- It's always good practice to verify that data is properly
-- normalized.
local trainMean = torch.mean(trainData.data, 1)

for i = 1, trainData.data:size(1) do
   trainData.data[i]:add(-trainMean)
end
for i = 1, valData.data:size(1) do
   valData.data[i]:add(-trainMean)
end
for i = 1, testData.data:size(1) do
   testData.data[i]:add(-trainMean)
end
torch.save(paths.concat(opt.cachepath, dirName, 'stat.t7'), trainMean)

-----------------------------------------------------------------------------------
local classes_td = {[1] = 'classes,targets\n'}
for _,cat in pairs(classes) do
   table.insert(classes_td, cat .. ',1\n')
end

local file = io.open(paths.concat(opt.save, 'categories.txt'), 'w')
file:write(table.concat(classes_td))
file:close()

local histClasses = torch.histc(trainData.labels:double(), #classes, 1, #classes)

-- Exports
opt.dataClasses = classes
opt.dataconClasses = conClasses
opt.datahistClasses = histClasses

-- Chose if you want to send test data or validation data
return {
   trainData = trainData,
   testData = testData,
}
