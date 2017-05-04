----------------------------------------------------------------------
-- Cityscape data loader,
-- Abhishek Chaurasia,
-- August 2016
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset

torch.setdefaulttensortype('torch.FloatTensor')
----------------------------------------------------------------------
-- Cityscape dataset:

local trsize, tesize

trsize = 2975 -- cityscape train images
tesize = 500  -- cityscape validation images
local classes = {'Unlabeled', 'Road', 'Sidewalk', 'Building', 'Wall', 'Fence',
                 'Pole', 'TrafficLight', 'TrafficSign', 'Vegetation', 'Terrain',
                 'Sky', 'Person', 'Rider', 'Car', 'Truck',
                 'Bus', 'Train', 'Motorcycle', 'Bicycle'}
local conClasses = {'Road', 'Sidewalk', 'Building', 'Wall', 'Fence',
                    'Pole', 'TrafficLight', 'TrafficSign', 'Vegetation', 'Terrain',
                    'Sky', 'Person', 'Rider','Car', 'Truck',
                    'Bus', 'Train', 'Motorcycle', 'Bicycle'} -- 19 classes

local nClasses = #classes

--------------------------------------------------------------------------------
-- Ignoring unnecessary classes
local classMap = {[-1] =  {1}, -- licence plate
                  [0]  =  {1}, -- Unlabeled
                  [1]  =  {1}, -- Ego vehicle
                  [2]  =  {1}, -- Rectification border
                  [3]  =  {1}, -- Out of roi
                  [4]  =  {1}, -- Static
                  [5]  =  {1}, -- Dynamic
                  [6]  =  {1}, -- Ground
                  [7]  =  {2}, -- Road
                  [8]  =  {3}, -- Sidewalk
                  [9]  =  {1}, -- Parking
                  [10] =  {1}, -- Rail track
                  [11] =  {4}, -- Building
                  [12] =  {5}, -- Wall
                  [13] =  {6}, -- Fence
                  [14] =  {1}, -- Guard rail
                  [15] =  {1}, -- Bridge
                  [16] =  {1}, -- Tunnel
                  [17] =  {7}, -- Pole
                  [18] =  {1},  -- Polegroup
                  [19] =  {8}, -- Traffic light
                  [20] =  {9}, -- Traffic sign
                  [21] = {10}, -- Vegetation
                  [22] = {11}, -- Terrain
                  [23] = {12}, -- Sky
                  [24] = {13}, -- Person
                  [25] = {14}, -- Rider
                  [26] = {15}, -- Car
                  [27] = {16}, -- Truck
                  [28] = {17}, -- Bus
                  [29] =  {1}, -- Caravan
                  [30] =  {1}, -- Trailer
                  [31] = {18}, -- Train
                  [32] = {19}, -- Motorcycle
                  [33] = {20}, -- Bicycle
                              }

-- saving training histogram of classes
local histClasses = torch.Tensor(#classes):zero()

--------------------------------------------------------------------------------
print '\n\27[31m\27[4mLoading cityscape dataset\27[0m'
print('# of classes: ' .. #classes)

local trainData, testData
local loadedFromCache = false
local dirName = opt.imHeight .. '_' .. opt.imWidth
paths.mkdir(paths.concat(opt.cachepath, dirName))
local cityscapeCachePath = paths.concat(opt.cachepath, dirName, 'data.t7')

if opt.cachepath ~= "none" and paths.filep(cityscapeCachePath) then
   print('\27[32mData cache found at: \27[0m\27[4m' .. cityscapeCachePath .. '\27[0m')
   local dataCache = torch.load(cityscapeCachePath)
   trainData = dataCache.trainData
   testData = dataCache.testData
   histClasses = dataCache.histClasses
   loadedFromCache = true
   dataCache = nil
   collectgarbage()
else
   print("Classes are:")
   print(classes)
   local function has_image_extensions(filename)
      local ext = string.lower(path.extension(filename))

      -- compare with list of image extensions
      local img_extensions = {'.jpeg', '.jpg', '.png', '.ppm', '.pgm'}
      for i = 1, #img_extensions do
         if ext == img_extensions[i] then
            return true
         end
      end
      return false
   end

   -- initialize data structures:
   trainData = {
      data = torch.FloatTensor(trsize, opt.channels, opt.imHeight, opt.imWidth),
      labels = torch.FloatTensor(trsize, opt.imHeight, opt.imWidth),
      preverror = 1e10, -- a really huge number
      size = function() return trsize end
   }

   testData = {
      data = torch.FloatTensor(tesize, opt.channels, opt.imHeight, opt.imWidth),
      labels = torch.FloatTensor(tesize, opt.imHeight, opt.imWidth),
      preverror = 1e10, -- a really huge number
      size = function() return tesize end
   }

   print('==> Loading training files')

   local dpathRoot = opt.datapath .. '/leftImg8bit/train/'

   assert(paths.dirp(dpathRoot), 'No training folder found at: ' .. opt.datapath)
   --load training images and labels:
   local c = 1
   for dir in paths.iterdirs(dpathRoot) do
      local dpath = dpathRoot .. dir .. '/'
      for file in paths.iterfiles(dpath) do

         -- process each image
         if has_image_extensions(file) and c <= trsize then
            local imgPath = path.join(dpath, file)

            --load training images:
            local dataTemp = image.load(imgPath)
            trainData.data[c] = image.scale(dataTemp,opt.imWidth, opt.imHeight)

            -- Load training labels:
            -- Load labels with same filename as input image.
            imgPath = string.gsub(imgPath, "leftImg8bit", "gtFine")
            imgPath = string.gsub(imgPath, ".png", "_labelIds.png")


            -- label image data are resized to be [1,nClasses] in [0 255] scale:
            local labelIn = image.load(imgPath, 1, 'byte')
            local labelFile = image.scale(labelIn, opt.imWidth, opt.imHeight, 'simple'):float()

            labelFile:apply(function(x) return classMap[x][1] end)

            -- Syntax: histc(data, bins, min, max)
            histClasses = histClasses + torch.histc(labelFile, #classes, 1, #classes)

            -- convert to int and write to data structure:
            trainData.labels[c] = labelFile

            c = c + 1
            if c % 20 == 0 then
               xlua.progress(c, trsize)
            end
            collectgarbage()
         end
      end
   end
   print('')

   print('==> Loading testing files')
   dpathRoot = opt.datapath .. '/leftImg8bit/val/'

   assert(paths.dirp(dpathRoot), 'No testing folder found at: ' .. opt.datapath)
   -- load test images and labels:
   local c = 1
   for dir in paths.iterdirs(dpathRoot) do
      local dpath = dpathRoot .. dir .. '/'
      for file in paths.iterfiles(dpath) do

         -- process each image
         if has_image_extensions(file) and c <= tesize then
            local imgPath = path.join(dpath, file)

            --load training images:
            local dataTemp = image.load(imgPath)
            testData.data[c] = image.scale(dataTemp, opt.imWidth, opt.imHeight)

            -- Load validation labels:
            -- Load labels with same filename as input image.
            imgPath = string.gsub(imgPath, "leftImg8bit", "gtFine")
            imgPath = string.gsub(imgPath, ".png", "_labelIds.png")


            -- load test labels:
            -- label image data are resized to be [1,nClasses] in in [0 255] scale:
            local labelIn = image.load(imgPath, 1, 'byte')
            local labelFile = image.scale(labelIn, opt.imWidth, opt.imHeight, 'simple'):float()

            labelFile:apply(function(x) return classMap[x][1] end)

            -- convert to int and write to data structure:
            testData.labels[c] = labelFile

            c = c + 1
            if c % 20 == 0 then
               xlua.progress(c, tesize)
            end
            collectgarbage()
         end
      end
   end
end

if opt.cachepath ~= "none" and not loadedFromCache then
   print('\27[32m'..'==> Saving data to cache: \27[0m' .. cityscapeCachePath)
   local dataCache = {
      trainData = trainData,
      testData = testData,
      histClasses = histClasses
   }
   torch.save(cityscapeCachePath, dataCache)
   dataCache = nil
   collectgarbage()
end

----------------------------------------------------------------------
print '==> Normalizing data'

-- It's always good practice to verify that data is properly
-- normalized.
local trainMean = torch.zeros(3, trainData.data:size(3), trainData.data:size(4))
for i = 1, opt.channels do
   trainMean[i] = trainData.data[{{}, i, {}, {}}]:mean()
end

for i = 1, trainData.data:size(1) do
   trainData.data[i]:add(-trainMean)
end
for i = 1, testData.data:size(1) do
   testData.data[i]:add(-trainMean)
end
torch.save(paths.concat(opt.cachepath, dirName, 'stat.t7'), trainMean)

----------------------------------------------------------------------

local classes_td = {[1] = 'classes,targets\n'}
for _,cat in pairs(classes) do
   table.insert(classes_td, cat .. ',1\n')
end

local file = io.open(paths.concat(opt.save, 'categories.txt'), 'w')
file:write(table.concat(classes_td))
file:close()

-- Exports
opt.dataClasses = classes
opt.dataconClasses  = conClasses
opt.datahistClasses = histClasses

return {
   trainData = trainData,
   testData = testData,
}
