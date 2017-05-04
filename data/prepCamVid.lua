----------------------------------------------------------------------
-- Sample CamVid videos to get train/test/validation images
-- and corresponding labels
--
-- Abhishek Chaurasia,
-- November, 2016
----------------------------------------------------------------------
require 'image'
require 'xlua'
----------------------------------------------------------------------
local N = 701
local trsize = 367
local tesize = 233
local vasize = 101

local dirRoot  = opt.datapath
local imHeight = 720
local imWidth  = 960

local red      = '\27[31m'
local green    = '\27[32m'
local resetCol = '\27[0m'
----------------------------------------------------------------------

print '\n\27[31m\27[4mPreparing CamVid dataset for data loader\27[0m'
--------------------------------------------------------------------------------
-- Function to check if the given file is a valid video
local function validVideo(filename)
   local ext = string.lower(paths.extname(filename))

   local videoExt = {'avi', 'mp4', 'mxf'}
   for i = 1, #videoExt do
      if ext == videoExt[i] then
         return true
      end
   end
   print(red .. ext .. " extension is NOT supported!!!" .. resetCol)
   return false
end

-- Function to read txt file and return image and ground truth path
function getPath(filepath)
   print("Filenames and their role found in: " .. filepath)
   local file = io.open(filepath, 'r')
   local role = {}
   local fileList = {}
   local fline = file:read()
   while fline ~= nil do
      local col1, col2 = fline:match("([^,]+),([^,]+)")
      table.insert(role, col1)
      table.insert(fileList, col2)
      fline = file:read()
   end
   return role, fileList
end

--------------------------------------------------------------------------------
-- Initialize class Frame which can be used to read videos/camera
local frame = assert(dofile('misc/framevideo.lua'))

local source = {}
-- switch input sources
source.w = 720
source.h = 960
source.fps = 30

local labelPrefixTable = {'0001TP_0', 'Seq05VD_f', '0006R0_f', '0016E5_'}
local labelStart       = {6690,   0, 930, 390}
local labelOffset      = {  30,   1, 931, 391}
local maxSampleFrames  = { 124, 171, 101, 305}

--------------------------------------------------------------------------------
-- Initialize data structures:
--------------------------------------------------------------------------------
local trainData = {
      data      = torch.FloatTensor(trsize, 3, imHeight, imWidth),
      labels    = torch.FloatTensor(trsize, imHeight, imWidth),
      size      = function() return trsize end
}
local testData  = {
      data      = torch.FloatTensor(tesize, 3, imHeight, imWidth),
      labels    = torch.FloatTensor(tesize, imHeight, imWidth),
      size      = function() return tesize end
}
local valData   = {
      data      = torch.FloatTensor(vasize, 3, imHeight, imWidth),
      labels    = torch.FloatTensor(vasize, imHeight, imWidth),
      size      = function() return vasize end
}

local trc = 1
local tec = 1
local vac = 1
local totalCount = 1          -- Overall counter for whole dataset
--------------------------------------------------------------------------------
-- forward img and gather label and input frame for that label
-- Input : directory path containing videos, directory number
-- Output: tensors storing labels and their images
--------------------------------------------------------------------------------
local function forwardSeq(input, dirN, role, fileList)
   -- source height and width gets updated by __init based on the input video
   frame:init(input, source)
   local nFrames = frame.nFrames()        -- # of total frames in the video


   local img = frame.forward(img)
   local n = - labelOffset[dirN]          -- Counter for frame index
   local count = 1                        -- Counter for how many frames have been added to one sequence
   local labelPath
   local labelPrefix = labelPrefixTable[dirN]
   local label = torch.zeros(imHeight, imWidth)
   local buggyLabel = dirRoot .. dirN .. '/label/Seq05VD_f02610_L.png'

   while count <= maxSampleFrames[dirN] do
      xlua.progress(count, maxSampleFrames[dirN])
      --------------------------------------------------------------------------
      -- Save representation alongwith corresponding label, only if label exists
      --------------------------------------------------------------------------
      local labelIdx = n + labelStart[dirN]
      labelPath = dirRoot .. dirN .. '/label/' .. labelPrefix .. string.format('%05d_L.png', labelIdx)
      if paths.filep(labelPath) then
         count = count + 1

         -- Load current label
         local labelRGB = image.load(labelPath, 3, 'byte')/64
         -- Convert RGB into grayscale
         label = labelRGB[1] * 16 + labelRGB[2] * 4 + labelRGB[3]

         if labelPath == buggyLabel then
            local mask = label:eq(21)
            label = label - 21 * mask
         end

         local verifyLabel = dirRoot .. dirN .. '/label/' .. fileList[totalCount]
         if role[totalCount] == 'train' and labelPath == verifyLabel then
            trainData.data[trc] = img[1]:clone()
            trainData.labels[trc] = label:clone()
            totalCount = totalCount + 1
            trc = trc + 1
         elseif role[totalCount] == 'test' and labelPath == verifyLabel then
            testData.data[tec] = img[1]:clone()
            testData.labels[tec] = label:clone()
            totalCount = totalCount + 1
            tec = tec + 1
         elseif role[totalCount] == 'val' and labelPath == verifyLabel then
            valData.data[vac] = img[1]:clone()
            valData.labels[vac] = label:clone()
            totalCount = totalCount + 1
            vac = vac + 1
         else
            print('\27[31mLooking for \27[0m' .. verifyLabel .. '\27[31m in \27[0m' .. role[totalCount] .. 'set')
            error('But something went wrong with:' .. labelPath)
         end
      end

      img = frame.forward(img)
      n = n + 1
      collectgarbage()
   end
   collectgarbage()
end

-----------------------------------------------------------------------------------
-- Main section
-----------------------------------------------------------------------------------
local loadedFromCache = false
local cacheDir = opt.cachepath
local camvidCachePath = paths.concat(cacheDir, 'trainTestVal.t7')

if not paths.dirp(cacheDir) then paths.mkdir(cacheDir) end

--------------------------------------------------------------------------------
-- Acquire image and ground truth paths for training and testing set
assert(paths.dirp(dirRoot), 'No folder found at: ' .. dirRoot)

local role, fileList = getPath('./misc/dataDistributionCV.txt')
for dirN = 1, 4 do
   local dirPath = dirRoot .. dirN .. '/input/'

   for file in paths.iterfiles(dirPath) do
      print(red .. "\nGetting input images and labels for: " .. resetCol .. file)
      -- process each image
      if validVideo(file) then
         local vidPath = dirPath .. file
         forwardSeq(vidPath, dirN, role, fileList)

         print(green .. "Loaded input images and labels!!!" .. resetCol)
      end
   end
end
print()
print(string.format("%s# of training   data :%s %d", green, resetCol, trainData:size()))
print(string.format("%s# of testing    data :%s %d", green, resetCol, testData:size()))
print(string.format("%s# of validation data :%s %d", green, resetCol, valData:size()))
-----------------------------------------------------------------------------------
print('Saving compatible data for data-loader: ' .. camvidCachePath)
local dataCache = {
   trainData = trainData,
   testData  = testData,
   valData   = valData,
}
torch.save(camvidCachePath, dataCache)
collectgarbage()
