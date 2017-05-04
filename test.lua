----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data.
--
-- Written by  : Abhishek Chaurasia, Eugenio Culurcielo
-- Dated       : August 2016
----------------------------------------------------------------------

require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
-- Logger:
errorLogger = optim.Logger(paths.concat(opt.save, 'error.log'))

print '==> defining test procedure'
local testConf

if opt.dataconClasses then
   testConf = optim.ConfusionMatrix(opt.dataconClasses)
else
   testConf = optim.ConfusionMatrix(opt.dataClasses)
end

local best_IoU   = {0, 0}      -- Value, epoch #
local best_iIoU  = {0, 0}
local best_GAcc  = {0, 0}
local best_error = {10e4, 0}
local metricName = {'testError', 'IoU', 'iIoU', 'GAcc'}
local metricFlag = {0, 0, 0, 0}    -- Flag to save values in best-number.txt

-- Batch test:
local x = torch.Tensor(opt.batchSize, opt.channels, opt.imHeight, opt.imWidth)
local yt = torch.Tensor(opt.batchSize, opt.imHeight, opt.imWidth)
x = x:cuda()
yt = yt:cuda()

local function saveConfMatrix(filename, teConfMat, trConfMat)
   local file = io.open(filename, 'w')
   file:write("--------------------------------------------------------------------------------\n")
   if opt.saveTrainConf then
      file:write("Training:\n")
      file:write("================================================================================\n")
      file:write(tostring(trConfMat))
      file:write("\n--------------------------------------------------------------------------------\n")
   end
   file:write("Testing:\n")
   file:write("================================================================================\n")
   file:write(tostring(teConfMat))
   file:write("\n--------------------------------------------------------------------------------")
   file:close()
end

local function gatherBestMetric(currentVal, currentEpoch, metric, metricMode)
   local metricIndex = 0
   if metricMode == 1 then
      if currentVal < metric[1] then
         metricIndex = 1             -- Based on metric name
         metric[1] = currentVal
         metric[2] = currentEpoch
      end
   else
      if currentVal > metric[1] then
         metricIndex = 1             -- Based on metric name
         metric[1] = currentVal
         metric[2] = currentEpoch
      end
   end
   return metricIndex
end

-- test function
function test(testData, classes, epoch, trainConf, model, loss )
   ----------------------------------------------------------------------
   -- local vars
   local time = sys.clock()
   -- total loss error
   local err = 0
   local totalerr = 0

   -- This matrix records the current confusion across classes

   model:evaluate()

   -- test over test data
   for t = 1, testData:size(), opt.batchSize do
      -- disp progress
      xlua.progress(t, testData:size())

      -- batch fits?
      if (t + opt.batchSize - 1) > testData:size() then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         x[idx]:copy(testData.data[i])
         yt[idx]:copy(testData.labels[i])
         idx = idx + 1
      end

      -- test sample
      local y = model:forward(x)

      err = loss:forward(y,yt)
      local y = y:transpose(2, 4):transpose(2, 3)
      y = y:reshape(y:numel()/y:size(4), #classes):sub(1, -1, 2, #opt.dataClasses)
      local _, predictions = y:max(2)
      predictions = predictions:view(-1)
      local k = yt:view(-1)
      if opt.dataconClasses then k = k - 1 end
      testConf:batchAdd(predictions, k)

      totalerr = totalerr + err
      collectgarbage()
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print(string.format('==> Time to test 1 sample = %2.2f, %s', (time*1000), 'ms'))

   -- print average error in train dataset
   totalerr = totalerr / (testData:size()*(#opt.dataconClasses) / opt.batchSize)
   print(string.format('\nTrain Error: %1.4f', trainError))
   print(string.format('Test  Error: %1.4f', totalerr))
   -- save/log current net
   errorLogger:add{['Training error'] = trainError,
                   ['Testing error'] = totalerr}
   if opt.plot then
      errorLogger:style{['Training error'] = '-',
      ['Testing error'] = '-'}
      errorLogger:display(opt.showPlot)
      errorLogger:plot()
   end

   -- TODO Get rid of this save. Right now when not saved, metric dont get any values
   -- Save the last Confusion Matrix
   local filename = paths.concat(opt.save, 'lastConfusionMatrix.txt')
   saveConfMatrix(filename, testConf, trainConf)

   filename = paths.concat(opt.save, 'model-last.net')
   torch.save(filename, model:clearState():get(1))

   -- Calculate IoU, iIoU, Global Accuracy
   local IoU = testConf.averageValid * 100
   local iIoU = torch.sum(testConf.unionvalids)/#opt.dataconClasses * 100
   local GAcc = testConf.totalValid * 100
   print(string.format('\nIoU: %2.2f%% | iIoU : %2.2f%% | AvgAccuracy: %2.2f%%', IoU, iIoU, GAcc))

   -- See if the latest value is better
   metricFlag[1] = gatherBestMetric(totalerr, epoch, best_error, 1)
   metricFlag[2] = gatherBestMetric(IoU, epoch, best_IoU, 2)
   metricFlag[3] = gatherBestMetric(iIoU, epoch, best_iIoU, 3)
   metricFlag[4] = gatherBestMetric(GAcc, epoch, best_GAcc, 4)

   -- Update model and confusion matrix file if better value is found
   local updateFile = 0
   local dumFlag = 0
   for i = 1, 4 do
      if metricFlag[i] == 1 then
         if dumFlag == 0 then
            io.write('\27[32mBetter ' .. metricName[i])
         else
            io.write(', ' .. metricName[i])
         end
         filename = paths.concat(opt.save, 'model-' .. metricName[i] .. '.net')
         torch.save(filename, model:clearState():get(1))

         filename = paths.concat(opt.save, 'confusionMatrix-' .. metricName[i] .. '.txt')
         saveConfMatrix(filename, testConf, trainConf)
         metricFlag[i] = 0
         updateFile = 1
         dumFlag = 1
      end
   end
   if dumFlag == 1 then
      io.write(' found\27[31m!!!\27[0m\n')
   end

   -- Update best numbers
   if updateFile == 1 then
      filename = paths.concat(opt.save, 'best-number.txt')
      local file = io.open(filename, 'w')
      file:write("----------------------------------------\n")
      file:write(string.format('Best test error: %2.2f, in epoch: %d', best_error[1], best_error[2]))
      file:write("\n----------------------------------------\n")
      file:write(string.format('Best        IoU: %2.2f, in epoch: %d', best_IoU[1], best_IoU[2]))
      file:write("\n----------------------------------------\n")
      file:write(string.format('Best       iIoU: %2.2f, in epoch: %d', best_iIoU[1], best_iIoU[2]))
      file:write("\n----------------------------------------\n")
      file:write(string.format('Best   accuracy: %2.2f, in epoch: %d', best_GAcc[1], best_GAcc[2]))
      file:write("\n----------------------------------------\n")
      file:close()
      metricIndex = 0
   end

   if opt.saveAll then
      filename = paths.concat(opt.save, 'all/model-' .. epoch .. '.net')
      torch.save(filename, model:clearState():get(1))

      filename = paths.concat(opt.save, 'all/confusionMatrix-' .. epoch .. '.txt')
      saveConfMatrix(filename, testConf, trainConf)
   end
   --resetting confusionMatrix
   trainConf:zero()
   testConf:zero()

   collectgarbage()
end

-- Export:
return test
