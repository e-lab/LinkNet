----------------------------------------------------------------------
-- Train a network for semantic segmentation
--
-- Abhishek Chaurasia, Eugenio Culurciello
----------------------------------------------------------------------

require 'pl'
require 'nn'

io.write('\27[0;0f\27[0J')
----------------------------------------------------------------------
-- Local repo files
local opts = require 'opts'
opt = opts.parse(arg)

-- nb of threads and fixed seed (for repeatable experiments)
-- torch.setnumthreads(opt.threads)
torch.manualSeed(12)
torch.setdefaulttensortype('torch.FloatTensor')

-- print('==> switching to CUDA')
require 'cudnn'
require 'cunn'
cutorch.setDevice(opt.devid)
print('\n\27[32mModels will be saved in \27[0m\27[4m' .. opt.save .. '\27[0m')
os.execute('mkdir -p ' .. opt.save)
if opt.saveAll then
   os.execute('mkdir -p ' .. opt.save .. '/all')
end

----------------------------------------------------------------------
local data, chunks, ft
if opt.dataset == 'cv' then
   data  = require 'data/loadCamVid'
elseif opt.dataset == 'cs' then
   data = require 'data/loadCityscapes'
else
   error ("Dataset loader not found. (Available options are: cv/cs")
end

local filename = paths.concat(opt.save,'opt.txt')
local file = io.open(filename, 'w')
for i,v in pairs(opt) do
    file:write(tostring(i)..' : '..tostring(v)..'\n')
end
file:close()

----------------------------------------------------------------------
local epoch = 1

t = paths.dofile(opt.model)

local train = require 'train'
local test  = require 'test'

print('\27[31m\27[4m\nTraining and testing started\27[0m')
print('[batchSize = ' ..  opt.batchSize .. ']')
while epoch < opt.maxepoch do
   print(string.format('\27[31m\27[4m\nEpoch # %d\27[0m', epoch))
   print('==> Training:')
   local trainConf, model, loss = train(data.trainData, opt.dataClasses, epoch)
   print('==> Testing:')
   test(data.testData, opt.dataClasses, epoch, trainConf, model, loss )
   trainConf = nil
   collectgarbage()
   epoch = epoch + 1
end
