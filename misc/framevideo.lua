local frame = {}
torch.setdefaulttensortype('torch.FloatTensor')

local pf = function(...) print(string.format(...)) end
local Cb = sys.COLORS.blue
local Cn = sys.COLORS.none

--[[
   opt fields
      input       filename
      batch       batch size
      loglevel    if w

   source fields
      w           image width
      h           image height

--]]
function frame:init(input, source)

   local vd = require('libvideo_decoder')
   local status = false
   local width = source.w
   status, source.h, source.w, source.length, source.fps = vd.init(input);
   source.origw = source.w
   source.origh = source.h
   if not status then
      error("No video")
   else
      pf(Cb..'video statistics: %s fps, %dx%d (%s frames)'..Cn,
         (source.fps and tostring(source.fps) or 'unknown'),
         source.h,
         source.w,
         (source.length and tostring(source.length) or 'unknown'))
   end

   framefunc = vd.frame_rgb
   local img_tmp = torch.FloatTensor(1, 3, source.h, source.w)

   frame.nFrames = function()
      return source.length
   end

   -- set frame forward function
   frame.forward = function(img)
      local n = 1
      for i=1,1 do
         if not framefunc(img_tmp[i]) then
            if i == 1 then
               return false
            end
            n = i-1
            break
         end
      end
      if n == 1 then
         img = img_tmp
      else
         img = img_tmp:narrow(1,1,n)
      end
      return img
   end

end

return frame
