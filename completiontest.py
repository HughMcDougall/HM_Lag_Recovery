import time

donetime = time.localtime()
hr = donetime.tm_hour
mi = donetime.tm_min
mo = donetime.tm_mon
da = donetime.tm_mday

outstr = "Program finishes at %i:%i on %i / %i" %(hr, mi, mo, da)

print(outstr)
