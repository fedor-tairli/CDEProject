from time import sleep
import os

from Machine import Pin


#LSB
p0 = (0,Pin.OUT)
p1 = (2,Pin.OUT)
p2 = (4,Pin.OUT)
p3 = (5,Pin.OUT)
p4 = (16,Pin.OUT)
p5 = (17,Pin.OUT)
p6 = (18,Pin.OUT)
p7 = (19,Pin.OUT)
#MSB

pinput = Pin(13,Pin.IN)

def shifty(outbits): # Set Bits from 0 to 7 using bitshifting and binary comparison
    p0.value((outbits>>0)&0b00000001) # Shifts the bits to the right by 0 and compares the last bit
    p1.value((outbits>>1)&0b00000001)
    p2.value((outbits>>2)&0b00000001)
    p3.value((outbits>>3)&0b00000001)
    p4.value((outbits>>4)&0b00000001)
    p5.value((outbits>>5)&0b00000001)
    p6.value((outbits>>6)&0b00000001)
    p7.value((outbits>>7)&0b00000001) # Shifts the bits to the right by 7 and compares the last bit which is now the original first bit


# Bitshifting above makes it more readable by making the value of current DAC output an integer and not a list of bits
# pin.value allows setting value with bool input (not using explicit logic)

while True:
    DacVal = 0
    MSBVal = 0b10000000 # =255

    for LoopVal in range(8):
        DacVal += MSBVal # the type should be conserved

        shifty(DacVal)
        sleep(0.001) # Sleep doesnt work properly for values under 1ms
        # Sleeping less than 1ms doesnt allow time for input Pin's internal capacitors to charge and discharge
        # print can be made here instead of sleep (print is slow so its longer than 1ms)

        if pinput.value():
            DacVal -= MSBVal # too high, go back

        MSBVal //= 2 # Shift the MSB to the right by 1, save inplace

        # sleep(0.5) # Pause after each approximation
