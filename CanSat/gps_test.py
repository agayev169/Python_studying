import serial

gps_port = '/dev/ttyUSB0'
baud_rate = 9600

ser = serial.Serial(gps_port, baud_rate)

# while True:
msg = ser.readline()
print(msg)