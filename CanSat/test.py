import matplotlib.pyplot as plt
import drawnow
import serial
import time
import csv

plt.ion()

serial_port = '/dev/ttyUSB0'
baud_rate = 9600
# file_name = "CanSat2018_AstroUFAZ.csv"
# header_row = ['TEAM ID', 'TIME', 'ALT_SENSOR', 'OUTSIDE_TEMP',
			  # 'INSIDE_TEMP', 'VOLTAGE', 'FSW_STATE', 'BONUS']


ser = serial.Serial(serial_port, baud_rate)

# myFile = open(file_name, 'wb')
# writer = csv.writer(myFile)
# writer.writerow(header_row)

pressure, temp, alt = [], [], []

def plot():
	plt.subplot(1, 3, 1)
	plt.xlabel("Pressure")
	plt.ylabel("Time")
	plt.plot(pressure, 'ro-')
	plt.subplot(1, 3, 2)
	plt.xlabel("Temperature")
	plt.ylabel("Time")
	plt.plot(temp, 'bo-')
	plt.subplot(1, 3, 3)
	plt.xlabel("Altitude")
	plt.ylabel("Time")
	plt.plot(alt, 'go-')

while True:
	msg = ser.readline()
	print(msg)
	try:
		data = msg.split(',')
		# print(data)
		pressure.append(float(data[0]))
		temp.append(float(data[1]))
		alt.append(float(data[2]))
	except:
		print("Shit!")

	# drawnow.drawnow(plot)
	# time.sleep(0.8)