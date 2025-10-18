from queue import Queue
import bluetooth as bt
import socket
import sys
import subprocess
import json
from rpi_config import *

class AndroidInterface:
    """
    Interface between Raspberry Pi and Android device over Bluetooth.
    """
    def __init__(self, RPiMain):
        self.RPiMain = RPiMain
        self.host = RPI_IP
        self.uuid = BT_UUID
        self.msg_queue = Queue()
        self.socket = None
        self.client_socket = None
        self.client_info = None

    def connect(self):
        # Grant permission for Bluetooth access
        subprocess.run("sudo chmod o+rw /var/run/sdp", shell=True) 

        # Establish and bind socket
        self.socket = bt.BluetoothSocket(bt.RFCOMM)
        print("[Android] BT socket established successfully.")
    
        try:
            self.port = self.socket.getsockname()[1]
            print("[Android] Waiting for connection on RFCOMM channel", self.port)
            self.socket.bind((self.host, bt.PORT_ANY))
            print("[Android] BT socket binded successfully.")
            
            # Turning advertisable
            subprocess.run("sudo hciconfig hci0 piscan", shell=True)
            self.socket.listen(128)
            
            # Advertise Bluetooth service
            bt.advertise_service(
                self.socket, 
                "Group27-Server", 
                service_id=self.uuid, 
                service_classes=[self.uuid, bt.SERIAL_PORT_CLASS], 
                profiles=[bt.SERIAL_PORT_PROFILE])

        except socket.error as e:
            print("[Android] ERROR: Android socket binding failed -", str(e))
            sys.exit()
            
        print("[Android] Waiting for Android connection...")
        try:
            self.client_socket, self.client_info = self.socket.accept()
            print("[Android] Accepted connection from", self.client_info)    
        except socket.error as e:
            print("[Android] ERROR: connection failed -", str(e))

    def disconnect(self):
        try:
            if self.socket:
                self.socket.close()
            print("[Android] Disconnected from Android successfully.")
        except Exception as e:
            print("[Android] ERROR: Failed to disconnect from Android -", str(e))
            
    def reconnect(self):
        self.disconnect()
        self.connect()

    def listen(self):
        while True:
            try:
                message = self.client_socket.recv(BT_BUFFER_SIZE)
                if not message:
                    print("[Android] Android disconnected remotely. Reconnecting...")
                    self.reconnect()
                    continue

                decodedMsg = message.decode("utf-8")
                if len(decodedMsg) <= 1:
                    continue

                print("[Android] Read from Android:", decodedMsg[:MSG_LOG_MAX_SIZE])
                parsedMsg = json.loads(decodedMsg)
                msg_type = parsedMsg["type"]

                # Route messages to the appropriate destination
                if msg_type == 'NAVIGATION':
                    self.RPiMain.STM.msg_queue.put(message) 
                elif msg_type == 'START_TASK' or msg_type == 'FASTEST_PATH':
                    self.RPiMain.PC.msg_queue.put(message)
            except (socket.error, IOError, Exception, ConnectionResetError) as e:
                print("[Android] ERROR:", str(e))
                self.reconnect()
                

    def send(self):
        # Continuously send messages to Android
        while True: 
            message = self.msg_queue.get()
            exception = True
            while exception: 
                try:
                    self.client_socket.sendall(message)
                    print("[Android] Write to Android: " + message.decode("utf-8")[:MSG_LOG_MAX_SIZE])
                except Exception as e:
                    print("[Android] ERROR: Failed to write to Android -", str(e))
                    self.reconnect()
                else:
                    exception = False  # done sending, get next message

