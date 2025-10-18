from queue import Queue 
import time 
import serial 
import time

STM_BAUDRATE = 115200
STM_ACK_MSG = "A"
STM_COMMAND_DELAY = 0.5
STM_GYRO_RESET_COMMAND = "RESET"
STM_GYRO_RESET_DELAY = 4 # time to wait for gyro reset
STM_SERIAL_PORT = "/dev/ttyACM0"  # Adjust as necessary

#### TO ADJUST BASED ON TESTING
#STM_NAV_COMMAND = ["START", "CORRE","SNAPP",
#                  "LF060","RF060","RF060","LF060",
#                  "START","CORRE", "SNAPP", "RF089", 
#                  "SB010","WL999","LF179", "WL999", "SF005", "LF087", "HOMEE", 
#                  "LF090", "DUMEE", "RF090","START","ENYAO"]

#STM_NAV_COMMAND = ["START", "CORRE","SNAPP",
#                  "RF058","LF058","LF058","RF058",
#                  "START","CORRE", "SNAPP", "LF089", 
#                  "SB010","WR999","RF179", "WR999", "SF005", "RF087", "HOMEE", 
#                  "RF090", "DUMEE", "LF090","START","ENYAO"]

#STM_NAV_COMMAND = ["START", "CORRE","SNAPP",
#                  "RF058","LF058","LF058","RF058",
#                  "START","CORRE", "SNAPP", "RF089", 
#                  "SB010","WL999","LF179", "WL999", "SF005", "LF087", "HOMEE", 
#                  "LF090", "DUMEE", "RF090","START","ENYAO"]

#STM_NAV_COMMAND = ["CORRE"]
#STM_NAV_COMMAND = ["WL999"]
STM_NAV_COMMAND = ["WR999"]
##############################################

class STMTestInterface:
    """
    Test interface to send STM commands from RPI to STM without PC connection
    """
    def __init__(self):
        self.serial = None
        self.msg_queue = Queue()
        self.move_counter = 0

    def connect(self):
        # Connect to STM using available serial ports
        try:
            self.serial = serial.Serial(STM_SERIAL_PORT, STM_BAUDRATE, write_timeout=0)
            print("[STM] Connected to STM successfully.")
            self.clean_buffers()
        except Exception as e:
            print("[STM] ERROR: Failed to connect to STM -", str(e))
    
    def reconnect(self): 
        # Reconnect to STM by closing the current connection and establishing a new one
        if self.serial is not None and self.serial.is_open:
            self.serial.close()
        self.connect()
    
    def clean_buffers(self):
        # Reset input and output buffers of the serial connection
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

    def listen(self):
        # Listen for messages from STM
        message = None
        while True:
            try:
                message = self.serial.read().decode("utf-8")
                print("[STM] Read from STM:", message)
                
                if len(message) < 1:
                    continue
                else: 
                    break

            except Exception as e:
                message = str(e)
                break

        return message

    def send(self):
        # Test code without PC
        message = {
            "type": "NAVIGATION",
            "data": {
            "commands":  STM_NAV_COMMAND, 
            "path": [],
            }
        }
        message_type = 'NAVIGATION'
        # end of test code

        if message_type == "NAVIGATION":
            # Convert/adjust turn or obstacle routing commands
            commands: list[str] = message["data"]["commands"]
            
            # Real code
            for idx, command in enumerate(commands):
                
                print("[RPI] Writing to STM:", command)
                self.write_to_stm(command)
        else:
            print("[STM] WARNING: Rejecting message with unknown type [%s] for STM" % message_type)

    def write_to_stm(self, command):
        # Write a command to STM, handling exceptions and reconnecting if necessary
        self.clean_buffers()
        exception = True
        while exception:
            try:
                print("[STM] Sending command", command)
                encoded_string = command.encode()
                byte_array = bytearray(encoded_string)
                self.serial.write(byte_array)
            except Exception as e:
                print("[STM] ERROR: Failed to write to STM -", str(e)) 
                exception = True
                self.reconnect() 
            else:
                exception = False
                if command == STM_GYRO_RESET_COMMAND:
                    print("[STM] Waiting %ss for reset" % STM_GYRO_RESET_DELAY)
                    time.sleep(STM_GYRO_RESET_DELAY)
                else:
                    print("[STM] Waiting for ACK")
                    self.wait_for_ack()
                    time.sleep(STM_COMMAND_DELAY)


    def wait_for_ack(self):
        # Wait for ACK message from STM
        message = self.listen()
        print(message)
        if message  == STM_ACK_MSG:
            print("[STM] Received ACK from STM") 
        else:
            print("[STM] ERROR: Unexpected message from STM -", message)
            self.reconnect() 
        
    # Run test code without PC
if __name__ == "__main__":
    stm = STMTestInterface()
    stm.connect()
    stm.send()
