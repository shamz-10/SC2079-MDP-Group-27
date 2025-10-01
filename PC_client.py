import os
import socket
import threading
from queue import Queue
import json
import time
import shutil
import base64

from image_recognition import model_inference
from image_recognition.stitch_images import stitching_images
from Algorithm import algo_backwards as algo # Using updated algo_backwards

# Configuration
TASK_2 = False  #TODO: Change to False for task 1, True for task 2

# Constants
RPI_IP = "192.168.27.27"  # Replace with the Raspberry Pi's IP address
PC_PORT = 8888            # Replace with the port used by the PC server
PC_BUFFER_SIZE = 1024
# NUM_OF_RETRIES = 2


class MovementTraceNavigator:
    """
    Adapter that:
    1) runs algo.task1(obstacles_file) to generate movement_trace.json
    2) loads it and splits into segments between IMAGE_REC markers
    3) provides old-school methods: generate_path, get_command_to_next_obstacle, get_obstacle_id
    """
    def __init__(self):
        self.segments = []       # list of {"commands": [...], "path": [[r,c], ...]}
        self.segment_idx = 0
        self.obs_id = 1
        self.trace = None

    def _load_trace(self, path="movement_trace.json"):
        with open(path, "r") as f:
            self.trace = json.load(f)

    def _split_into_segments(self):
        """
        Split trace.data.commands/path into chunks up to each IMAGE_REC token.
        Each movement command advances one path index; IMAGE_REC does not.
        """
        cmds = self.trace["data"]["commands"]
        path = self.trace["data"]["path"]
        segments = []

        i_cmd = 0
        i_path = 0  # index into path states; starts at 0

        while i_cmd < len(cmds):
            seg_cmds = []
            start_path_idx = i_path

            # accumulate until IMAGE_REC or end
            while i_cmd < len(cmds):
                cmd = cmds[i_cmd]
                seg_cmds.append(cmd)
                # print("cmd appended", cmd)
                i_cmd += 1

                if i_cmd < len(cmds) and cmd == "IMAGE":
                    break

                # Advance along path according to command semantics
                # Straight moves are encoded like SB050, SF140 (distance in mm/10)
                # Rotations like RB080, RF090 do not advance grid position
                steps = 0
                if cmd and len(cmd) >= 3 and cmd[0] == 'S':
                    try:
                        steps = max(2, int(cmd[2:]) // 10)
                    except Exception:
                        steps = 2
                elif (cmd[0] == 'L' or cmd[0]== 'R'):
                    steps=1 #TODO: Change according to 1x3 turn
                else:
                    steps = 0
                
                # print("print cmd, steps:",cmd, steps)

                # Move i_path forward but never past the final state
                if steps > 0:
                    i_path = min(i_path + steps, len(path) - 1)

            # segment path is states [start..i_path] inclusive
            seg_path = path[start_path_idx:i_path+1] if i_path >= start_path_idx else [path[start_path_idx]]

            if seg_cmds or seg_path:
                segments.append({"commands": seg_cmds, "path": seg_path})

            # consume IMAGE_REC boundary (don’t move along path)
            # if i_cmd < len(cmds) and cmds[i_cmd] == "IMAGE":
            #     i_cmd += 1
        print("Segments:", segments)

        self.segments = segments
        self.segment_idx = 0
        self.obs_id = 0

    def generate_path(self, message):
        """
        Runs algo to regenerate movement_trace.json and prepares segments.
        Expects message like: {"type":"START_TASK","data":{"obstacles_file":"obstacles.json"}}
        """
        # obstacles_path = message.get("data", {}).get("obstacles_file", "obstacles.json")
        # if not os.path.exists(obstacles_path):
        #     raise FileNotFoundError(f"'{obstacles_path}' not found.")

        # This will compute and write movement_trace.json (and show animation)
        algo.task1(message)

        # Load and prepare segments
        self._load_trace("Algorithm/movement_trace.json")
        self._split_into_segments()

    def get_command_to_next_obstacle(self):
        """
        Returns one segment (commands + path) up to the next IMAGE_REC boundary,
        in the same NAVIGATION packet shape your RPi expects.
        """
        if self.segment_idx >= len(self.segments):
            return {"type": "END"}  # or handle however your protocol expects

        seg = self.segments[self.segment_idx]
        self.segment_idx += 1
        # obs_id increments after every boundary segment we send
        out = {
            "type": "NAVIGATION",
            "data": {
                "commands": seg["commands"],
                "path": seg["path"],
            }
        }

        print("get_command_to_next_obstacle",out)
        return out

    def get_obstacle_id(self):
        # Return current (1-based) or 0-based as needed (kept string to match your original usage)
        # We bump obs_id when we actually delivered a movement segment
        current = self.obs_id
        self.obs_id += 1
        return current

    def has_task_ended(self):
        print("segment_dx, segments", self.segment_idx, len(self.segments)+1)
        return self.segment_idx >= len(self.segments)


class PCClient:
    def __init__(self):
        # Initialize PCClient with connection details
        self.host = RPI_IP
        self.port = PC_PORT
        self.client_socket = None
        self.msg_queue = Queue()
        self.send_message = False
        self.image_record = []
        self.task_2 = TASK_2
        self.obs_order_count = 0

        # NEW: provide a t1 object compatible with your old calls
        self.t1 = MovementTraceNavigator()

    def connect(self):
        # Establish a connection with the PC
        retries: int = 0
        while not self.send_message:  # Keep trying until successful connection
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((self.host, self.port))
                self.send_message = True
                print("[PC Client] Connected to PC successfully.")
            except socket.error as e:
                retries += 1
                print("[PC Client] ERROR: Failed to connect -", str(e), "Retry no." + str(retries), "in 1 second...")
                time.sleep(1)

    def disconnect(self):
        # Disconnect from the PC
        try:
            if self.client_socket is not None:
                self.client_socket.close()
                self.send_message = False
                print("[PC Client] Disconnected from rpi.")
        except Exception as e:
            print("[PC Client] Failed to disconnect from rpi:", str(e))
    
    def reconnect(self):
        # Disconnect and then connect again
        print("[PC Client] Reconnecting...")
        self.send_message = False
        self.disconnect()
        self.connect()

    def send(self):
        while True:
            if self.send_message:
                message = self.msg_queue.get()
                exception = True
                while exception:
                    try:
                        self.client_socket.sendall(self.prepend_msg_size(message))
                        print("[PC Client] Write to RPI: first 100=", message[:100])
                    except Exception as e:
                        print("[PC Client] ERROR: Failed to write to RPI -", str(e))
                        self.reconnect()
                    else:
                        exception = False
            
    def prepend_msg_size(self, message):
        message_bytes = message.encode("utf-8")
        message_len = len(message_bytes)
        length_bytes = message_len.to_bytes(4, byteorder="big")
        return length_bytes + message_bytes

    def receive_messages(self):
        try:
            image_counter = 0
            obs_id = 0
            retries = 0
            command = None
            while True:
                # Receive the length of the message
                length_bytes = self.receive_all(4)
                if not length_bytes:
                    print("[PC Client] PC Server disconnected.")
                    self.reconnect()
                message_length = int.from_bytes(length_bytes, byteorder="big")

                # Receive the actual message data
                message = self.receive_all(message_length)
                if not message:
                    print("[PC Client] PC Server disconnected remotely.")
                    self.reconnect()

                print("[PC Client] Received message: first 100:", message)

                message = json.loads(message)

                if message["type"] == "START_TASK":
                    # Generate movement_trace.json using algo, then send segment-by-segment
                    try:
                        print("[PC Client] Generating movement_trace via algo.task1 ...")
                        self.t1.generate_path(message)
                    except Exception as e:
                        print(f"[PC Client] ERROR running planner: {e}")
                        continue

                    command = self.t1.get_command_to_next_obstacle()  # first segment
                    obs_id = str(self.t1.get_obstacle_id())
                    if command and command.get("type") == "NAVIGATION":
                        self.msg_queue.put(json.dumps(command))
                        print("[PC Client] Sent first NAVIGATION segment.")
                    else:
                        print("[PC Client] No NAVIGATION segment available (maybe END).")

                elif message["type"] == "FASTEST_PATH":
                    command = {"type": "FASTEST_PATH"}
                    self.msg_queue.put(json.dumps(command))
                
                elif message["type"] == "test":
                    message = {"type": "IMAGE_RESULTS", "data": {"obs_id": "3", "img_id": "39"}}
                    self.msg_queue.put(json.dumps(message))

                elif message["type"] == "IMAGE_TAKEN":
                    # Add image inference implementation here:
                    encoded_image = message["data"]["image"]
                    # Decode the base64 encoded image string
                    decoded_image = base64.b64decode(encoded_image)
                    os.makedirs("captured_images", exist_ok=True)

                    if self.task_2:
                        image_path = f"captured_images/task2_obs_id_{obs_id}_{image_counter}.jpg"
                    else:
                        image_path = f"captured_images/task1_obs_id_{obs_id}_{image_counter}.jpg"
                    
                    print("Before opening image")

                    with open(image_path, "wb") as img_file:
                        img_file.write(decoded_image)

                    print("Before calling image recognition model")

                    image_prediction = model_inference.image_inference(
                        image_or_path=image_path,
                        obs_id=str(obs_id), 
                        image_counter=image_counter, 
                        image_id_map=[],
                        task_2=self.task_2
                    )
                    image_counter += 1
                    print(image_prediction)

                    if message["final_image"] == True:
                        
                        # Get last prediction and move forward
                        while image_prediction['data']['img_id'] == None and self.image_record is not None:
                            if self.image_record:
                                image_prediction = self.image_record.pop()
                            else:
                                break
                        
                        # TODO: Implement function to handle no prediction
                        # If still can't find a prediction, repeat the last command
                        # if image_prediction['data']['img_id'] == None:# and NUM_OF_RETRIES > retries:
                            
                            # if command['type'] == 'FASTEST_PATH':
                            #     image_prediction['data']['img_id'] = "38" # 38 is right, 39 is left
                            # else:
                            #     last_path = command['data']['path'][-1]
                            #     if (retries+1)%2==0:
                            #         command = {"type": "NAVIGATION", "data": {"commands": ['RF010','RB010'], "path": [last_path, last_path]}}
                            #     else:
                            #         command = {"type": "NAVIGATION", "data": {"commands": ['RB010','RF010'], "path": [last_path, last_path]}}

                            # self.msg_queue.put(json.dumps(command))
                            # retries += 1
                            # continue
                        

                        # copy image to images_result folder and rename them according to obs_id
                        destination_folder = "images_result"
                        os.makedirs(destination_folder, exist_ok=True)
                        if self.task_2:
                            destination_file = f"{destination_folder}/task2_result_obs_id_{obs_id}.jpg"
                        else:
                            destination_file = f"{destination_folder}/task1_result_obs_id_{obs_id}.jpg"
                        # image_path = image_prediction["image_path"] 
                        image_path = image_path # use the last captured image

                        print("Image path: ",image_path)
                        print("Destination file: ",destination_file)
                        shutil.copy(image_path, destination_file)

                        print(f"Image copied to {destination_file}")

                        # Remove unnecessary data
                        del image_prediction["data"]["bbox_area"]
                        del image_prediction["image_path"]


                        print("before detection send")
                        message = json.dumps(image_prediction)
                        
                        ######### For testing override ###############
                        # message = {"type": "IMAGE_RESULTS", "data": {"obs_id": f"{obs_id}", "img_id": "20"}}
                        ######### end of temp test code ##############

                        self.msg_queue.put(message)
                        print("after msg queue put message")
                        # self.t1.update_image_id(image_prediction['data']['img_id'])
                        image_counter = 0
                        retries = 0
                        if self.task_2:
                            obs_id += 1       

                        # Update self.t1 to input new path, may put this above the image inference if we don't want to wait and stop
                        if not self.t1.has_task_ended():
                            print("Sending next command")
                            command = self.t1.get_command_to_next_obstacle()
                            self.msg_queue.put(json.dumps(command))
                            obs_id = str(self.t1.get_obstacle_id())
                        else:
                            if not self.task_2:
                                print("[Algo] Task 1 ended")
                                stitching_images(r'images_result', r'image_recognition\stitched_image.jpg')
                                break # exit thread

                        self.image_record = [] # reset the image record

        except socket.error as e:
            print("[PC Client] ERROR:", str(e))

    def receive_all(self, size):
        data = b""
        while len(data) < size:
            chunk = self.client_socket.recv(size - len(data))
            if not chunk:
                raise ConnectionError("Connection closed unexpectedly")
            data += chunk
        return data
    

if __name__ == "__main__":
    client = PCClient()
    client.connect()
    
    PC_client_receive = threading.Thread(target=client.receive_messages, name="PC-Client_listen_thread")
    PC_client_send = threading.Thread(target=client.send, name="PC-Client_send_thread")

    PC_client_send.start()
    print("[PC Client] Sending thread started successfully")

    PC_client_receive.start()
    print("[PC Client] Listening thread started successfully")

    # Optionally join:
    PC_client_receive.join()
    PC_client_send.join()
    # print("[PC Client] All threads concluded, cleaning up...")

    client.disconnect()


    # For testing task 1 path generation
    # client.t1.generate_path({"type":"START_TASK","data":{"obstacles_file":"obstacles.json"}})