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
from Algorithm.task1_manager import Task1Manager

# Configuration
TASK_2 = True  #TODO: Change to False for task 1, True for task 2
NUM_OF_RETRIES = 1  # Number of retries for image inference if no detections

# Constants
RPI_IP = "192.168.27.27"  # Replace with the Raspberry Pi's IP address
PC_PORT = 8888            # Replace with the port used by the PC server
PC_BUFFER_SIZE = 1024


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
        self.image_id_set = set()
        self.t1 = Task1Manager()

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
                        print("[PC Client] Write to RPI: first 100=", message[:200])
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

                print("[PC Client] Received message: first 100:", message[:200])

                message = json.loads(message)

                # Task 1 start
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

                # Task 2 start
                elif message["type"] == "FASTEST_PATH":
                    command = {"type": "FASTEST_PATH"}
                    self.msg_queue.put(json.dumps(command))
                
                # elif message["type"] == "test":
                #     message = {"type": "IMAGE_RESULTS", "data": {"obs_id": "3", "img_id": "39"}}
                #     self.msg_queue.put(json.dumps(message))

                elif message["type"] == "IMAGE_TAKEN":
                    # Add image inference implementation here:
                    encoded_image = message["data"]["image"]
                    # Decode the base64 encoded image string
                    decoded_image = base64.b64decode(encoded_image)
                    os.makedirs("captured_images", exist_ok=True)

                    if self.task_2:
                        image_path = f"captured_images/task2_obs_id_{obs_id}_{image_counter}.jpg"
                    else:
                        image_path = f"captured_images/task1_obs_id_{self.t1.obs_order[int(obs_id)]}_{image_counter}.jpg"
                    
                    # print("Before opening image")

                    with open(image_path, "wb") as img_file:
                        img_file.write(decoded_image)

                    # print("Before calling image recognition model")
                    if self.task_2:
                        image_prediction = model_inference.image_inference(
                            image_or_path=image_path,
                            obs_id=str(obs_id),
                            image_counter=image_counter, 
                            image_id_map=[],
                            task_2=self.task_2
                        )
                    else:
                        image_prediction = model_inference.image_inference(
                            image_or_path=image_path,
                            obs_id=str(self.t1.obs_order[int(obs_id)]), 
                            image_counter=image_counter, 
                            image_id_map=[],
                            task_2=self.task_2
                        )
                    # If not final image, store the prediction for potential back-tracing
                    self.image_record.append(image_prediction)
                    image_counter += 1
                    print(image_prediction)

                    if message["final_image"] == True:
                        
                        # If no detections in final image, back trace to previous images
                        while image_prediction['data']['img_id'] == None and self.image_record is not None:
                            if self.image_record:
                                image_prediction = self.image_record.pop()
                            else:
                                break

                        # TODO: Comment out if expecting no detections
                        # If still no detections, move back to last obstacle and retry once
                        # if image_prediction['data']['img_id'] == None and NUM_OF_RETRIES > retries:

                        #     last_path = command['data']['path'][-1]
                        #     command = {"type": "NAVIGATION", "data": {"commands": ['SB010','IMAGE','SF010'], "path": [last_path, last_path]}}
                        #     self.msg_queue.put(json.dumps(command))
                        #     retries += 1
                        #     continue

                        #################### Duplicate image id check ####################
                        # TODO: Uncomment if want to prevent duplicate image ids, but risk messing up downstream if earlier images registered the wrong image id
                        # Ensure no duplicate image ids are sent
                        # img_id = image_prediction['data']['img_id']
                        # if img_id is not None and img_id in self.image_id_set:
                        #     print(f"Duplicate image id {img_id} detected, skipping...")
                        #     continue
                        # self.image_id_set.add(img_id)
                        #################################################################
                                                
                        # copy image to images_result folder and rename them according to obs_id
                        destination_folder = "images_result"
                        os.makedirs(destination_folder, exist_ok=True)
                        if self.task_2:
                            destination_file = f"{destination_folder}/task2_result_obs_id_{obs_id}.jpg"
                        else:
                            destination_file = f"{destination_folder}/task1_result_obs_id_{self.t1.obs_order[int(obs_id)]}.jpg"
                        image_path = image_prediction["image_path"] 
                        
                        # if no detections
                        # if not os.path.exists(image_path):
                        #     image_path = f"captured_images/task1_obs_id_{self.t1.obs_order[int(obs_id)]}_{image_counter-1}.jpg"

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
                        # message = json.dumps({"type": "IMAGE_RESULTS", "data": {"obs_id": f"{obs_id}", "img_id": "20"}})
                        ######### end of temp test code ##############

                        self.msg_queue.put(message)
                        print("after msg queue put message")
                        image_counter = 0
                        retries = 0
                        if self.task_2:
                            obs_id += 1       

                        if self.task_2:
                            if obs_id > 1:
                                stitching_images(r'images_result', r'image_recognition/stitched_image.jpg')

                        # Update self.t1 to input new path, may put this above the image inference if we don't want to wait and stop
                        if not self.t1.has_task_ended():
                            print("Sending next command")
                            command = self.t1.get_command_to_next_obstacle()
                            self.msg_queue.put(json.dumps(command))
                            obs_id = str(self.t1.get_obstacle_id())
                        else:
                            if not self.task_2:
                                print("[Algo] Task 1 ended")
                                stitching_images(r'images_result', r'image_recognition/stitched_image.jpg')
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
    print("[PC Client] All threads concluded, cleaning up...")

    client.disconnect()