from pc import PCInterface
from android import AndroidInterface
from stm import STMInterface
from threading import Thread

TASK_2 = True

class RPiMain:
    def __init__(self, task2):
        # Initialize interfaces
        self.Android = AndroidInterface(self)
        self.PC = PCInterface(self, task2=task2)
        self.STM = STMInterface(self, task2=task2)

    def connect_components(self):
        # Connect all components
        self.Android.connect()
        self.PC.connect()
        self.STM.connect()
        
    def cleanup(self):
    # Disconnect from all components
        self.Android.disconnect()
        self.PC.disconnect()
        self.STM.disconnect()

    def run(self):
        print("[RPiMain] Starting RPiMain...")

        # Connect components
        self.connect_components()
        print("[RPiMain] Components connected successfully")

        # Create threads for sending messages
        Android_send = Thread(target=self.Android.send, name="Android_send_thread")
        PC_send = Thread(target=self.PC.send, name="PC_send_thread")
        STM_send = Thread(target=self.STM.send, name="STM_send_thread")

        # Create threads for receiving messages
        Android_listen = Thread(target=self.Android.listen, name="Android_listen_thread")
        PC_listen = Thread(target=self.PC.listen, name="PC_listen_thread")

        # Start sending threads
        Android_send.start()
        PC_send.start()
        STM_send.start()
        print("[RPiMain] Sending threads started successfully")

        # Start listening threads
        Android_listen.start()
        PC_listen.start()
        print("[RPiMain] Listening threads started successfully")

        # Wait for threads to end
        Android_send.join()
        PC_send.join()
        STM_send.join()
        Android_listen.join()
        PC_listen.join()

        print("[RPiMain] All threads concluded, cleaning up...")

        # Cleanup after threads finish
        self.cleanup()

        print("[RPiMain] Exiting RPiMain...") 

if __name__ == "__main__":
    rpi = RPiMain(TASK_2)
    rpi.run()
