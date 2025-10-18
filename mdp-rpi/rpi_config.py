# Configuration constants
# LOCATION = "OUT" # IN (indoors) / OUT (outdoors) / NONE (disable turn adjustment)
LOCATION = "NONE"

RPI_IP = "192.168.27.27"
MSG_LOG_MAX_SIZE = 150 # characters

# PC Interface
PC_PORT = 8888
PC_BUFFER_SIZE = 2048

# Camera Interface
NUM_IMAGES = 2

# Android Interface
BT_UUID = "00001101-0000-1000-8000-00805f9b34fb"
BT_BUFFER_SIZE = 2048

# STM Interface
STM_BAUDRATE = 115200
STM_ACK_MSG = "A"
STM_NAV_COMMAND_FORMAT = '^[SLR][FB][0-9]{3}$' # task 1

# Task 2: translate PC commands for moving around obstacles to STM_NAV_COMMAND_FORMAT
# Second Left and Right commands include returning to carpark commands
STM_OBS_ROUTING_MAP = {
    "FIRSTLEFT": ["LF060", "RF060", "RF060", "LF060"],
    "FIRSTRIGHT": ["RF060", "LF060", "LF060", "RF060"],
    "SECONDLEFT": ["LF089","SB045","WR999","RF179",
                   "WR999","SF005","RF089","HOMEE",
                   "RF090","DUMEE","LF090","START","ENYAO"],
    "SECONDRIGHT": ["RF089","SB045","WL999","LF179",
                    "WL999","SF005","LF088","HOMEE",
                    "LF090","DUMEE","RF090","START","ENYAO"],
    "SECONDLEFTALT": ["LF089", "SB025","WR999", "RF179", 
                      "WR999","SF005", "RF089","HOMEE",
                      "RF090","DUMEE","LF090","START","ENYAO"],
    "SECONDRIGHTALT": ["RF089","SB025", "WL999", "LF179", 
                       "WL999","SF005","LF088","HOMEE",
                       "LF090","DUMEE","RF090","START","ENYAO"]

}
