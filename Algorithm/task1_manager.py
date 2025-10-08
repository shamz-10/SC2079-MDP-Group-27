import json

from Algorithm import algo_testing as algo


class Task1Manager:
    """
    Adapter that:
    1) runs algo.task1(obstacles_file) to generate movement_trace.json
    2) loads it and splits into segments between IMAGE_REC markers
    3) provides old-school methods: generate_path, get_command_to_next_obstacle, get_obstacle_id
    """
    def __init__(self):
        self.segments = []       # list of {"commands": [...], "path": [[r,c], ...]}
        self.segment_idx = 0
        self.obs_id = 0
        self.trace = None
        self.obs_order = []

    def generate_path(self, message):
        """
        Runs algo to regenerate movement_trace.json and prepares segments.
        Expects message like: {"type":"START_TASK","data":{"obstacles_file":"obstacles.json"}}
        """
        print("Generating Movement Trace Path...")
        # This will compute and write movement_trace.json
        algo.task1(message)

        # Load and prepare segments
        self._load_trace("Algorithm/movement_trace.json")
        self._split_into_segments()
        self._load_obstacle_order()

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

            # accumulate until IMAGE or end
            while i_cmd < len(cmds):
                cmd = cmds[i_cmd]
                seg_cmds.append(cmd)
                # print("cmd appended", cmd)
                i_cmd += 1

                if i_cmd < len(cmds) and cmd == "IMAGE":
                    break

                # Determine how many path steps for grid coordinates to move forward
                steps = 0
                if cmd and len(cmd) >= 3 and cmd[0] == 'S':
                    try:
                        # steps = max(2, int(cmd[2:]) // 5)
                        steps = max(2, int(cmd[2:]) // 10)
                    except Exception:
                        steps = 2
                elif (cmd[0] == 'L' or cmd[0]== 'R'):
                    steps=1
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

        print("Segments:", segments)

        self.segments = segments
        
        # Sanity check to reset indices
        self.segment_idx = 0
        self.obs_id = 0
    
    def _load_obstacle_order(self, path="Algorithm/obstacle_visit_order.json"):
        with open(path, "r") as f:
            obstacles = json.load(f)
        self.obs_order = obstacles
        print("Obstacle order:", self.obs_order)

    def get_command_to_next_obstacle(self):
        """
        Returns one segment (commands + path) up to the next IMAGE_REC boundary,
        in the same NAVIGATION packet shape your RPi expects.
        """
        # Sanity check is has_task_ended fails
        if self.segment_idx >= len(self.segments):
            return {"type": "END"} 

        seg = self.segments[self.segment_idx]
        self.segment_idx += 1

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
        """
        Returns the index for the obstacle_visit_order to get the Android obstacle ID.
        Called together with get_command_to_next_obstacle
        """
        current = self.obs_id
        self.obs_id += 1
        return current

    def has_task_ended(self):
        print("segment_idx, segments", self.segment_idx, len(self.segments)+1)
        return self.segment_idx >= len(self.segments)