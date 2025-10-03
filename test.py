commands = ['SF020', 'RF087', 'SF065', 'LF087', 'LB087', 'SF010', 'IMAGE', 'LB087', 'RB087', 'SF005', 'LB087', 'SF005', 'IMAGE', 'SB065', 'LF087', 'SF030', 'IMAGE']
path = [
      [
        1,
        1
      ],
      [
        1,
        1
      ],
      [
        1,
        2
      ],
      [
        1,
        2
      ],
      [
        1,
        3
      ],
      [
        4,
        4
      ],
      [
        4,
        4
      ],
      [
        5,
        4
      ],
      [
        5,
        4
      ],
      [
        6,
        4
      ],
      [
        6,
        4
      ],
      [
        7,
        4
      ],
      [
        7,
        4
      ],
      [
        8,
        4
      ],
      [
        8,
        4
      ],
      [
        9,
        4
      ],
      [
        9,
        4
      ],
      [
        10,
        4
      ],
      [
        10,
        4
      ],
      [
        12,
        8
      ],
      [
        11,
        5
      ],
      [
        11,
        5
      ],
      [
        12,
        5
      ],
      [
        9,
        7
      ],
      [
        7,
        10
      ],
      [
        8,
        10
      ],
      [
        5,
        11
      ],
      [
        5,
        11
      ],
      [
        5,
        11
      ],
      [
        5,
        12
      ],
      [
        5,
        12
      ],
      [
        5,
        13
      ],
      [
        5,
        13
      ],
      [
        5,
        14
      ],
      [
        5,
        14
      ],
      [
        5,
        15
      ],
      [
        5,
        15
      ],
      [
        5,
        16
      ],
      [
        5,
        16
      ],
      [
        5,
        17
      ],
      [
        5,
        17
      ],
      [
        9,
        15
      ],
      [
        9,
        15
      ],
      [
        10,
        15
      ],
      [
        10,
        15
      ],
      [
        11,
        15
      ],
      [
        11,
        15
      ],
      [
        12,
        15
      ]
    ]

def _split_into_segments(commands, path):
        """
        Split trace.data.commands/path into chunks up to each IMAGE_REC token.
        Each movement command advances one path index; IMAGE_REC does not.
        """
        cmds = commands
        path = path
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

            # seg_path = scale_path_to_20x20(seg_path)

            if seg_cmds or seg_path:
                segments.append({"commands": seg_cmds, "path": seg_path})

            # consume IMAGE_REC boundary (don’t move along path)
            # if i_cmd < len(cmds) and cmds[i_cmd] == "IMAGE":
            #     i_cmd += 1
        print("Commands:", commands)
        print("Segments:", segments)

def _dedup_path(path_slice):
    deduped = [path_slice[0]]
    for p in path_slice[1:]:
        if p != deduped[-1]:
            deduped.append(p)
    return deduped

print("After deduplication:", _dedup_path(path))

new_path = _dedup_path(path)


_split_into_segments(commands, new_path)