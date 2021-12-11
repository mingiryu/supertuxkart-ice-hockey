import pickle
import torch
import numpy as np
from .models import load_planner, load_detector, control, dense_transforms
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from PIL import Image

# Objective1: Build a planner model to create aim points from images stored in pkl file
#             Train planner model(s) using data in pkl file. We can pick and choose what to train for (attker -> ball, defender -> enemy karts)
#                   - sub objective: attacker to focus on ball (easy)
#                   - sub objective: defender attack enemy kart, protect goal, pickup items (hard)
# Objective2: Build a controller to take an action (accel, steer, fire)
#           Eg attacker flow:
#           - Goal 1: move to the ball
#           - Goal 2: if ball in range (|dist from ball to kart| < 3)
#                           move and aim towards goal

class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        ######################
        #   General Params   #
        ######################
        self.team = None
        self.num_players = None
        self.goal = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.frame = 0
        self.backup_timeout = 30
        self.target_velocity = [15, 15]
        self.memory_limit = 5
        self.puck_thresh = 0.2
        self.reset_counter = 0
        self.possession_time = 0
        self.backup_counter = [0, 0]
        self.backup_angle = [0.0, 0.0]
        self.last_backup = [0, 0]
        self.memory = [[], []] # pop and push every frame
        self.last_location = [None, None]
        self.lost_counter = [0, 0]

        ####################
        #   Debug Params   #
        ####################
        self.fig = None
        self.ax = None

        #######################
        #   Attacker Params   #
        #######################
        # self.attacker_planner = load_planner('attacker.th').to(self.device)
        self.attacker_detector = load_detector().eval().to(self.device)
        self.attacker_starting_location = None
        self.attacker_last_location = None

        #######################
        #   Defender Params   #
        #######################
        # self.defender_planner = load_planner('defender.th').to(self.device)
        self.defender_detector = load_detector().eval().to(self.device)
        self.defender_starting_location = None
        self.defender_last_location = None
        self.defender_target = -1
        self.defender_location_mem = None

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.fig, self.ax = plt.subplots(1,2)
        self.team, self.num_players = team, num_players
        self.goal = [0, 0, 64.5] if team % 2 == 0 else [0, 0, -64.5]

        return ['tux'] * num_players

    ####################
    #   Helper Funcs   #
    ####################
    def _to_image(self, world_coord, proj, view):
        """
        Translates 3D world coords to a 2D coord in the projection and view given
        :param world_coord: The [x,y,z] coord in the world to convert to a 2D point
        :param proj: The projection of the kart's camera
        :param view: The view of the kart's camera
        :return: A 2D numpy array of the world_coord translated based on the kart's camera projection and view
        """
        p = proj @ view @ np.array(list(world_coord) + [1])
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

    def _backup(self, counter, angle, frame, id):
        """
        Sets values within the agent to backup for x frames given by counter, and the angle to backup by
        :param counter: The number of frames to backup for
        :param angle: The angle to backup by
        :param frame: The current frame to set
        :param agent: The agent to apply the setup to
        :return: None
        """
        self.last_backup[id] = frame
        self.backup_counter[id] = counter
        self.backup_angle[id] = angle

    def _in_poss_of_puck(self, score):
        """
        Checks the average puck locations stored in memory. If average is below some threshold, return true else false
        :return: Return true if the average of the puck locations in memory is below some threshold
        """
        if score >= 12:
            self.possession_time += 1
            return True
        else:
            self.possession_time = 0
            return False

    def _is_goal_ahead(self, kart):
        front = kart['front']
        loc = kart['location']

        heading_to_ends = abs(front[2]) - abs(loc[2]) # if |front_z| > |loc_z|, heading to ends of field
        x_aligned = abs(front[0]) - abs(loc[0]) <= 0.5 # if |front_x| - |loc_x| <= 0.5, we are nearly pointing straight
        goal_ahead = np.sign(front[2] - loc[2]) == np.sign(self.goal[2]) # goal_z sign must match the diff between front_z and loc_z


        return heading_to_ends > 0.0 and x_aligned and goal_ahead

    def _check_isPlayerKart(self, cx, cy):
        """
        Checks if the predicted point is a part of the player kart object itself.
        If yes, returns (0, 0). If no, just returns (cx, cy)
        """
        out_x, out_y = cx, cy

        if (140 <= cx and cx <= 260) and (140 <= cy and cy <= 235): # It is a part of the player kart
            out_x, out_y = 0.0, 0.0

        return out_x, out_y

    def _compute_quadrant(self, direction_vector):
        """
        Calculates the quadrant of the given vector
        :param direction_vector: the aim direction of the kart
        :return: the quadrant number
        """
        if np.sign(direction_vector[0]) >= 0.0 and np.sign(direction_vector[2]) > 0.0:
            return 1
        elif np.sign(direction_vector[0]) < 0.0 and np.sign(direction_vector[2]) >= 0.0:
            return 2
        elif np.sign(direction_vector[0]) <= 0.0 and np.sign(direction_vector[2]) < 0.0:
            return 3
        elif np.sign(direction_vector[0]) > 0.0 and np.sign(direction_vector[2]) <= 0.0:
            return 4

    def _move_to_quadrant(self, dest_q, loc_q, aim_q, kart):
        """
        Function returns an aim_point in the direction of the dest_q
        :param dest_q: The destination quadrant
        :param loc_q: The starting location quadrant of kart
        :param aim_q: The quadrant the kart is aiming at
        :param kart: The kart information
        :return: An aim point in the direction of the dest_q
        """
        q1 = [30, 0, 45]
        q2 = [-30, 0, 45]
        q3 = [-30, 0, -45]
        q4 = [30, 0, -45]
        ap = None
        if dest_q == 1:
            ap = q1
        elif dest_q == 2:
            ap = q2
        elif dest_q == 3:
            ap = q3
        else:
            ap = q4
        return self._to_image(ap, np.array(kart['camera']['projection']).T, np.array(kart['camera']['view']).T)

    def _generate_dict(self, aim_point, score, kart, id):
        """
        Generates a dict of actions
        :param aim_point: The point to aim towards
        :param score: The score for the predicted detection
        :param kart: The kart whom we are building the dict for
        :param id: ID of the kart. 0 or 1 (attacker or defender)
        :return: A dict of actions
        """
        dict = {'acceleration':0.0, 'steer':0.0, 'brake': False, 'fire': False, 'nitro': False}
        vel = np.linalg.norm(kart['velocity'])

        ### general actions ###
        # if we have nitro and the aim_point is nearly straight ahead
        if aim_point[0] < 0.05 and aim_point[0] > -0.05:
            dict['nitro'] = True

        # compute acceleration
        dict['acceleration'] = 1.0 if vel < self.target_velocity[id] else 0.0

        # compute steering
        steer_angle = 4 * aim_point[0]
        dict['steer'] = np.clip(steer_angle * 4, -1, 1)

        # did not detect a puck, turn in a circle till we find it
        # TODO: improve turning based on area of the map. use 0,0,0 as anchor?
        if aim_point[0] == 0.0 and aim_point[1] == 0.0 and self.frame - self.reset_counter > self.backup_timeout:
            dict['steer'] = 1

        ### rescue actions ###
        if self.frame - self.last_backup[id] > self.backup_timeout:
            # passed the puck?
            if abs(aim_point[0]) > 0.85:

                # if aim_point[1] <= 0.5:
                print("PASSED PUCK BACKUP", id)
                score = 0.0
                self._backup(30, -1 * dict['steer'], self.frame, id)
            if aim_point[1] > 0.3:
                print("PUCK > 0.3 BACKUP", id)
                score = 0.0
                self._backup(30, -1 * dict['steer'], self.frame, id)

            # heading into a goal... don't do that
            # second check is checking that front-z value > location-z value. ie, kart is heading into goal
            if abs(kart['location'][2]) >= abs(self.goal[2]) and abs(kart['front'][2]) > abs(kart['location'][2]):
                print("GOAL BACKUP", id)
                angle = 0.0
                if (kart['location'][0] < 0.0 and kart['location'][2] < 0.0) or (kart['location'][0] > 0.0 and kart['location'][2] > 0.0):
                    # print("GOAL BACKUP")
                    # top left or bottom right -> backup to left
                    angle = -1.0
                if (kart['location'][0] > 0.0 and kart['location'][2] < 0.0) or (kart['location'][0] < 0.0 and kart['location'][2] > 0.0):
                    # print("GOAL BACKUP")
                    # top right or bottom left -> backup to right
                    angle = 1.0
                self._backup(30, angle, self.frame, id)

            # might be stuck near wall. try backing up
            if vel < 1.0:
                print("STUCK BACKUP", id)
                self._backup(15, -1 * dict['steer'], self.frame, id)

        # perform backup
        if self.backup_counter[id] > 0:
            if score == 0.0 or self.backup_counter[id] <= 5:
                dict['brake'] = True
                dict['acceleration'] = 0.0
                dict['steer'] = self.backup_angle[id]
                self.backup_counter[id] -= 1
                # if self.backup_counter[id] == 0:
                    # completed a full backup but did not receive a new score. emptying memory
                    # self.memory[id] = []

            # puck came back into screen. forcibly backup
            if self.backup_counter[id] > 5 and score != 0.0:
                self.backup_counter[id] = 5

        return dict

    def act(self, player_state, player_image):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        #####################
        #   ATTACKER CODE   #
        #####################
        # some setup
        attacker_dict = {'acceleration':0.0, 'steer':0.0, 'brake': False, 'fire': False, 'nitro': False}
        attacker = player_state[0]['kart']
        attacker_cam = player_state[0]['camera']
        attacker_vel = np.linalg.norm(attacker['velocity'])
        attacker_loc_2d = self._to_image(attacker['location'], np.array(attacker_cam['projection']).T, np.array(attacker_cam['view']).T)
        attacker_in_poss = False
        tensor = F.to_tensor(player_image[0])
        tensor = tensor[None, :].to(self.device)
        if self.attacker_starting_location == None:
            self.attacker_starting_location = attacker['location']

        # predicting attacker_aim_point
        score = 0.0
        center_x = 0.0
        center_y = 0.0
        detections = self.attacker_detector.detect(tensor[0], min_score=5.0)
        attacker_aim_point = np.array([0.0, 0.0])
        for c in range(2,3):
            if len(detections[c]) > 0:
                cls = detections[c]
                if len(cls) > 0:
                    self.lost_counter[0] = 0
                    # self.target_velocity[0] =
                    s, cx, cy, w, h = cls[0]
                    score = s
                    # normalizing to (-1, 1)
                    center_x = float((cx) / 200) - 1
                    center_y = float((cy) / 150) - 1
                    attacker_aim_point = np.array([center_x, center_y])

                    # add attacker_aim_point to memory
                    if len(self.memory[0]) == self.memory_limit:
                        self.memory[0].pop(0)
                    self.memory[0].append((score, attacker_aim_point))
            elif self.lost_counter[0] <= 15:
                # check our memory for the last puck location
                if len(self.memory[0]) > 0:
                    score = 0.0
                    _, ap = self.memory[0][-1]
                    attacker_aim_point = ap
            self.lost_counter[0] += 1

        # print("DEF LOC MEM:", self.defender_location_mem)
        print("LOST COUNTER:", self.lost_counter[0])
        if score == 0.0 and attacker_aim_point[0] == 0.0 and attacker_aim_point[1] == 0.0 and len(self.memory[0]) == 0 and self.defender_location_mem != None:
            # check our defender's memory
            def_loc, def_front = self.defender_location_mem
            # def_dir_vec = np.array(def_front) - np.array(def_loc)
            # att_dir_vec = np.array(attacker['front']) - np.array(attacker['location'])
            # # def_loc_q = self._compute_quadrant(def_loc)
            # def_aim_q = self._compute_quadrant(def_dir_vec)
            # att_loc_q = self._compute_quadrant(attacker['location'])
            # att_aim_q = self._compute_quadrant(att_dir_vec)
            #
            # attacker_aim_point = self._move_to_quadrant(def_aim_q, att_loc_q, att_aim_q, player_state[0])
            attacker_aim_point = self._to_image(def_loc, np.array(attacker_cam['projection']).T, np.array(attacker_cam['view']).T)

        # if in possession of the puck, ignore attacker_aim_point
        # instead, start driving to goal
        if self._in_poss_of_puck(score):
            if self.possession_time > 5:
                attacker_in_poss = True
                # is the goal ahead?
                if self._is_goal_ahead(attacker):
                    # print("HEADING TO GOAL!")
                    goal_point = self._to_image(self.goal, np.array(attacker_cam['projection']).T, np.array(attacker_cam['view']).T)
                    d = np.linalg.norm(np.array(attacker_aim_point[0]) - np.array(goal_point[0]))
                    gain = 0.25
                    angle_of_attack = np.sign(attacker['location'][2]) * np.sign(attacker['location'][0]) * np.sign(self.goal[2]) * -1
                    attacker_aim_point[0] += d * gain * angle_of_attack
                else:
                    # print("LOCATION:", attacker['front'], attacker['location'])
                    # start moving the ball closer to the goal rather than just pushing it
                    goal_sign = np.sign(self.goal[2])
                    direction_vector = np.array(attacker['front']) - np.array(attacker['location'])
                    curr_q = self._compute_quadrant(attacker['location'])
                    aim_q = self._compute_quadrant(direction_vector)

                    left_angle = 0.1 * goal_sign * -1
                    right_angle = 0.1 * goal_sign
                    attack_qs = [1,2] if self.team % 2 == 0 else [3,4]
                    def_qs = [3,4] if self.team % 2 == 0 else [1,2]

                    # attacking
                    if curr_q in attack_qs:
                        if curr_q == attack_qs[0] and aim_q != curr_q:
                            # aim to the left (1 and 3)
                            attacker_aim_point[0] += left_angle
                        else:
                            # aim to the right
                            attacker_aim_point[0] += right_angle

                        if curr_q == attack_qs[1] and aim_q != curr_q:
                            # aim to the right (2 and 4)
                            attacker_aim_point[0] += right_angle
                        else:
                            # aim to the left
                            attacker_aim_point[0] += left_angle

                    # defending
                    if curr_q in def_qs:
                        # red team
                        if curr_q in [3,4] and aim_q in [2,3]:
                            # aim to the left
                            attacker_aim_point[0] += left_angle
                        if curr_q in [3,4] and aim_q in [1,4]:
                            # aim to the right
                            attacker_aim_point[0] += right_angle
                        # blue team
                        if curr_q in [1,2] and aim_q in [1,4]:
                            # aim to the left
                            attacker_aim_point[0] += left_angle
                        if curr_q in [1,2] and aim_q in [2,3]:
                            # aim to the right
                            attacker_aim_point[0] += right_angle

        attacker_dict = self._generate_dict(attacker_aim_point, score, attacker, 0)

        # reset calculation
        if self.attacker_last_location != None:
            d = np.linalg.norm(np.array([attacker['location'][0], attacker['location'][2]]) - np.array([self.attacker_last_location[0], self.attacker_last_location[2]]))
            if d > 4:
                print("\t\n\nRESETTING\n\n")
                self.target_velocity[0] = 10
                self.reset_counter = self.frame
                self.memory[0] = []
        self.attacker_last_location = attacker['location']

        #####################
        #   DEFENDER CODE   #
        #####################
        # some setup
        defender_dict = {'acceleration':0.0, 'steer':0.0, 'brake': False, 'fire': False, 'nitro': False}
        defender = player_state[1]['kart']
        defender_cam = player_state[1]['camera']
        defender_vel = np.linalg.norm(defender['velocity'])
        defender_loc_2d = self._to_image(defender['location'], np.array(defender_cam['projection']).T, np.array(defender_cam['view']).T)
        tensor = F.to_tensor(player_image[1])
        tensor = tensor[None, :].to(self.device)
        if self.defender_starting_location == None:
            self.defender_starting_location = defender['location']

        # predicting defender_aim_point
        score = 0.0
        center_x = 0.0
        center_y = 0.0
        detections = self.defender_detector.detect(tensor[0], min_score=5.0)
        defender_aim_points = [np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])]
        defender_aim_point = np.array([0.0, 0.0])
        friendly_kart = self._to_image(attacker['location'], np.array(defender_cam['projection']).T, np.array(defender_cam['view']).T)
        ### priority 1: shoot enemies with items if available ###
        for c in range(0,3):
            if len(detections[c]) > 0:
                cls = detections[c]
                if c == 0:
                    for s, cx, cy, w, h in cls:
                        cx, cy = self._check_isPlayerKart(cx, cy)
                        score = s
                        # normalizing to (-1, 1)
                        center_x = float((cx) / 200) - 1
                        center_y = float((cy) / 150) - 1
                        if cx != 0 and cy != 0:
                            kart_loc = np.array([center_x, center_y])
                            d = np.linalg.norm(np.array(kart_loc) - np.array(friendly_kart))
                            if d > 0.1:
                                # if d > 0.1, most likely an enemy kart
                                defender_aim_points[c] = np.array(kart_loc)
                                break

                else:
                    ### priority 2 & 3: shoot puck with items if available, pickup item ###
                    if len(cls) > 0 :
                        s, cx, cy, w, h = cls[0]
                        score = s
                        # normalizing to (-1, 1)
                        center_x = float((cx) / 200) - 1
                        center_y = float((cy) / 150) - 1
                        aim_point = np.array([center_x, center_y])

                        defender_aim_points[c] = np.array(aim_point)

                        # add puck to memory
                        if c == 2:
                            if len(self.memory[1]) == self.memory_limit:
                                self.memory[1].pop(0)
                            self.memory[1].append((score, aim_point))
                            self.defender_location_mem = (defender['location'], defender['front'])
                    else:
                        # can't find something, just go for the puck.
                        self.defender_target = -1
                        # check our memory for the last puck location
                        if len(self.memory[1]) > 0:
                            score = 0.0
                            _, ap = self.memory[1][-1]
                            defender_aim_points[c] = np.array(ap)

        # assess the situation and react
        if defender['powerup']['type'] > 0:
            # have an item. shoot it at an enemy
            self.defender_target = 1
            defender_aim_point = defender_aim_points[0]
        elif self.lost_counter[0] > 10:
            # our attacker is lost. search for the puck and communicate it to them
            self.defender_target = 0
            defender_aim_point = defender_aim_points[1]
        else:
            # all other conditions are fine. collect items to shoot
            self.defender_target = 0
            defender_aim_point = defender_aim_points[2]


        # if self.defender_target == 2 and self.frame - self.reset_counter > 30:
        #     if self._in_poss_of_puck(score):
        #         if self.possession_time > 5:
        #             # is the goal ahead?
        #             if self._is_goal_ahead(defender):
        #                 # print("HEADING TO GOAL!")
        #                 goal_point = self._to_image(self.goal, np.array(defender_cam['projection']).T, np.array(defender_cam['view']).T)
        #                 d = np.linalg.norm(np.array(defender_aim_point[0]) - np.array(goal_point[0]))
        #                 gain = 0.25
        #                 angle_of_attack = np.sign(defender['location'][2]) * np.sign(defender['location'][0]) * np.sign(self.goal[2]) * -1
        #                 defender_aim_point[0] += d * gain * angle_of_attack
        #             else:
        #                 # print("LOCATION:", attacker['front'], attacker['location'])
        #                 # start moving the ball closer to the goal rather than just pushing it
        #                 goal_sign = np.sign(self.goal[2])
        #                 direction_vector = np.array(defender['front']) - np.array(defender['location'])
        #                 curr_q = self._compute_quadrant(defender['location'])
        #                 aim_q = self._compute_quadrant(direction_vector)
        #
        #                 left_angle = 0.1 * goal_sign * -1
        #                 right_angle = 0.1 * goal_sign
        #                 attack_qs = [1,2] if self.team % 2 == 0 else [3,4]
        #                 def_qs = [3,4] if self.team % 2 == 0 else [1,2]
        #
        #                 # attacking
        #                 if curr_q in attack_qs:
        #                     if curr_q == attack_qs[0] and aim_q != curr_q:
        #                         # aim to the left (1 and 3)
        #                         defender_aim_point[0] += left_angle
        #                     else:
        #                         # aim to the right
        #                         defender_aim_point[0] += right_angle
        #
        #                     if curr_q == attack_qs[1] and aim_q != curr_q:
        #                         # aim to the right (2 and 4)
        #                         defender_aim_point[0] += right_angle
        #                     else:
        #                         # aim to the left
        #                         defender_aim_point[0] += left_angle
        #
        #                 # defending
        #                 if curr_q in def_qs:
        #                     # red team
        #                     if curr_q in [3,4] and aim_q in [2,3]:
        #                         # aim to the left
        #                         defender_aim_point[0] += left_angle
        #                     if curr_q in [3,4] and aim_q in [1,4]:
        #                         # aim to the right
        #                         defender_aim_point[0] += right_angle
        #                     # blue team
        #                     if curr_q in [1,2] and aim_q in [1,4]:
        #                         # aim to the left
        #                         defender_aim_point[0] += left_angle
        #                     if curr_q in [1,2] and aim_q in [2,3]:
        #                         # aim to the right
        #                         defender_aim_point[0] += right_angle

        # defender_dict = {"acceleration":1.0, "steer": 0.0}
        defender_dict = self._generate_dict(defender_aim_point, score, defender, 1)
        if self.defender_target == 1 and self.frame - self.reset_counter > 50:
            defender_dict['fire'] = True

        # reset calculation
        if self.defender_last_location != None:
            d = np.linalg.norm(np.array([defender['location'][0], defender['location'][2]]) - np.array([self.defender_last_location[0], self.defender_last_location[2]]))
            if d > 4:
                self.target_velocity[1] = 10
                self.reset_counter = self.frame
                self.memory[1] = []
                self.defender_location_mem = None
        self.defender_last_location = defender['location']


        # debug logging
        # print("ATTACKER AIM POINT", attacker_aim_point, np.array([400, 300]) / 2)
        for index, row in enumerate(self.ax):
            row.clear()
            if index == 0:
                row.imshow(Image.fromarray(player_image[0]))
                WH2 = np.array([400, 300]) / 2
                row.add_artist(plt.Circle(WH2*(1+np.array(attacker_loc_2d)), 2, ec='b', fill=False, lw=1.5))
                row.add_artist(plt.Circle(WH2*(1+np.array(self._to_image(self.goal, np.array(attacker_cam['projection']).T, np.array(attacker_cam['view']).T))), 2, ec='y', fill=False, lw=1.5))
                row.add_artist(plt.Circle(WH2*(1+attacker_aim_point), 2, ec='r', fill=False, lw=1.5))
            if index == 1:
                row.imshow(Image.fromarray(player_image[1]))
                WH2 = np.array([400, 300]) / 2
                row.add_artist(plt.Circle(WH2*(1+np.array(defender_loc_2d)), 2, ec='b', fill=False, lw=1.5))
                row.add_artist(plt.Circle(WH2*(1+np.array(friendly_kart)), 2, ec='g', fill=False, lw=1.5))
                row.add_artist(plt.Circle(WH2*(1+np.array(self._to_image(self.goal, np.array(defender_cam['projection']).T, np.array(defender_cam['view']).T))), 2, ec='y', fill=False, lw=1.5))
                row.add_artist(plt.Circle(WH2*(1+defender_aim_point), 2, ec='r', fill=False, lw=1.5))
        plt.pause(1e-3)

        self.frame += 1

        return [attacker_dict, defender_dict]
        # return [attacker_dict, dict(acceleration=0, steer=0)]
