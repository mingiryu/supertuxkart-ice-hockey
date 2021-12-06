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
        self.target_velocity = 20
        self.memory_limit = 5
        self.puck_thresh = 0.2
        self.reset_counter = 0

        ####################
        #   Debug Params   #
        ####################
        self.fig = None
        self.ax = None

        #######################
        #   Attacker Params   #
        #######################
        self.attacker_planner = load_planner('attacker.th').to(self.device)
        self.attacker_backup_counter = 0
        self.attacker_backup_angle = None
        self.attacker_last_backup = 0
        self.attacker_starting_location = None
        self.attacker_memory = [] # pop and push every frame
        self.attacker_last_location = None

        #######################
        #   Defender Params   #
        #######################
        self.defender_planner = load_planner('defender.th').to(self.device)
        self.defender_backup_counter = 0
        self.defender_backup_angle = None
        self.defender_last_backup = 0
        self.defender_starting_location = None
        self.defender_memory = [] # pop and push every frame
        self.defender_last_location = None

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

    def _backup(self, counter, angle, frame, agent):
        """
        Sets values within the agent to backup for x frames given by counter, and the angle to backup by
        :param counter: The number of frames to backup for
        :param angle: The angle to backup by
        :param frame: The current frame to set
        :param agent: The agent to apply the setup to
        :return: None
        """
        if agent == 0:
            self.attacker_last_backup = frame
            self.attacker_backup_counter = counter
            self.attacker_backup_angle = angle

        if agent == 1:
            self.defender_last_backup = frame
            self.defender_backup_counter = counter
            self.defender_backup_angle = angle

    def _in_poss_of_puck(self, next_aim_point, kart_loc):
        """
        Checks the average puck locations stored in memory. If average is below some threshold, return true else false
        :return: Return true if the average of the puck locations in memory is below some threshold
        """
        puck_x = []
        puck_y = []
        avg_x = 1
        avg_y = 1

        dists = []
        # calculate dists between next_aim_point and those in memory
        if len(self.attacker_memory) == self.memory_limit:
            for pos in self.attacker_memory:
                puck_x.append(pos[0])
                puck_y.append(pos[1])
                dists.append(np.linalg.norm(np.array(next_aim_point) - np.array(pos)))
        else:
            return False

        # get average puck x and y's
        avg_x = np.mean(np.array(puck_x))
        avg_y = np.mean(np.array(puck_y))
        # if dist < another threshold, ie very close, we are in possession
        if np.linalg.norm(np.array([avg_x, avg_y]) - kart_loc) <= 0.25:
            # if avg_dist < threshold, check avg puck_x and avg puck_y dist from kart
            avg_dist = np.mean(np.array(dists))
            print("AVG DISTS:", avg_dist)
            return avg_dist <= self.puck_thresh
        else:
            return False

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
        tensor = F.to_tensor(player_image[0])
        tensor = tensor[None, :].to(self.device)
        if self.attacker_starting_location == None:
            self.attacker_starting_location = attacker['location']

        # predicting attacker_aim_point
        attacker_aim_point = self.attacker_planner(tensor).detach().cpu().numpy()
        attacker_aim_point = attacker_aim_point[0]

        # if in possession of the puck, ignore attacker_aim_point
        # instead, start driving to goal
        if self._in_poss_of_puck(attacker_aim_point, attacker_loc_2d):
            print("ATTACKER: IN POSSESSION OF THE PUCK!")
            self.target_velocity = 10
            attacker_aim_point = self._to_image(self.goal, np.array(attacker_cam['projection']).T, np.array(attacker_cam['view']).T)

        # if we have nitro and the aim_point is nearly straight ahead
        if attacker_aim_point[0] < 0.05 and attacker_aim_point[0] > -0.05:
            attacker_dict['nitro'] = True

        # compute acceleration
        attacker_dict['acceleration'] = 1.0 if attacker_vel < self.target_velocity else 0.0

        # compute steering
        steer_angle = 3 * attacker_aim_point[0]
        attacker_dict['steer'] = np.clip(steer_angle * 3, -1, 1)

        # if the puck is far to the sides, start braking
        if abs(attacker_aim_point[0]) > 0.3:
            attacker_dict['brake'] = True
            attacker_dict['acceleration'] = 0.3

        # can we backup? do we have a reason to do so?
        if self.frame - self.attacker_last_backup > self.backup_timeout and self.frame - self.reset_counter > self.backup_timeout:
            # did we just pass the puck? backup a smidge
            if attacker_aim_point[1] >= 0.95:
                print("ATTACKER: PASSED THE PUCK!")
                self._backup(10, -1 * attacker_dict['steer'], self.frame, 0)

            # heading into a goal... don't do that
            # second check is checking that front-z value > location-z value. ie, kart is heading into goal
            if abs(attacker['location'][2]) >= abs(self.goal[2]) and abs(attacker['front'][2]) > abs(attacker['location'][2]):
                print("ATTACKER HEADING INTO A GOAL")
                angle = 0.0
                if (attacker['location'][0] < 0.0 and attacker['location'][2] < 0.0) or (attacker['location'][0] > 0.0 and attacker['location'][2] > 0.0):
                    # top left or bottom right -> backup to left
                    angle = -1.0
                if (attacker['location'][0] > 0.0 and attacker['location'][2] < 0.0) or (attacker['location'][0] < 0.0 and attacker['location'][2] > 0.0):
                    # top right or bottom left -> backup to right
                    angle = 1.0
                self._backup(10, angle, self.frame, 0)

            # calculate dist between current pos and prev pos
            # if dist == 0, that probably means we reset.
            d = 1.0
            if self.attacker_last_location != None:
                d = np.linalg.norm(np.array([attacker['location'][0], attacker['location'][1]]) - np.array([self.attacker_last_location[0], self.attacker_last_location[1]]))
                if d == 0:
                    print("\n\n\tSETTING RESET COUNTER\n\n")
                    self.reset_counter = self.frame

            # might be stuck near wall. try backing up
            if attacker_vel < 1.0 and d != 0.0:
                print("ATTACKER STUCK!", self.attacker_starting_location, attacker['location'])
                self._backup(10, -1 * attacker_dict['steer'], self.frame, 0)

        # perform backup
        if self.attacker_backup_counter > 0:
            attacker_dict['brake'] = True
            attacker_dict['acceleration'] = 0.0
            attacker_dict['steer'] = self.attacker_backup_angle
            self.attacker_backup_counter -= 1

        # add attacker_point_point to memory
        if len(self.attacker_memory) == self.memory_limit:
            self.attacker_memory.pop(0)
        else:
            self.attacker_memory.append(attacker_aim_point)

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
        # Is this the aim point for gift (pickup item)?
        defender_aim_point = self.defender_planner(tensor).detach().cpu().numpy()
        defender_aim_point = defender_aim_point[0]

        # if in possession of the puck, ignore defender_aim_point
        # instead, start driving to goal
        # if self._in_poss_of_puck():
        #     print("IN POSSESSION OF THE PUCK!")
        #     defender_aim_point = self._to_image(self.goal, defender_cam['projection'], defender_cam['view'])
        if self._in_poss_of_puck(defender_aim_point, defender_loc_2d):
            print("DEFENDER: IN POSSESSION OF THE PUCK!")
            self.target_velocity = 10
            defender_aim_point = self._to_image(self.goal, np.array(defender_cam['projection']).T, np.array(defender_cam['view']).T)

        # if we have nitro and the aim_point is nearly straight ahead
        #if defender_aim_point[0] < 0.05 and defender_aim_point[0] > -0.05:
            #print("DEFENDER: HAS NITRO!")
            #defender_dict['nitro'] = True

        # compute acceleration
        defender_dict['acceleration'] = 1.0 if defender_vel < self.target_velocity else 0.0

        # compute steering
        steer_angle = 3 * defender_aim_point[0]
        defender_dict['steer'] = np.clip(steer_angle * 3, -1, 1)

        # if the puck is far to the sides, start braking
        if abs(defender_aim_point[0]) > 0.3:
            defender_dict['brake'] = True
            defender_dict['acceleration'] = 0.3

        # can we backup? do we have a reason to do so?
        #if self.frame - self.defender_last_backup > self.backup_timeout:
        if self.frame - self.defender_last_backup > self.backup_timeout and self.frame - self.reset_counter > self.backup_timeout:
            # did we just pass the puck? backup a smidge
            if defender_aim_point[1] >= 0.95:
                print("DEFENDER: PASSED THE PUCK!")
                self._backup(10, -1 * defender_dict['steer'], self.frame, 1)

            # heading into a goal... don't do that
            # second check is checking that front-z value > location-z value. ie, kart is heading into goal
            if abs(defender['location'][2]) >= abs(self.goal[2]) and abs(defender['front'][2]) > abs(defender['location'][2]):
                print("DEFENDER: HEADING INTO A GOAL")
                angle = 0.0
                if (defender['location'][0] < 0.0 and defender['location'][2] < 0.0) or (defender['location'][0] > 0.0 and defender['location'][2] > 0.0):
                    # top left or bottom right -> backup to left
                    angle = -1.0
                if (defender['location'][0] > 0.0 and defender['location'][2] < 0.0) or (defender['location'][0] < 0.0 and defender['location'][2] > 0.0):
                    # top right or bottom left -> backup to right
                    angle = 1.0
                self._backup(10, angle, self.frame, 1)

            # calculate dist between current pos and prev pos
            # if dist == 0, that probably means we reset.
            e = 1.0
            if self.defender_last_location != None:
                e = np.linalg.norm(np.array([defender['location'][0], defender['location'][1]]) - np.array([self.defender_last_location[0], self.defender_last_location[1]]))
                if e == 0:
                    print("\n\n\tDEFENDER: SETTING RESET COUNTER\n\n")
                    self.reset_counter = self.frame

            # might be stuck near wall. try backing up
            #if defender_vel < 1.0 and (self.defender_starting_location[0] != defender['location'][0] and self.defender_starting_location[2] != defender['location'][2]) and (abs(defender['location'][0]) > 40 or abs(defender['location'][2]) > 64):
            if defender_vel < 1.0 and e != 0.0:
                print("DEFENDER: STUCK!", self.defender_starting_location, defender['location'])
                self._backup(10, -1 * defender_dict['steer'], self.frame, 1)

        # perform backup
        if self.defender_backup_counter > 0:
            defender_dict['brake'] = True
            defender_dict['acceleration'] = 0.0
            defender_dict['steer'] = self.defender_backup_angle
            self.defender_backup_counter -= 1

        # add defender_point_point to memory
        if len(self.defender_memory) == self.memory_limit:
            self.defender_memory.pop(0)
        else:
            self.defender_memory.append(defender_aim_point)

        self.defender_last_location = defender['location']




        # debug logging
        for index, row in enumerate(self.ax):
            row.clear()
            if index == 0:
                row.imshow(Image.fromarray(player_image[0]))
                WH2 = np.array([400, 300]) / 2
                row.add_artist(plt.Circle(WH2*(1+attacker_loc_2d), 2, ec='b', fill=False, lw=1.5))
                row.add_artist(plt.Circle(WH2*(1+attacker_aim_point), 2, ec='yellow', fill=False, lw=1.5))
        #     if index == 1:
        #         row.imshow(Image.fromarray(player_image[1]))
        #         # col.imshow(self.k.render_data[0].instance)
        #         WH2 = np.array([400, 300]) / 2
        #         row.add_artist(plt.Circle(WH2*(1+defender_loc_2d), 2, ec='b', fill=False, lw=1.5))
        #         # col.add_artist(plt.Circle(WH2*(1+self._to_image(aim_point_world, proj, view)), 2, ec='r', fill=False, lw=1.5))
        #         row.add_artist(plt.Circle(WH2*(1+defender_aim_point), 2, ec='yellow', fill=False, lw=1.5))
        plt.pause(1e-10)

        self.frame += 1

        # return [attacker_dict, defender_dict]
        #return [attacker_dict, dict(acceleration=1, steer=0)]   #original code
        return [attacker_dict, defender_dict]
