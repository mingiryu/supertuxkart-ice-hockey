import pickle
import torch
import numpy as np
from .models import load_planner, load_detector, control, dense_transforms
from torchvision.transforms import functional as F

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
        self.team = None
        self.num_players = None
        self.goal = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.attacker_planner = load_planner().to(self.device)
        self.attacker_detector = load_detector().to(self.device)
        self.attacker_backup = 0
        self.attacker_starting_location = None
        self.attacker_memory = [] # pop and push every frame

    def _to_image(self, world_coord, proj, view):
        """
        :param world_coord: The [x,y,z] coord in the world to convert to a 2D point
        :param proj: The projection of the kart's camera
        :param view: The view of the kart's camera
        :return: A 2D numpy array of the world_coord translated based on the kart's camera projection and view
        """
        p = proj @ view @ np.array(list(world_coord) + [1])
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

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
        attacker_loc_2d = self._to_image(attacker['location'], attacker_cam['projection'], attacker_cam['view'])
        tensor = F.to_tensor(player_image[0])
        tensor = tensor[None, :].to(self.device)
        if self.attacker_starting_location == None:
            self.attacker_starting_location = attacker['location']

        # predicting attacker_aim_point
        attacker_aim_point = self.attacker_planner(tensor).detach().cpu().numpy()
        # attacker_heatmap = self.attacker_detector(tensor).detach().cpu().numpy()

        attacker_aim_point = attacker_aim_point[0]
        attacker_dist = np.linalg.norm(attacker_loc_2d - attacker_aim_point)
        # print(attacker_dist)

        # if we have nitro and the aim_point is nearly straight ahead
        if attacker_aim_point[0] < 0.05 and attacker_aim_point[0] > -0.05:
            attacker_dict['nitro'] = True

        # compute acceleration
        attacker_dict['acceleration'] = 1.0 if attacker_vel < 10.0 else 0.0

        # compute steering
        steer_angle = 3 * attacker_aim_point[0]
        attacker_dict['steer'] = 1 if steer_angle * 3 > 0 else -1#np.clip(steer_angle * 3, -1, 1)
        print(attacker_aim_point, 1 if steer_angle * 3 > 0 else -1)

        # predicting that the puck is behind us. brake and backup.
        if attacker_aim_point[0] < -0.1 and attacker_aim_point[1] > 0.7:
            print("BRAKING LEFT")
            attacker_dict['steer'] = -1 * attacker_dict['steer']
            attacker_dict['acceleration'] = 0.0
            attacker_dict['brake'] = True
        elif attacker_aim_point[0] > 0.1 and attacker_aim_point[1] > 0.7:
            print("BRAKING RIGHT")
            attacker_dict['steer'] = -1 * attacker_dict['steer']
            attacker_dict['acceleration'] = 0.0
            attacker_dict['brake'] = True

        # might be stuck. back up for 5 frames
        # print(attacker_vel)
        if attacker_vel < 0.1 and attacker['location'][0] != self.attacker_starting_location[0] and attacker['location'][2] != self.attacker_starting_location[2]:
            self.attacker_backup = 7

        if self.attacker_backup > 0:
            attacker_dict['brake'] = True
            attacker_dict['acceleration'] = 0.0
            self.attacker_backup -= 1


        #####################
        #   DEFENDER CODE   #
        #####################

        # TODO: Write defender control code

        return [attacker_dict, dict(acceleration=1, steer=0)]
