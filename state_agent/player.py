class Team:
    agent_type = 'state'

    def __init__(self):
        """
        TODO: Load your agent here. Load network parameters, and other parts of our model
        We will call this function with default arguments only
        """
        # creating vars to store team color, number of players, goal_point
        self.team = None
        self.num_players = None
        self.goal_line = None

    def _get_quadrant(self, coords1, coords2):
        """
            Returns the quadrant of coords1 wrt coords2
            Useful for determining the way the kart is facing
             (-,-)
                                     |
                                     |
                     Q2              |           Q1
                                     |
             ------------------------O-------------------------- 
                                     |
                                     |
                     Q3              |           Q4
                                     |
                                     |
                                                            (+,+)

            :param coords1: List of coords [x, y(not used), z]
            :param coords2: List of coords that acts as the anchor [x, y(not used), z]
            :return: Int specifying the quadrant the coords resides in
        """
        if(coords1[0] < coords2[0] and coords1[2] < coords2[2]):
            return 2
        elif(coords1[0] < coords2[0] and coords1[2] > coords2[2]):
            return 3
        elif(coords1[0] > coords2[0] and coords1[2] > coords2[2]):
            return 4
        else:
            return 1

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
        # set our class vars and return an array of kart names == num_players
        self.team, self.num_players = team, num_players
        print("TEAM:", team)
        if self.team == 0:
            self.goal_line = [0, 0,  80]
        else:
            self.goal_line = [0, 0, -80]
        return ['puffy'] * num_players

    def act(self, player_state, opponent_state, soccer_state):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             You can ignore the camera here.
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param opponent_state: same as player_state just for other team

        :param soccer_state: dict  Mostly used to obtain the puck location
                             ball:  Puck information
                               - location: float3 world location of the puck

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
        #   Attacker Code   #
        #####################
        #   ball in front?
        #       move ball to goal => aim_point = (0,78) or (0,-78)
        #   else
        #       move to pos with ball in front => aim_point = ball position
        dict1 = {'acceleration':0.0, 'steer':0.0, 'brake': False, 'fire': False}

        # calculate euclidean distance between front of kart and ball
        attacker_loc = player_state[0]['kart']['location']
        attacker_front = player_state[0]['kart']['front']
        attacker_vel = player_state[0]['kart']['velocity']
        aim_point = soccer_state['ball']['location']
        kart_q = self._get_quadrant(attacker_front, attacker_loc)
        ball_q = self._get_quadrant(aim_point, attacker_loc)

        # print("KART AND PUCK Q's:", kart_q, ball_q, (attacker_front[0], attacker_front[2])


        # 1. first off, which way are we facing? are we facing the puck?
        # ans: orient ourselves
        print("ORIENTATION:", kart_q, ball_q, (attacker_front[0], attacker_front[2]), (aim_point[0], aim_point[2]))
        if kart_q == 1 and ball_q == 3 or kart_q == 3 and ball_q == 1:
            # ball is behind us. turn around
            print("TURNING AROUND")
            dict1['brake'] = True
            dict1['steer'] = -1.0
        elif kart_q == 2 and ball_q == 4 or kart_q == 4 and ball_q == 2:
            # ball is behind us. turn around
            print("TURNING AROUND")
            dict1['brake'] = True
            dict1['steer'] = 1.0
        elif kart_q == 1 and ball_q == 2 or kart_q == 4 and ball_q == 3:
            print("TURNING RIGHT")
            # dict1['acceleration'] = 1.0
            dict1['steer'] = -1.0
        elif kart_q == 2 and ball_q == 1 or kart_q == 3 and ball_q == 4:
            print("TURNING LEFT")
            # dict1['acceleration'] = 1.0
            dict1['steer'] = 1.0
        # 2. now we're oriented and facing the ball. how do we move towards it?
        # ans: use coords to steer towards ball
        else:
            dict1['acceleration'] = 0.8
            # is the ball on the left or the right?
            if aim_point[0] - attacker_front[0] > 0:
                print("MOVING TOWARDS BALL RIGHT")
                dict1['steer'] = 0.8
            elif aim_point[0] - attacker_front[0] < 0:
                print("MOVING TOWARDS BALL LEFT")
                dict1['steer'] = 0.8


        # in control of the puck. now aim it towards the goal

        return [dict1, dict(acceleration=1, steer=0)]
