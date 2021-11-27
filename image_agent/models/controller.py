import pystk


def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    if aim_point[0] < -0.1:
        action.steer = -0.8
        action.brake = True
    elif aim_point[0] > 0.1:
        action.steer = 0.8
        action.brake = True
    else:
        action.steer = 0.0
        action.brake = False

    # if we have nitro and the aim_point is nearly straight ahead
    if aim_point[0] < 0.05 and aim_point[0] > -0.05:
        action.nitro = True

    # aim_point is far to the sides, let's be a drift king
    if abs(aim_point[0]) > 0.3:
        action.drift = True
        action.acceleration = 0.1
    else:
        action.acceleration = 1.0 if current_vel < 30.0 else 0.0
    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
