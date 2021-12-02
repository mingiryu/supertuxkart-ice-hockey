import pystk

backup_counter = 0
frame = 0

def control(aim_point, current_vel, steer_gain=3, skid_thresh=0.5, target_vel=20, puck_in_view=None):
    import numpy as np
    action = pystk.Action()
    global backup_counter
    global frame
    # print(aim_point)

    # if we have nitro and the aim_point is nearly straight ahead
    if aim_point[0] < 0.05 and aim_point[0] > -0.05:
        action.nitro = True

    # aim_point is far to the sides, let's be a drift king
    if abs(aim_point[0]) > 0.3:
        action.drift = True
        action.acceleration = 0.1
    else:
        action.acceleration = 1.0 if current_vel < 30.0 else 0.0

    steer_angle = steer_gain * aim_point[0]
    # Compute steering
    action.steer = np.clip(steer_angle * steer_gain, -1, 1)

    if aim_point[0] < -0.1 and aim_point[1] > 0.7:
        print("TURNING LEFT")
        action.steer = -1 * action.steer
        action.acceleration = 0.0
        action.brake = True
    elif aim_point[0] > 0.1 and aim_point[1] > 0.7:
        print("TURNING RIGHT")
        action.steer = -1 * action.steer
        action.acceleration = 0.0
        action.brake = True

    print(current_vel)
    # might be stuck. back up
    if current_vel < 0.1 and frame > 5:
        backup_counter = 5

    if backup_counter > 0:
        action.brake = True
        action.acceleration = 0.0
        backup_counter -= 1


    frame += 1
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
