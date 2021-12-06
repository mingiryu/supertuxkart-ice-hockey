import pystk

backup_counter = 0
backup_angle = 0
frame = 0
last_rescue = 0
RESCUE_TIMEOUT = 30

def defender_control(gift_aim_point, op_kart1_aim_point=None, op_kart2_aim_point=None, curr_def_vel, def_steer_gain=3, def_target_vel=15):
    import numpy as np
    action = pystk.Action()
    global backup_counter
    global backup_angle
    global frame
    global last_rescue
    global RESCUE_TIMEOUT

    # if we have nitro and the aim_point is nearly straight ahead
    if gift_aim_point[0] < 0.05 and gift_aim_point[0] > -0.05:
        action.nitro = True

    # compute acceleration
    if current_def_vel < def_target_vel:
        action.acceleration = 1.0
    else:
        action.acceleration = 0.0

    # compute steering
    def_steer_angle = def_steer_gain * gift_aim_point[0]
    action.steer = np.clip(def_steer_angle * def_steer_gain, -1, 1)

    # if the gift is far to the sides, start braking
    if abs(gift_aim_point[0]) > 0.3:
        action.brake = True
        action.acceleration = 0.1

    # might be stuck. try backing up
    if def_current_vel < 1.0 and frame - last_rescue > RESCUE_TIMEOUT:
        last_rescue = frame
        backup_counter = 10
        backup_angle = -1 * action.steer

    # compute backup
    if backup_counter > 0:
        action.brake = True
        action.acceleration = 0.0
        action.steer = backup_angle
        backup_counter -= 1

    frame += 1

    # If gift is found, increment the counter variable.
    # If all gifts are found, now start attacking the opponent karts 1 or 2

    return action





def control(aim_point, current_vel, steer_gain=3, skid_thresh=0.5, target_vel=15, puck_in_view=None):
    import numpy as np
    action = pystk.Action()
    global backup_counter
    global backup_angle
    global frame
    global last_rescue
    global RESCUE_TIMEOUT

    # if we have nitro and the aim_point is nearly straight ahead
    if aim_point[0] < 0.05 and aim_point[0] > -0.05:
        action.nitro = True

    # compute acceleration
    action.acceleration = 1.0 if current_vel < target_vel else 0.0

    # compute steering
    steer_angle = steer_gain * aim_point[0]
    action.steer = np.clip(steer_angle * steer_gain, -1, 1)

    # if the puck is far to the sides, start braking
    if abs(aim_point[0]) > 0.3:
        action.brake = True
        action.acceleration = 0.1

    # might be stuck. try backing up
    if current_vel < 1.0 and frame - last_rescue > RESCUE_TIMEOUT:
        last_rescue = frame
        backup_counter = 10
        backup_angle = -1 * action.steer

    # compute backup
    if backup_counter > 0:
        action.brake = True
        action.acceleration = 0.0
        action.steer = backup_angle
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
        print("RESULTS:", pytux.state.soccer.score)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
