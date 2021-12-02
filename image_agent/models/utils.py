import numpy as np
import pystk
import pickle
from torchvision.transforms import functional as F


from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from . import dense_transforms

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
DATASET_PATH = 'test_data'

def _to_image(x, proj, view):
    p = proj @ view @ np.array(list(x) + [1])
    return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)


class SuperTuxDataset(Dataset):
    # model = 0 is a planner, model = 1 is a detector
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), model=0):
        from PIL import Image
        from glob import glob
        from os import path
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2)
        self.data = []
        with (open(dataset_path, "rb")) as openfile:
            while True:
                try:
                    d = pickle.load(openfile)

                    # gathering necessary info
                    t1_imgs = d['team1_images']
                    t2_imgs = d['team2_images']
                    puck = d['soccer_state']['ball']['location']
                    k1 = d['team1_state'][0]
                    k2 = d['team1_state'][1]
                    k3 = d['team2_state'][0]
                    k4 = d['team2_state'][1]

                    # gathering puck positions for all karts
                    puck1 = _to_image(puck, np.array(k1['camera']['projection']).T, np.array(k1['camera']['view']).T)
                    puck2 = _to_image(puck, np.array(k2['camera']['projection']).T, np.array(k2['camera']['view']).T)
                    puck3 = _to_image(puck, np.array(k3['camera']['projection']).T, np.array(k3['camera']['view']).T)
                    puck4 = _to_image(puck, np.array(k4['camera']['projection']).T, np.array(k4['camera']['view']).T)

                    # gathering enemy kart locations for all karts
                    ek1_1 = _to_image(k3['kart']['location'], np.array(k1['camera']['projection']).T, np.array(k1['camera']['view']).T) # (x,y) in the view of K1
                    ek2_1 = _to_image(k4['kart']['location'], np.array(k1['camera']['projection']).T, np.array(k1['camera']['view']).T) # (x,y) in the view of K1

                    # gathering images for all karts
                    image1 =  Image.fromarray(t1_imgs[0])
                    image2 =  Image.fromarray(t1_imgs[1])
                    image3 =  Image.fromarray(t2_imgs[0])
                    image4 =  Image.fromarray(t2_imgs[1])

                    # gathering instances for all karts
                    instance1 = d['team1_instances'][0]
                    instance2 = d['team1_instances'][1]
                    instance3 = d['team2_instances'][0]
                    instance4 = d['team2_instances'][1]

                    # checking existence of puck
                    exist1 = False
                    exist2 = False
                    exist3 = False
                    exist4 = False
                    if any(8 in x for x in instance1):
                        exist1 = True
                    if any(8 in x for x in instance2):
                        exist2 = True
                    if any(8 in x for x in instance3):
                        exist3 = True
                    if any(8 in x for x in instance4):
                        exist4 = True

                    # if puck doesn't exist on screen, set the aim point behind us.
                    # controller should handle this case
                    if not exist1:
                        val = 0.8 if puck1[0] > 0 else -0.8
                        puck1 = np.array([val, 1])
                    if not exist2:
                        val = 0.8 if puck2[0] > 0 else -0.8
                        puck2 = np.array([val, 1])
                    if not exist3:
                        val = 0.8 if puck3[0] > 0 else -0.8
                        puck3 = np.array([val, 1])
                    if not exist4:
                        val = 0.8 if puck4[0] > 0 else -0.8
                        puck4 = np.array([val, 1])

                    # building labels for detector
                    # 1 when pixel is 8 (projectile/puck) else 0
                    instance1[instance1 != 8] = 0
                    instance2[instance2 != 8] = 0
                    instance3[instance3 != 8] = 0
                    instance4[instance4 != 8] = 0

                    # logging help
                    counter = 0
                    WH2 = np.array([400, 300]) / 2
                    for row in ax:
                        for index, col in enumerate(row):
                            # print(counter, index)
                            col.clear()
                            if counter == 0 and index == 0:
                                # col.imshow(t1_imgs[0])
                                col.imshow(instance1)
                                col.add_artist(plt.Circle(WH2*(1+_to_image(k1['kart']['location'], np.array(k1['camera']['projection']).T, np.array(k1['camera']['view']).T)), 2, ec='b', fill=False, lw=1.5))
                                col.add_artist(plt.Circle(WH2*(1+puck1), 2, ec='r', fill=False, lw=1.5))
                            if counter == 0 and index == 1:
                                # col.imshow(t2_imgs[0])
                                col.imshow(instance3)
                                col.add_artist(plt.Circle(WH2*(1+_to_image(k3['kart']['location'], np.array(k3['camera']['projection']).T, np.array(k3['camera']['view']).T)), 2, ec='b', fill=False, lw=1.5))
                                col.add_artist(plt.Circle(WH2*(1+puck3), 2, ec='r', fill=False, lw=1.5))
                            if counter == 1 and index == 0:
                                # col.imshow(t1_imgs[1])
                                col.imshow(instance2)
                                col.add_artist(plt.Circle(WH2*(1+_to_image(k2['kart']['location'], np.array(k2['camera']['projection']).T, np.array(k2['camera']['view']).T)), 2, ec='b', fill=False, lw=1.5))
                                col.add_artist(plt.Circle(WH2*(1+puck2), 2, ec='r', fill=False, lw=1.5))
                            if counter == 1 and index == 1:
                                # col.imshow(t2_imgs[1])
                                col.imshow(instance4)
                                col.add_artist(plt.Circle(WH2*(1+_to_image(k4['kart']['location'], np.array(k4['camera']['projection']).T, np.array(k4['camera']['view']).T)), 2, ec='b', fill=False, lw=1.5))
                                col.add_artist(plt.Circle(WH2*(1+puck4), 2, ec='r', fill=False, lw=1.5))
                        counter += 1
                    plt.pause(1e-3)
                    # print(tensor1.shape, tensor2.shape, tensor3.shape, tensor4.shape)
                    # print(puck_1, puck_2, puck_3, puck_4)
                    # print(exist1, exist2, exist3, exist4)
                    # print()

                    # append accumulated info to self.data
                    if model == 0:
                        # planner
                        if exist1:
                            self.data.append((image1, np.array(puck1).astype(np.float32)))
                        if exist2:
                            self.data.append((image2, np.array(puck2).astype(np.float32)))
                        if exist3:
                            self.data.append((image3, np.array(puck3).astype(np.float32)))
                        if exist4:
                            self.data.append((image4, np.array(puck4).astype(np.float32)))

                    if model == 1:
                        # detector
                        if exist1:
                            self.data.append((image1, np.array(instance1).astype(np.float32)))
                        if exist2:
                            self.data.append((image2, np.array(instance2).astype(np.float32)))
                        if exist3:
                            self.data.append((image3, np.array(instance3).astype(np.float32)))
                        if exist4:
                            self.data.append((image4, np.array(instance4).astype(np.float32)))
                except EOFError:
                    break
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data


def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128, model=0):
    dataset = SuperTuxDataset(dataset_path, transform=transform, model=model)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


class PyTux:
    _singleton = None

    def __init__(self, screen_width=400, screen_height=300):
        assert PyTux._singleton is None, "Cannot create more than one pytux object"
        PyTux._singleton = self
        self.config = pystk.GraphicsConfig.hd()
        self.config.screen_width = screen_width
        self.config.screen_height = screen_height
        pystk.init(self.config)
        self.k = None

    @staticmethod
    def _point_on_track(distance, track, offset=0.0):
        """
        Get a point at `distance` down the `track`. Optionally applies an offset after the track segment if found.
        Returns a 3d coordinate
        """
        node_idx = np.searchsorted(track.path_distance[..., 1],
                                   distance % track.path_distance[-1, 1]) % len(track.path_nodes)
        d = track.path_distance[node_idx]
        x = track.path_nodes[node_idx]
        t = (distance + offset - d[0]) / (d[1] - d[0])
        return x[1] * t + x[0] * (1 - t)

    @staticmethod
    def _to_image(x, proj, view):
        p = proj @ view @ np.array(list(x) + [1])
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

    def rollout(self, track, controller, planner=None, max_frames=1000, verbose=False, data_callback=None):
        """
        Play a level (track) for a single round.
        :param track: Name of the track
        :param controller: low-level controller, see controller.py
        :param planner: high-level planner, see planner.py
        :param max_frames: Maximum number of frames to play for
        :param verbose: Should we use matplotlib to show the agent drive?
        :param data_callback: Rollout calls data_callback(time_step, image, 2d_aim_point) every step, used to store the
                              data
        :return: Number of steps played
        """
        if self.k is not None and self.k.config.track == track:
            self.k.restart()
            self.k.step()
        else:
            if self.k is not None:
                self.k.stop()
                del self.k
            config = pystk.RaceConfig()
            config.mode = config.RaceMode.SOCCER
            config.step_size = 0.1
            config.num_kart = 2
            config.track = track
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
            config.players[0].team = 0
            config.players.append(
                    pystk.PlayerConfig("", pystk.PlayerConfig.Controller.AI_CONTROL, 1))

            self.k = pystk.Race(config)
            self.k.start()
            self.k.step()

        state = pystk.WorldState()
        track = pystk.Track()

        last_rescue = 0

        if verbose:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1)

        for t in range(max_frames):
            state.update()

            kart = state.players[0].kart
            # if np.isclose(kart.overall_distance / track.length, 1.0, atol=2e-3):
            #     if verbose:
            #         print("Finished at t=%d" % t)
            #     break

            proj = np.array(state.players[0].camera.projection).T
            view = np.array(state.players[0].camera.view).T

            # aim_point_world = self._point_on_track(kart.distance_down_track+TRACK_OFFSET, track)

            # aim_point_image for circle on screen for puck
            aim_point_world = state.soccer.ball.location
            aim_point_image = self._to_image(aim_point_world, proj, view)

            # aim_point_image for circle on screen for enemy kart
            # aim_point_world1 = state.players[1].kart.location
            # aim_point_image1 = self._to_image(aim_point_world1, proj, view)

            shifted = self.k.render_data[0].instance >> pystk.object_type_shift
            exists = True
            if any(8 in x for x in shifted):
                exists = True

            if not exists and abs(aim_point_image[0]) != 1:
                val = 1.0 if aim_point_image[0] > 0 else -1.0
                aim_point_image = np.array([val, 0])

            if data_callback is not None:
                data_callback(t, np.array(self.k.render_data[0].image), [aim_point_image])

            if planner:
                image = np.array(self.k.render_data[0].image)
                aim_point_image = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()

            current_vel = np.linalg.norm(kart.velocity)
            action = controller(aim_point_image, current_vel, puck_in_view=exists)

            # if current_vel < 1.0 and t - last_rescue > RESCUE_TIMEOUT:
            #     last_rescue = t
            #     action.rescue = True

            if verbose:
                ax.clear()
                ax.imshow(self.k.render_data[0].image)
                # ax.imshow(self.k.render_data[0].instance)
                WH2 = np.array([self.config.screen_width, self.config.screen_height]) / 2
                ax.add_artist(plt.Circle(WH2*(1+self._to_image(kart.location, proj, view)), 2, ec='b', fill=False, lw=1.5))
                ax.add_artist(plt.Circle(WH2*(1+self._to_image(aim_point_world, proj, view)), 2, ec='r', fill=False, lw=1.5))
                ax.add_artist(plt.Circle(WH2*(1+aim_point_image), 2, ec='yellow', fill=False, lw=1.5))
                if planner:
                    ap = aim_point_world
                    ax.add_artist(plt.Circle(WH2*(1+aim_point_image), 2, ec='g', fill=False, lw=1.5))
                plt.pause(1e-3)

            self.k.step(action)
            t += 1
        return t, kart.overall_distance

    def close(self):
        """
        Call this function, once you're done with PyTux
        """
        if self.k is not None:
            self.k.stop()
            del self.k
        pystk.clean()


if __name__ == '__main__':
    from .controller import control
    from argparse import ArgumentParser
    from os import makedirs


    def noisy_control(aim_pt, vel, puck_in_view=None):
        return control(aim_pt + np.random.randn(*aim_pt.shape) * aim_noise,
                       vel + np.random.randn() * vel_noise, puck_in_view=puck_in_view)


    parser = ArgumentParser("Collects a dataset for the high-level planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-o', '--output', default=DATASET_PATH)
    parser.add_argument('-n', '--n_images', default=1000, type=int)
    parser.add_argument('-m', '--steps_per_game', default=1000, type=int)
    parser.add_argument('--aim_noise', default=0.1, type=float)
    parser.add_argument('--vel_noise', default=5, type=float)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    try:
        makedirs(args.output)
    except OSError:
        pass
    pytux = PyTux()

    for track in args.track:
        n, images_per_game = 0, args.n_images
        aim_noise, vel_noise = 0, 0

        def collect(_, im, pt):
            from PIL import Image
            from os import path
            global n
            id = n if n < images_per_game else np.random.randint(0, n + 1)
            if id < images_per_game:
                fn = path.join(args.output, track + '_%05d' % id)
                Image.fromarray(im).save(fn + '.png')
                with open(fn + '.csv', 'w') as f:
                    for i in range(len(pt)):
                        f.write('%0.1f,%0.1f' % tuple(pt[i]))
                        if i != len(pt)-1:
                            f.write(',')
            n += 1

        while n < args.steps_per_game:
            # print("N:", n)
            steps, how_far = pytux.rollout(track, noisy_control, max_frames=1000, verbose=args.verbose, data_callback=collect)
            # print(steps, how_far)
            # Add noise after the first round
            aim_noise, vel_noise = args.aim_noise, args.vel_noise

    pytux.close()
