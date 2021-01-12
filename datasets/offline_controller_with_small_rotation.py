""" Exhaustive BFS and Offline Controller. """

import importlib
from collections import deque
import json
import copy
import time
import random
import os
import platform

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from .base_controller import BaseController

class ThorAgentState:
    """ Representation of a simple state of a Thor Agent which includes
        the position, horizon and rotation. """

    def __init__(self, x, y, z, rotation, horizon):
        self.x = round(x, 2)
        self.y = y
        self.z = round(z, 2)
        self.rotation = round(rotation)
        self.horizon = round(horizon)

    @classmethod
    def get_state_from_evenet(cls, event, forced_y=None):
        """ Extracts a state from an event. """
        state = cls(
            x=event.metadata["agent"]["position"]["x"],
            y=event.metadata["agent"]["position"]["y"],
            z=event.metadata["agent"]["position"]["z"],
            rotation=event.metadata["agent"]["rotation"]["y"],
            horizon=event.metadata["agent"]["cameraHorizon"],
        )
        if forced_y != None:
            state.y = forced_y
        return state

    def __eq__(self, other):
        """ If we check for exact equality then we get issues.
            For now we consider this 'close enough'. """
        if isinstance(other, ThorAgentState):
            return (
                self.x == other.x
                and
                # self.y == other.y and
                self.z == other.z
                and self.rotation == other.rotation
                and self.horizon == other.horizon
            )
        return NotImplemented

    def __str__(self):
        """ Get the string representation of a state. """
        """
        return '{:0.2f}|{:0.2f}|{:0.2f}|{:d}|{:d}'.format(
            self.x,
            self.y,
            self.z,
            round(self.rotation),
            round(self.horizon)
        )
        """
        return "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
            self.x, self.z, round(self.rotation), round(self.horizon)
        )

    def position(self):
        """ Returns just the position. """
        return dict(x=self.x, y=self.y, z=self.z)

class OfflineControllerWithSmallRotationEvent:
    """ A stripped down version of an event. Only contains lastActionSuccess, sceneName,
        and optionally state and frame. Does not contain the rest of the metadata. """

    def __init__(self, last_action_success, scene_name, state=None, frame=None, score=None):
        self.metadata = {
            "lastActionSuccess": last_action_success,
            "sceneName": scene_name,
        }
        if state is not None:
            self.metadata["agent"] = {}
            self.metadata["agent"]["position"] = state.position()
            self.metadata["agent"]["rotation"] = {
                "x": 0.0,
                "y": state.rotation,
                "z": 0.0,
            }
            self.metadata["agent"]["cameraHorizon"] = state.horizon
        self.frame = frame
        self.score = score


class OfflineControllerWithSmallRotation(BaseController):
    """ A stripped down version of the controller for non-interactive settings.
        Only allows for a few given actions. Note that you must use the
        ExhaustiveBFSController to first generate the data used by OfflineControllerWithSmallRotation.
        Data is stored in offline_data_dir/<scene_name>/.

        Can swap the metadata.json for a visible_object_map.json. A script for generating
        this is coming soon. If the swap is made then the OfflineControllerWithSmallRotation is faster and
        self.using_raw_metadata will be set to false.

        Additionally, images.hdf5 may be swapped out with ResNet features or anything
        that you want to be returned for event.frame. """

    def __init__(
        self,
        grid_size=0.25,
        fov=100,
        offline_data_dir="../thordata/mixed_offline_data",
        grid_file_name="grid.json",
        graph_file_name="graph.json",
        metadata_file_name="visible_object_map.json",
        # metadata_file_name='metadata.json',
        images_file_name="images.hdf5",
        debug_mode=True,
        actions=["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"],
        visualize=False,
        local_executable_path=None,
    ):

        super(OfflineControllerWithSmallRotation, self).__init__()
        self.grid_size = grid_size
        self.offline_data_dir = offline_data_dir
        self.grid_file_name = grid_file_name
        self.graph_file_name = graph_file_name
        self.metadata_file_name = metadata_file_name
        self.images_file_name = images_file_name
        self.scores_file_name = 'resnet50_score.hdf5'
        self.grid = None
        self.graph = None
        self.metadata = None
        self.images = None
        self.scores = None
        self.using_raw_metadata = True
        self.actions = actions
        # Allowed rotations.
        self.rotations = [0, 45, 90, 135, 180, 225, 270, 315]
        # Allowed horizons.
        self.horizons = [0, 30]
        self.debug_mode = debug_mode
        self.fov = fov

        self.local_executable_path = local_executable_path

        self.y = None

        self.last_event = None


        self.visualize = visualize

        self.scene_name = None
        self.state = None
        self.last_action_success = True

        self.h5py = importlib.import_module("h5py")
        self.nx = importlib.import_module("networkx")
        self.json_graph_loader = importlib.import_module("networkx.readwrite")

    def start(self):
        pass

    def get_full_state(self, x, y, z, rotation=0.0, horizon=0.0):
        return ThorAgentState(x, y, z, rotation, horizon)

    def get_state_from_str(self, x, z, rotation=0.0, horizon=0.0):
        return ThorAgentState(x, self.y, z, rotation, horizon)

    def reset(self, scene_name=None):

        if scene_name is None:
            scene_name = "FloorPlan28_physics"

        if scene_name != self.scene_name:
            self.scene_name = scene_name
            with open(
                os.path.join(
                    self.offline_data_dir, self.scene_name, self.grid_file_name
                ),
                "r",
            ) as f:
                self.grid = json.load(f)
            with open(
                os.path.join(
                    self.offline_data_dir, self.scene_name, self.graph_file_name
                ),
                "r",
            ) as f:
                graph_json = json.load(f)
            self.graph = self.json_graph_loader.node_link_graph(
                graph_json
            ).to_directed()
            with open(
                os.path.join(
                    self.offline_data_dir, self.scene_name, self.metadata_file_name
                ),
                "r",
            ) as f:
                self.metadata = json.load(f)
                # Determine if using the raw metadata, which is structured as a dictionary of
                # state -> metatdata. The alternative is a map of obj -> states where object is visible.
                key = next(iter(self.metadata.keys()))
                try:
                    float(key.split("|")[0])
                    self.using_raw_metadata = True
                except ValueError:
                    self.using_raw_metadata = False

            if self.images is not None:
                self.images.close()
            self.images = self.h5py.File(
                os.path.join(
                    self.offline_data_dir, self.scene_name, self.images_file_name
                ),
                "r",
            )

            if self.scores is not None:
                self.scores.close()
            self.scores = self.h5py.File(
                os.path.join(
                    self.offline_data_dir, self.scene_name, self.scores_file_name
                ),
                "r",
            )

        self.state = self.get_full_state(
            **self.grid[0], rotation=random.choice(self.rotations)
        )
        self.y = self.state.y
        self.last_action_success = True
        self.last_event = self._successful_event()


    def randomize_state(self):
        self.state = self.get_state_from_str(
            *[float(x) for x in random.choice(list(self.images.keys())).split("|")]
        )
        self.state.horizon = 0
        self.last_action_success = True
        self.last_event = self._successful_event()


    def back_to_start(self, start):
        self.state = start

    def get_next_state(self, state, action, copy_state=False):
        """ Guess the next state when action is taken. Note that
            this will not predict the correct y value. """
        if copy_state:
            next_state = copy.deepcopy(state)
        else:
            next_state = state
        if action == "MoveAhead":
            if next_state.rotation == 0:
                next_state.z += self.grid_size
            elif next_state.rotation == 90:
                next_state.x += self.grid_size
            elif next_state.rotation == 180:
                next_state.z -= self.grid_size
            elif next_state.rotation == 270:
                next_state.x -= self.grid_size
            elif next_state.rotation == 45:
                next_state.z += self.grid_size
                next_state.x += self.grid_size
            elif next_state.rotation == 135:
                next_state.z -= self.grid_size
                next_state.x += self.grid_size
            elif next_state.rotation == 225:
                next_state.z -= self.grid_size
                next_state.x -= self.grid_size
            elif next_state.rotation == 315:
                next_state.z += self.grid_size
                next_state.x -= self.grid_size
            else:
                raise Exception("Unknown Rotation")
        elif action == "RotateRight":
            next_state.rotation = (next_state.rotation + 45) % 360
        elif action == "RotateLeft":
            next_state.rotation = (next_state.rotation - 45) % 360
        elif action == "LookUp":
            if abs(next_state.horizon) <= 1:
                return None
            next_state.horizon = next_state.horizon - 30
        elif action == "LookDown":
            if abs(next_state.horizon - 60) <= 1 or abs(next_state.horizon - 30) <= 1:
                return None
            next_state.horizon = next_state.horizon + 30
        return next_state

    def step(self, action, raise_for_failure=False):

        if "action" not in action or action["action"] not in self.actions:
            if action["action"] == "Initialize":
                return
            raise Exception("Unsupported action.")

        action = action["action"]

        next_state = self.get_next_state(self.state, action, True)

        if next_state is not None:
            next_state_key = str(next_state)
            neighbors = self.graph.neighbors(str(self.state))

            if next_state_key in neighbors:
                self.state = self.get_state_from_str(
                    *[float(x) for x in next_state_key.split("|")]
                )
                self.last_action_success = True
                event = self._successful_event()

                self.last_event = event
                return event

        self.last_action_success = False
        self.last_event.metadata["lastActionSuccess"] = False
        return self.last_event

    def shortest_path(self, source_state, target_state):
        return self.nx.shortest_path(self.graph, str(source_state), str(target_state))


    def shortest_path_to_target(self, source_state, objId, get_plan=False):
        """ Many ways to reach objId, which one is best? """
        states_where_visible = []
        if self.using_raw_metadata:
            for s in self.metadata:
                objects = self.metadata[s]["objects"]
                visible_objects = [o["objectId"] for o in objects if o["visible"]]
                if objId in visible_objects:
                    states_where_visible.append(s)
        else:
            states_where_visible = self.metadata[objId]

        # transform from strings into states
        states_where_visible = [
            self.get_state_from_str(*[float(x) for x in str_.split("|")])
            for str_ in states_where_visible
        ]

        best_path = None
        best_path_len = 0
        for t in states_where_visible:
            path = self.shortest_path(source_state, t)
            if len(path) < best_path_len or best_path is None:
                best_path = path
                best_path_len = len(path)
        best_plan = []

        if get_plan:
            best_plan = None

        return best_path, best_path_len, best_plan


    def object_is_visible(self, objId):
        if self.using_raw_metadata:
            objects = self.metadata[str(self.state)]["objects"]
            visible_objects = [o["objectId"] for o in objects if o["visible"]]
            return objId in visible_objects
        else:
            return str(self.state) in self.metadata[objId]

    def _successful_event(self):
        return OfflineControllerWithSmallRotationEvent(
            self.last_action_success, self.scene_name, self.state, self.get_image(), self.get_score()
        )

    def get_image(self):
        return self.images[str(self.state)][:]

    def get_score(self):
        return self.scores[str(self.state)][:]

    def all_objects(self):
        if self.using_raw_metadata:
            return [o["objectId"] for o in self.metadata[str(self.state)]["objects"]]
        else:
            return self.metadata.keys()

def env_create(n, queue):
    env = OfflineControllerWithSmallRotation()
    ss = ['FloorPlan1_physics', 'FloorPlan2_physics', 'FloorPlan3_physics',
    'FloorPlan4_physics', 'FloorPlan5_physics', 'FloorPlan6_physics',
    'FloorPlan7_physics', 'FloorPlan8_physics', 'FloorPlan9_physics',
    'FloorPlan10_physics', 'FloorPlan11_physics', 'FloorPlan12_physics',
    ]
    env.reset(random.choice(ss))
    for _ in range(n):
        env.step(dict(action = 'RotateLeft'))
        queue.put(1)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    from tqdm import tqdm
    result_queue = mp.Queue()
    processes = []
    

    threads = 1
    tim = 500000

    for thread_id in range(threads):
        p = mp.Process(
            target=env_create,
            args=(tim, result_queue),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
    print("Train agents created.")

    import time
    time1 = time.time()
    res = 0
    try:
        for _ in tqdm(range(threads*tim)):
            res += result_queue.get()
    finally:
        for p in processes:
            time.sleep(0.1)
            p.join()

    time2 = time.time()

    print(time2-time1)
    print(res)
        