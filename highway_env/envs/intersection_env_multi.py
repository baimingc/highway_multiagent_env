from __future__ import division, print_function, absolute_import
from gym.envs.registration import register
import numpy as np
from gym import spaces
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import observation_factory
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.control import MDPVehicle

import time



class IntersectionMAEnv(AbstractEnv):
    COLLISION_REWARD = -20
    HIGH_VELOCITY_REWARD = 1
    ARRIVED_REWARD = 10
    NUM_VEHICLE = 4

    ACTIONS = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER'
    }
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}
    
    def define_spaces(self):
        super().define_spaces()
        self.action_space = [spaces.Discrete(len(self.ACTIONS)) for _ in range(self.NUM_VEHICLE)]
        if "observation" not in self.config:
            raise ValueError("The observation configuration must be defined")
        self.observation = observation_factory(self, self.config["observation"])
        self.observation_space = [self.observation.space() for _ in range(self.NUM_VEHICLE)]
        
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "SelfKinematics",
                "vehicles_count": 4,
                "features": ["x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": True,
                "flatten": False,
                "observe_intentions": False
            },
            "duration": 20,  # [s]
            "destination": None,
            "screen_width": 600,
            "screen_height": 600,
            # "centering_position": [0.5, 0.6],
            "centering_position": [0.5, 0.5],
            # "scaling": 5.5 * 1.3,
            "scaling": 6,
            "collision_reward": IntersectionMAEnv.COLLISION_REWARD,
            "normalize_reward": False
        })
        return config

    def _reward(self, action):
        # reward = self.config["collision_reward"] * self.vehicle.crashed \
        #          + self.HIGH_VELOCITY_REWARD * (self.vehicle.velocity_index == self.vehicle.SPEED_COUNT - 1)
        # reward = self.ARRIVED_REWARD if self.has_arrived else reward
        # if self.config["normalize_reward"]:
        #     reward = utils.remap(reward, [self.config["collision_reward"], self.ARRIVED_REWARD], [0, 1])
        # return reward
        reward = 0
        for vehicle in self.vehicle:
            reward += self.config["collision_reward"] * vehicle.crashed \
                    + self.HIGH_VELOCITY_REWARD * (vehicle.velocity_index == vehicle.SPEED_COUNT - 1)
            reward += self.ARRIVED_REWARD if self.has_arrived else 0
        if self.config["normalize_reward"]:
            reward = utils.remap(reward, [self.config["collision_reward"], self.ARRIVED_REWARD], [0, 1])
        return reward

    def _is_terminal(self):
        """
            The episode is over when a collision occurs or when the access ramp has been passed.
        """

        return self.crashed \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"] \
               or self.has_arrived
        # return self.vehicle.crashed \
        #        or self.steps >= self.config["duration"] * self.config["policy_frequency"] \
        #        or self.has_arrived

    def reset(self):
        self._make_road()
        self._make_vehicles()
        self.steps = 0
        return super(IntersectionMAEnv, self).reset()

    def step(self, action):
        results = super(IntersectionMAEnv, self).step(action)
        self.steps += 1
        self._clear_vehicles()
        
        obs = results[0]
        agent_obs = []
        for i in range(4):
            agent_obs.append(np.array([obs[i%4], obs[(i+1)%4], obs[(i+2)%4], obs[(i+3)%4]]).flatten())
        agent_obs = [agent_obs]
        
        rewards = [[results[1], results[1], results[1], results[1]]]
        dones = [[results[2], results[2], results[2], results[2]]]
        # self._spawn_vehicle()
        return np.array(agent_obs), np.array(rewards), np.array(dones), results[3]

    def _make_road(self):
        """
            Make an 4-way intersection.

            The horizontal road has the right of way. More precisely, the levels of priority are:
                - 3 for horizontal straight lanes and right-turns
                - 1 for vertical straight lanes and right-turns
                - 2 for horizontal left-turns
                - 0 for vertical left-turns
            The code for nodes in the road network is:
            (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)
        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50   # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10))
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius, angle + np.radians(180), angle + np.radians(270),
                                      line_types=[n, c], priority=priority, speed_limit=10))
            # Left turn
            l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))
            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4),
                         StraightLane(start, end, line_types=[s, n], priority=priority, speed_limit=10))
            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % 4), "o" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))

        road = RegulatedRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self, n_vehicles=10):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as multiple ego-vehicle.
        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 2  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
        # for t in range(n_vehicles - 1):
        #     self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        # for _ in range(simulation_steps):
        #     [(self.road.act(), self.road.step(1 / self.SIMULATION_FREQUENCY)) for _ in range(self.SIMULATION_FREQUENCY)]

        # Challenger vehicle
        # self._spawn_vehicle(60, spawn_probability=1, go_straight=True, position_deviation=0.1, velocity_deviation=0)

        # Ego-vehicle
        MDPVehicle.SPEED_MIN = 0
        MDPVehicle.SPEED_MAX = 9
        MDPVehicle.SPEED_COUNT = 3
        # MDPVehicle.TAU_A = 1.0
        # ego_lane = self.road.network.get_lane(("o0", "ir0", 0))
        ego_lane = self.road.network.get_lane(("o0", "ir0", 0))
        # destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
        destination = "o1"
#         ego_lane.position(max(0, np.random.normal(10, 10)), 0),
        ego_vehicle1 = MDPVehicle(self.road,
                                 ego_lane.position(max(0, np.random.normal(10, 5)), 0),
                                 velocity=ego_lane.speed_limit,
                                 heading=ego_lane.heading_at(50)) \
            .plan_route_to(destination)
        self.road.vehicles.append(ego_vehicle1)

        ego_lane = self.road.network.get_lane(("o1", "ir1", 0))
        # destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
        destination = "o2"
        ego_vehicle2 = MDPVehicle(self.road,
                                 ego_lane.position(max(0, np.random.normal(10, 5)), 0),
                                 velocity=ego_lane.speed_limit,
                                 heading=ego_lane.heading_at(50)) \
            .plan_route_to(destination)
        self.road.vehicles.append(ego_vehicle2)

        ego_lane = self.road.network.get_lane(("o2", "ir2", 0))
        # destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
        destination = "o3"
        ego_vehicle3 = MDPVehicle(self.road,
                                 ego_lane.position(max(0, np.random.normal(10, 5)), 0),
                                 velocity=ego_lane.speed_limit,
                                 heading=ego_lane.heading_at(50)) \
            .plan_route_to(destination)
        self.road.vehicles.append(ego_vehicle3)

        ego_lane = self.road.network.get_lane(("o3", "ir3", 0))
        # destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
        destination = "o0"
        ego_vehicle4 = MDPVehicle(self.road,
                                 ego_lane.position(max(0, np.random.normal(10, 5)), 0),
                                 velocity=ego_lane.speed_limit,
                                 heading=ego_lane.heading_at(50)) \
            .plan_route_to(destination)
        self.road.vehicles.append(ego_vehicle4)
        
        self.vehicle = [ego_vehicle1, ego_vehicle2, ego_vehicle3, ego_vehicle4]
        # for v in self.road.vehicles:  # Prevent early collisions
        #     if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
        #         self.road.vehicles.remove(v)

    def _spawn_vehicle(self,
                       longitudinal=0,
                       position_deviation=1.,
                       velocity_deviation=1.,
                       spawn_probability=0.6,
                       go_straight=False):
        if self.np_random.rand() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(self.road, ("o" + str(route[0]), "ir" + str(route[0]), 0),
                                            longitudinal=longitudinal + 5 + self.np_random.randn() * position_deviation,
                                            velocity=8 + self.np_random.randn() * velocity_deviation)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self):
        is_leaving = lambda vehicle: "il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                                     and vehicle.lane.local_coordinates(vehicle.position)[0] \
                                     >= vehicle.lane.length - 4 * vehicle.LENGTH
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle is self.vehicle or not (is_leaving(vehicle) or vehicle.route is None)]

#     @property
#     def has_arrived(self):
#         arrived = False
#         for vehicle in self.vehicle:
#             arrived = arrived \
#             or "il" in vehicle.lane_index[0] \
#             and "o" in vehicle.lane_index[1] \
#             and vehicle.lane.local_coordinates(vehicle.position)[0] >= \
#             vehicle.lane.length - 3 * vehicle.LENGTH
#         # return "il" in self.vehicle.lane_index[0] \
#         #        and "o" in self.vehicle.lane_index[1] \
#         #        and self.vehicle.lane.local_coordinates(self.vehicle.position)[0] >= \
#         #        self.vehicle.lane.length - 3 * self.vehicle.LENGTH
#         return arrived
    @property
    def has_arrived(self):
        arrived = True
        for vehicle in self.vehicle:
            arrived = arrived \
            and np.linalg.norm(vehicle.destination - vehicle.position) < 40
        # return "il" in self.vehicle.lane_index[0] \
        #        and "o" in self.vehicle.lane_index[1] \
        #        and self.vehicle.lane.local_coordinates(self.vehicle.position)[0] >= \
        #        self.vehicle.lane.length - 3 * self.vehicle.LENGTH
        return arrived
    @property
    def crashed(self):
        crashed = False
        for vehicle in self.vehicle:
            crashed = crashed or vehicle.crashed
        # return "il" in self.vehicle.lane_index[0] \
        #        and "o" in self.vehicle.lane_index[1] \
        #        and self.vehicle.lane.local_coordinates(self.vehicle.position)[0] >= \
        #        self.vehicle.lane.length - 3 * self.vehicle.LENGTH
        return crashed

    def _cost(self, action):
        """
            The constraint signal is the occurence of collisions.
        """
        return float(self.vehicle.crashed)


register(
    id='intersection-multiagent-v0',
    entry_point='highway_env.envs:IntersectionMAEnv',
)
