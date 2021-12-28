import logging

import gym
from gym import Env
from gym.spaces import Box, Discrete
import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import pyglet
import carla

IMG_WIDTH = 256
IMG_HEIGHT = 256

SHOW_PREVIEW = False

IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE_UNDER50 = 30

stateWidth = 128
stateHeight = 128


def make_fixed_time_step(world, time_step=.05):
    settings = world.get_settings()
    settings.fixed_delta_seconds = time_step
    world.apply_settings(settings)


def set_no_rendering_mode(world):
    settings = world.get_settings()
    settings.no_rendering_mode = True
    world.apply_settings(settings)


def make_synchronous(world, state):
    settings = world.get_settings()
    settings.synchronous_mode = state
    world.apply_settings(settings)


# noinspection PyAttributeOutsideInit
class CarEnv(Env):
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self, autopilot):
        self.autopilotEnabled = autopilot
        self.collision_hist = []
        self.actor_list = []
        self.client = carla.Client("192.168.1.2", 2000)
        self.client.set_timeout(60.0)
        self.world = self.client.get_world()
        self.client.reload_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.action_space = Box(-1, 1, (3,))
        # self.action_space = Discrete(4)
        self.observation_space = Box(0, 1, (stateWidth, stateHeight, 3), np.uint8)
        self.vehicle = None
        self.traffic = False
        self.reset()

    def summonTraffic(self):
        self.actor_list2 = []
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        count = 50
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        if count < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif count > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, count, number_of_spawn_points)
            count = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= count:
                break
            try:
                blueprint = random.choice(blueprints)
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                blueprint.set_attribute('role_name', 'autopilot')
                batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))
            except Exception as e:
                print(e)
                print("exception caught")

        for response in self.client.apply_batch_sync(batch):
            if response.error:
                logging.error(response.error)
            else:
                self.actor_list2.append(response.actor_id)
        self.world.tick()
        print('spawned %d vehicles, press Ctrl+C to exit.' % len(self.actor_list2))
        self.traffic = True

    def reset(self):
        weather = carla.WeatherParameters(
            cloudiness=80.0,
            precipitation=80.0,
            sun_altitude_angle=70.0)

        self.world.set_weather(weather)

        try:
            if self.vehicle:
                self.vehicle.set_autopilot(False)
        except:
            print("Autopilot error caught")
        for agent in self.actor_list:
            try:
                agent.destroy()
            except:
                print(agent)
        #  pass
        self.collision_hist = []
        self.actor_list = []
        # self.world = self.client.reload_world()
        spawned = False
        while not spawned:
            try:
                self.transform = random.choice(self.world.get_map().get_spawn_points())
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
                spawned = True
            except:
                self.world.tick()

        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        if not self.traffic:
            self.summonTraffic()
        # self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(1)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        # self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(1)
        self.getState()
        self.kmhStart = None
        make_fixed_time_step(self.world, 0.1)
        make_synchronous(self.world, True)
        if self.autopilotEnabled:
            print("activating autopilot")
            self.vehicle.set_autopilot(True)
            print("activated autopilot")
        self.world.tick()
        self.world.tick()
        return self.state

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        # print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def getState(self):
        self.state = self.front_camera
        # self.state = cv2.cvtColor(self.front_camera, cv2.COLOR_BGR2GRAY)
        self.state = cv2.resize(self.state, dsize=(stateWidth, stateHeight))
        # cv2.imshow("state", self.state)

    def step(self, action):
        self.world.tick()
        if self.vehicle.is_at_traffic_light():
            traffic_light = self.vehicle.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                # world.hud.notification("Traffic light changed! Good to go!")
                traffic_light.set_state(carla.TrafficLightState.Green)

        # snapshot = self.world.wait_for_tick()
        if not self.autopilotEnabled:
            # act = [float(action[0]), float(action[1] * 2 - 1), 0 if action[2] < 0.7 else 1]
            act = [float(action[0]), float(action[1] * 2 - 1), 0 if action[2] < 0.7 else 1]
            print(act)
            self.vehicle.apply_control(carla.VehicleControl(throttle=act[0], steer=act[1], brake=act[2]))  # , brake=0 if action[2] < 0.7 else 1))
        # if action == 0:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1 * self.STEER_AMT))
        # elif action == 1:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        # elif action == 2:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1 * self.STEER_AMT))
        # elif action == 3:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0))
        # else:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0, brake=0.5))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        reward = 0
        if len(self.collision_hist) != 0:
            self.reset()
            done = True
            reward += -10
        elif kmh < 20:
            done = False
            if kmh > 10:
                reward += max(0.05, 0.5 - 0.5 * 10 / kmh)
            else:
                reward += -0.05
            if kmh < 5:
                if not self.kmhStart:
                    self.kmhStart = time.time()
                if time.time() - self.kmhStart > SECONDS_PER_EPISODE_UNDER50:
                    done = True
                    reward += -10
                # reward = -50
            else:
                self.kmhStart = None
        else:
            done = False
            reward += 0.8
        if self.autopilotEnabled:
            done = False
        # if self.episode_start + SECONDS_PER_EPISODE < time.time():
        #     done = Trueeeeee
        self.getState()
        # self.render()
        control = self.vehicle.get_control()

        return self.state, reward, done, [float(control.throttle), float((control.steer + 1) / 2), float(control.brake)]

    def render(self, mode=""):
        cv2.imshow("Render", cv2.resize(self.state, dsize=(256, 256)))
        cv2.waitKey(1)

    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    def resetTimer(self):
        self.kmhStart = time.time()

    def pause(self):
        # self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        self.resetTimer()
        # self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

    def close(self):
        # if self.vehicle:
        # self.vehicle.set_autopilot(False)
        #  for agent in self.actor_list:
        #  try:
        #     agent.destroy()
        #  except:
        #  pass
        self.vehicle = None
        self.actor_list = []
        self.traffic = False
        self.client.reload_world()

    def autopilot(self, state):
        self.vehicle.set_autopilot(state)
        self.autopilotEnabled = state


def main():
    env = CarEnv(True)
    try:
        import random

        while True:
            done = False
            while not done:
                front_camera, reward, done, actionsTaken = env.step([1, 0, 0])
                print(actionsTaken)
                env.render()
            env.reset()
    except Exception as e:
        env.close()


if __name__ == '__main__':
    main()
