#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import time

import carla

import math


# from getKeys import key_check


def get_transform(vehicle_location, angle, d=-8.4):
    a = math.radians(angle)
    location = carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + vehicle_location
    return carla.Transform(location, carla.Rotation(yaw=angle, pitch=-15))


def isZero(val):
    return -0.01 < val < 0.01


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(4.0)
    world = client.get_world()
    spectator = world.get_spectator()

    actors = world.get_actors()
    cameras = []
    for actor in actors:
        if actor.type_id == "sensor.camera.rgb":
            cameras.append(actor)
            # print(dir(actor))
    num = 0
    while True:
        if len(cameras) > num:
            if isZero(cameras[num].get_transform().location.x) and isZero(cameras[num].get_transform().location.y) and isZero(cameras[num].get_transform().location.z):
                cameras.pop(num)
                # print("Removed dead camera")
                continue
            timestamp = world.wait_for_tick().timestamp
            spectator.set_transform(get_transform(cameras[num].get_location(), cameras[num].get_transform().rotation.yaw))
        else:
            cameras = []
            actors = world.get_actors()
            for actor in actors:
                if actor.type_id == "sensor.camera.rgb":
                    cameras.append(actor)
            if len(cameras) <= num:
                num = 0
                if len(cameras) > 0:
                    print(f"Camera overflowed to {num}, out of {len(cameras)} cameras...")
                else:
                    world = client.get_world()
                    spectator = world.get_spectator()

        # if "N" in key_check():
        #     num += 1
        #     time.sleep(1)
        #     print(f"Camera switched to {num}, out of {len(cameras)} cameras...")
        #     key_check()
        # if "Xsdf" in key_check():
        #     if cameras:
        #         cameras[num].destroy()
        #         cameras.pop(num)
        #         time.sleep(1)
        #         print(f"Deleted camera {num}, out of {len(cameras)} cameras...")
        #         key_check()


if __name__ == '__main__':
    while True:
        try:
            main()
        except:
            pass
