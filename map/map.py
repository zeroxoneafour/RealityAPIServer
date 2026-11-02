from dataclasses import replace
from uuid import UUID
from time import time
import numpy as np
from map.coordinate import SERVER_COORDS, TURTLEBOT_COORDS, convert_lidar, convert_transform, convert_vec3
from map.device import Device, device_from_info, device_from_dict
from map.global_localization import LocalizationError, localize_in_room
from realityapi_pb2 import AllDeviceInfo, DeviceInfo, DeviceType, Lidar, RawLidarScan, Response, Vector3
import json, os
from map.localization import ScanLocalizer
from map.lidar import scan_to_world_points, simplify_lidar_2d, LIDAR_EPSILON

class Map:
    def __init__(self) -> None:
        self._devices: dict[str, Device] = {}
        self._localizers: dict[str, ScanLocalizer] = {}

    def add_device(self, device: Device):
        self._devices[device.id] = device
        print(f"Added device: {device} to dict")

    def exists(self, id: str) -> bool:
        return id in self._devices

    # function that defines what values are sent to updated_pos.json
    def export_to_json(self, filepath: str):
        devices_data = []

        for device in self._devices.values():
            print(f"{device}")
            devices_data.append(device.to_dict())

        # Write to a temp file, then swap it in, so the dashboard never
        # reads a half-written file.
        tmp_path = filepath + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(devices_data, f, indent=2)
        os.replace(tmp_path, filepath)

    def load_from_json(self, filepath: str):
        try:
            with open(filepath, 'r') as f:
                devices_data = json.load(f)

            devices_loaded = 0
            for device_data in devices_data:
                device = device_from_dict(device_data)
                if device is None:
                    print(f"Failed to load device {device_data.get('id')} (may be in wrong format)")
                    continue
                self.add_device(device)
                devices_loaded += 1
            print(f"Loaded {devices_loaded} devices from {filepath}")
        except FileNotFoundError:
            print(f"No existing JSON file at {filepath}, starting with empty map")
        except Exception as e:
            print(f"Error loading from JSON: {e}")

    def handle_lidar_scan(self, raw_scan: RawLidarScan) -> Response:
        device = self._devices.get(raw_scan.id)
        # Until a robot is placed in the room, its scans are used to place it.
        # This is also how a robot dropped for inactivity gets back on the map.
        if device is None or not device.localized:
            device = self.localize(raw_scan, device)
            if device is None:
                return Response(err_cannot_execute={})
        if device.coordinate_system is None:
            return Response(err_not_supported={})

        lidar = convert_lidar(scan_to_world_points(raw_scan), device.coordinate_system, SERVER_COORDS)
        transform = convert_transform(raw_scan.transform, device.coordinate_system, SERVER_COORDS)

        localizer = self._localizers.setdefault(device.id, ScanLocalizer())
        localizer.update(lidar)
        return self._update_device(device, DeviceInfo(
            id=device.id,
            transform=transform,
            lidar=simplify_lidar_2d(localizer.correct_lidar(lidar), epsilon=LIDAR_EPSILON),
        ))

    def localize(self, raw_scan: RawLidarScan, device: Device | None) -> Device | None:
        """Place a robot in the room from one scan, creating it if needed.

        The result is stored as the origin/yaw of the robot's coordinate system,
        so every later pose and scan from the robot is converted with it.
        Returns None if the scan cannot be placed.
        """
        # The robot's axis directions, without any offset.
        nominal = replace(
            device.coordinate_system if device and device.coordinate_system else TURTLEBOT_COORDS,
            origin=(0.0, 0.0, 0.0),
            yaw=0.0,
        )
        scan = convert_lidar(scan_to_world_points(raw_scan), nominal, SERVER_COORDS)
        robot = convert_vec3(raw_scan.transform.position, nominal, SERVER_COORDS)
        try:
            result = localize_in_room(
                np.array([[point.x, point.z] for point in scan.points]).reshape(-1, 2),
                np.array([robot.x, robot.z]),
            )
        except LocalizationError as e:
            print(f"Could not localize {raw_scan.id}: {e}")
            return None

        x, z = result.origin
        coordinate_system = replace(nominal, origin=(x, 0.0, z), yaw=result.yaw)
        if device is None:
            device = device_from_info(DeviceInfo(id=raw_scan.id, device_type=DeviceType.TURTLE, timestamp=time()))
            if device is None:
                return None
            self.add_device(device)
        device.coordinate_system = coordinate_system
        device.localized = True
        # Any drift map belongs to the old frame.
        self._localizers.pop(device.id, None)

        print(
            f"Localized {device.id}: odom origin at ({x:.2f}, {z:.2f}), "
            f"yaw {np.degrees(result.yaw):.1f} deg, {result.inlier_ratio:.0%} on walls, "
            f"ambiguity {result.ambiguity:.2f}"
        )
        return device

    def handle_device_info(self, device_info: DeviceInfo) -> Response:
        """Update a device from info reported in the device's own frame."""
        device = self._devices.get(device_info.id)
        if device is None: # if robot id does not exist, add it as a device
            device = device_from_info(device_info)
            if device is None:
                return Response(err_not_supported={})
            # Only a localization scan can place this kind of device in the room.
            if not device.localized:
                return Response(err_cannot_execute={})
            self.add_device(device)
        elif not device.localized:
            return Response(err_cannot_execute={})

        if device.coordinate_system is not None:
            if device_info.HasField("transform"):
                device_info.transform.CopyFrom(
                    convert_transform(device_info.transform, device.coordinate_system, SERVER_COORDS))
            if device_info.HasField("lidar"):
                device_info.lidar.CopyFrom(
                    convert_lidar(device_info.lidar, device.coordinate_system, SERVER_COORDS))

        return self._update_device(device, device_info)

    def _update_device(self, device: Device, device_info: DeviceInfo) -> Response:
        """Store info that is already in SERVER_COORDS."""
        # Every pose, from /odom or a scan, gets the correction exactly once here
        localizer = self._localizers.get(device.id)
        if localizer is not None and device_info.HasField("transform"):
            device_info.transform.CopyFrom(localizer.correct_transform(device_info.transform))
        device.handle_device_info(device_info)
        print(f"Set position for device: {device.id}")

        # A position report is also proof the device is still alive. Use the
        # server clock: the float32 wire timestamp is only accurate to ~128s.
        device.timestamp = time()

        # exports file to dashboard folder which updates the visual dashboard
        self.export_to_json("dashboard/updated_pos.json")

        return Response(ok={})

    def handle_heartbeat(self, device_id: str) -> Response:
        if not device_id in self._devices:
            return Response(err_id_not_found={})

        self._devices[device_id].heartbeat()

        return Response(ok={})

    def get_device_info(self, id: str) -> Response:
        if not id in self._devices:
            print(f"Device with id {id} does not exist")
            return Response(err_id_not_found={})

        dev = self._devices[id]
        return Response(device_info=dev.get_device_info())

    def get_all_device_info(self) -> Response:
        devices = [dev.get_device_info() for dev in self._devices.values()]
        return Response(all_device_info=AllDeviceInfo(devices=devices))

    def del_device(self, id: str) -> Device:
        return self._devices.pop(id)

    def update_last_time(self, id: str, new_time: float):
        self._devices[id].timestamp = new_time

    def heartbeat_check(self, id: str, timeout: float) -> bool:
        """Drop a device that has not been heard from within timeout seconds."""
        device = self._devices[id]

        if device.timestamp is not None and time() - device.timestamp >= timeout:
            print(f"Device {id} is inactive")
            self.del_device(id)
            return True

        return False

    def check_all(self, timeout: float):
        dropped = False
        for id in list(self._devices):
            dropped = self.heartbeat_check(id, timeout) or dropped

        # Re-export so the dashboard stops drawing a device that timed out.
        if dropped:
            self.export_to_json("dashboard/updated_pos.json")

    def get_lidar_data(self) -> Response:
        # Every localizer's accumulated map. Those points only track the X/Z
        # ground plane, so they are sent at height 0.
        points = []
        for localizer in self._localizers.values():
            if localizer.accumulated_scan is not None:
                points.extend(Vector3(x=float(x), y=0.0, z=float(z)) for x, z in localizer.accumulated_scan)
        return Response(lidar_data=Lidar(points=points))
