# FOR TESTING W/ KEYBOARD: ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true
import time
from uuid import uuid4
import math

import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from websockets.sync.client import connect

from init_turtlebot import create_turtlebot
from realityapi_pb2 import Response, ServerRequest, Vector3, RawLidarScan

SERVER_URI = "ws://0.0.0.0:65432"

# The server drops a device after 2 seconds of silence.
HEARTBEAT_INTERVAL = 1.0

# The robot counts as still below these odometry speeds (m/s and rad/s).
STILL_LINEAR = 0.01
STILL_ANGULAR = 0.02

# Seconds between localization attempts. Each one costs the server a room search.
LOCALIZE_RETRY = 2.0


class CoordinateNode(Node):
    def __init__(self):
        super().__init__('coordinate_node')
        self.robot = create_turtlebot()
        self.ws = connect(SERVER_URI)

        # Nothing but localization scans is sent until the server has placed
        # this robot in the room.
        self.localized = False
        self.have_odom = False
        self.moving = False
        self.last_localize_attempt = -math.inf
        self.get_logger().info('Hold still: localizing against the room')

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.position_update_callback,
            qos_profile_sensor_data
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_update_callback,
            qos_profile_sensor_data
        )

        self.create_timer(HEARTBEAT_INTERVAL, self.heartbeat_callback)

    def position_update_callback(self, data: Odometry):
        twist = data.twist.twist
        self.moving = (
            math.hypot(twist.linear.x, twist.linear.y) > STILL_LINEAR
            or abs(twist.angular.z) > STILL_ANGULAR
        )
        self.have_odom = True

        position = data.pose.pose.position
        pos_vec = Vector3(x=position.x, y=position.y, z=position.z)

        # Yaw about ROS +z from the orientation quaternion.
        q = data.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        rot_vec = Vector3(x=0.0, y=0.0, z=yaw)

        if pos_vec == self.robot.transform.position and rot_vec == self.robot.transform.rotation:
            return

        self.robot.transform.position.CopyFrom(pos_vec)
        self.robot.transform.rotation.CopyFrom(rot_vec)
        if self.localized:
            self.post_device_info(log=True)

    def lidar_update_callback(self, data: LaserScan):
        if self.localized:
            self.send_scan(data)
        else:
            self.localize(data)

    def localize(self, data: LaserScan):
        """Send one scan, taken while still, for the server to place in the room."""
        if not self.have_odom:
            self.get_logger().info('Waiting for odometry before localizing', throttle_duration_sec=2.0)
            return
        if self.moving:
            self.get_logger().warning('Hold still: cannot localize while moving', throttle_duration_sec=2.0)
            return
        now = time.monotonic()
        if now - self.last_localize_attempt < LOCALIZE_RETRY:
            return
        self.last_localize_attempt = now

        reply = self.send_scan(data, log=True)
        if reply.HasField('ok'):
            self.localized = True
            self.get_logger().info('Localized against the room; streaming pose and scans')
        else:
            self.get_logger().warning(
                f'Could not localize ({reply.WhichOneof("data")}); retrying. '
                'If this keeps failing, move the robot somewhere it can see more walls.')

    def send_scan(self, ros_data, log=False) -> Response:
        raw_scan = RawLidarScan(
            id = self.robot.id,
            transform = self.robot.transform,
            ranges = ros_data.ranges,
            range_min = ros_data.range_min,
            range_max = ros_data.range_max,
            angle_min = ros_data.angle_min,
            angle_increment = ros_data.angle_increment
        )
        return self.send_request(ServerRequest(post_raw_lidar_scan=raw_scan), log=log)

    def heartbeat_callback(self):
        if self.localized:
            self.send_request(ServerRequest(post_heartbeat=self.robot.id))

    def post_device_info(self, log=False):
        self.robot.timestamp = time.time()
        # Poses stay in the odometry frame; the server applies this robot's
        # room offset. Odometry updates reuse the last scan already stored on
        # the server.
        device_info = self.robot.get_device_info(include_lidar=False)
        self.send_request(
            ServerRequest(post_device_info=device_info),
            log=log,
        )

    def send_request(self, req: ServerRequest, log=False, include_lidar=False) -> Response:
        req.req_uuid = str(uuid4())
        started = time.perf_counter()
        self.ws.send(req.SerializeToString())
        reply = Response()
        reply.ParseFromString(self.ws.recv())
        latency_ms = (time.perf_counter() - started) * 1000
        if log:
            self.get_logger().info(f'>>> {self.describe(req, include_lidar=include_lidar)}')
            self.get_logger().info(f'<<< {reply.WhichOneof("data")} | {latency_ms:.1f} ms')
        if reply.req_uuid != req.req_uuid:
            self.get_logger().warning(
                f'response {reply.req_uuid} does not match request {req.req_uuid}')
        # The server forgets a robot that goes quiet, so start over.
        if self.localized and reply.WhichOneof('data') in ('err_id_not_found', 'err_cannot_execute'):
            self.localized = False
            self.get_logger().warning('Server lost this robot. Hold still: relocalizing')
        return reply

    def describe(self, req: ServerRequest, include_lidar=False) -> str:
        if req.HasField("post_heartbeat"):
            return f'heartbeat id {req.post_heartbeat}'
        if req.HasField("post_raw_lidar_scan"):
            scan = req.post_raw_lidar_scan
            return f'scan {len(scan.ranges)} ranges id {scan.id}'
        info = req.post_device_info
        if include_lidar:
            p = info.transform.position
            return (
                f'lidar {len(info.lidar.points)} points '
                f'at ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) id {info.id}'
            )
        p = info.transform.position
        return f'({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) id {info.id}'

    def destroy_node(self):
        self.ws.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CoordinateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
