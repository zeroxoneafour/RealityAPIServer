#!/usr/bin/env python

import asyncio
from websockets.asyncio.server import serve
from google.protobuf.empty_pb2 import Empty
from google.protobuf.message import DecodeError
from realityapi_pb2 import Command, DeviceInfo, Response, ServerRequest, RawLidarScan
from map.map import Map
from map.localization import ScanLocalizer
from map.room import get_room_info

# A device is dropped from the map once it has been quiet this long (seconds).
TIMEOUT = 2.0

# How often the monitor task sweeps the map for timed-out devices (seconds).
MONITOR_INTERVAL = 0.04

server_map = Map()
scan_localizer = ScanLocalizer()

# Handlers for each ServerRequest.data case.

def post_device_info(info: DeviceInfo) -> Response:
    # Each device's own coordinate system (and its offset) is applied in Map.
    print(f"<<< device info from {info.id}")
    return server_map.handle_device_info(info)

def post_heartbeat(device_id: str) -> Response:
    print(f"<<< heartbeat from {device_id}")
    return server_map.handle_heartbeat(device_id)

def fetch_all_device_info(_: Empty) -> Response:
    return server_map.get_all_device_info()

def fetch_single_device_info(device_id: str) -> Response:
    return server_map.get_device_info(device_id)

def fetch_lidar_data(_: Empty) -> Response:
    return server_map.get_lidar_data()

def fetch_room_data(_: Empty) -> Response:
    return Response(room_data=get_room_info())

def handle_raw_lidar_scan(raw_scan: RawLidarScan) -> Response:
    # The robot's first scan places it in the room (see Map.localize).
    return server_map.handle_lidar_scan(raw_scan)

def send_command(command: Command) -> Response:
    if not server_map.exists(command.target_id):
        return Response(err_id_not_found={})

    # TODO: forward to the target device as a ClientRequest once the server
    # tracks each device's connection.
    return Response(err_not_supported={})

# Keyed by the ServerRequest.data oneof field names.
HANDLERS = {
    "post_device_info": post_device_info,
    "post_heartbeat": post_heartbeat,
    "fetch_all_device_info": fetch_all_device_info,
    "fetch_single_device_info": fetch_single_device_info,
    "send_command": send_command,
    "fetch_lidar_data": fetch_lidar_data,
    "post_raw_lidar_scan": handle_raw_lidar_scan,
    "fetch_room_data": fetch_room_data,
}

def handle_request(req: ServerRequest) -> Response:
    kind = req.WhichOneof("data")
    if kind is None:
        ret = Response(err_not_supported={})
    else:
        ret = HANDLERS[kind](getattr(req, kind))

    ret.req_uuid = req.req_uuid
    return ret

async def handler(websocket):
    async for message in websocket:
        if isinstance(message, str):
            print("[!] ignoring text frame (expected binary protobuf)")
            continue

        req = ServerRequest()
        try:
            req.ParseFromString(message)                  # decode
        except DecodeError:
            print("[!] ignoring frame that is not a ServerRequest")
            continue

        ret = handle_request(req)
        await websocket.send(ret.SerializeToString())   # encode + send binary

async def monitor(map: Map):
    """Sweep the map so devices that stop sending heartbeats disappear."""
    while True:
        map.check_all(TIMEOUT)
        await asyncio.sleep(MONITOR_INTERVAL)


async def main():
    server_map.load_from_json("dashboard/updated_pos.json")
    server = await serve(handler, "0.0.0.0", 65432)
    asyncio.create_task(monitor(server_map))
    print("realityapi server on ws://0.0.0.0:65432")
    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
