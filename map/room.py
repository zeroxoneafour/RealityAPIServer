from realityapi_pb2 import RoomInfo, Vector2

# Room corners in meters, on the server's X/Z ground plane.
# (0, 0) is the corner furthest from the door.
# Vector2.x is server X and Vector2.y is server Z.
ROOM_CORNERS = [
    (0.0, 0.0),
    (0.0, 15.45),
    (5.16, 15.45),
    (5.16, 12.26),
    (9.07, 12.26),
    (9.07, 0.0),
]

def get_room_info() -> RoomInfo:
    return RoomInfo(corners=[Vector2(x=x, y=z) for x, z in ROOM_CORNERS])
