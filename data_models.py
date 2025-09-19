# --- Data Classes ---
import math

class Position:
    def __init__(self, data):
        self.x = data['x']
        self.y = data['y']

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

class ExtraStatus:
    def __init__(self, data):
        self.name = data['name']
        self.expire_at = data['expire_at']

class Player:
    def __init__(self, data):
        self.id = data['id']
        self.name = data['name']
        self.team = data['team']
        self.status = data['status']
        self.extra_status = [ExtraStatus(s) for s in data['extra_status']]
        self.position = Position(data['position'])
        self.direction = data['direction']
        self.bomb_pack_count = data['bomb_pack_count']
        self.sweet_potion_count = data['sweet_potion_count']
        self.agility_boots_count = data['agility_boots_count']
class Bomb:
    def __init__(self, data):
        self.position = Position(data['position'])
        self.owner_id = data['owner_id']
        self.explode_at = data['explode_at']
        self.range = data['range']
class MapItem:
    def __init__(self, data):
        self.type = data['type']
        self.position = Position(data['position'])

class MapCell:
    def __init__(self, data):
        self.terrain = data['terrain']
        self.ownership = data['ownership']
        self.owner_id = data.get('owner_id') # owner_id can be None

class Frame:
    def __init__(self, data):
        self.current_match_id = data['current_match_id']
        self.current_tick = data['current_tick']
        self.map = [[MapCell(cell) for cell in row] for row in data['map']]
        self.my_player = Player(data['my_player'])
        self.other_players = [Player(p) for p in data['other_players']]
        self.bombs = [Bomb(b) for b in data['bombs']]
        self.map_items = [MapItem(i) for i in data['map_items']]

class FrameList:
    def __init__(self):
        self.data = dict()
    def Add(self,number, frame):
        self.data[number] = frame
