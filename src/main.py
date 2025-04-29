from engine import Engine

engine = Engine()
engine.set_fen("k7/5R2/2K5/8/8/8/8/8 w - - 0 1")
move = engine.search(4)
# print("Move:", move)
