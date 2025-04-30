from engine import Engine
import chess

def uci_loop():
    engine = Engine()
    while True:
        try:
            command = input()
        except EOFError:
            break

        if command == "uci":
            print("id name Pawnstar")
            print("id author Ali Raza Khalid")
            print("uciok")

        elif command == "isready":
            print("readyok")

        elif command == "ucinewgame":
            engine = Engine()

        elif command.startswith("position"):
            parts = command.split(" ")
            if "startpos" in parts:
                engine.set_fen(chess.STARTING_FEN)
                move_index = parts.index("startpos") + 1
            elif "fen" in parts:
                fen_index = parts.index("fen") + 1
                fen = " ".join(parts[fen_index:fen_index + 6])
                engine.set_fen(fen)
                move_index = fen_index + 6
            else:
                continue

            if move_index < len(parts) and parts[move_index] == "moves":
                move_list = parts[move_index + 1:]
                moves = [chess.Move.from_uci(m) for m in move_list]
                engine.set_moves(moves)

        elif command.startswith("go"):
            # We'll use fixed depth for now
            move = engine.search(depth=4)
            print(f"bestmove {move}")

        elif command == "quit":
            break

        elif command == "d":
            engine.print_board()

if __name__ == "__main__":
    uci_loop()
