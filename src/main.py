from engine import Engine
import chess
import time

def parse_time_control(parts):
    """Parse UCI go command for time control."""
    wtime = btime = winc = binc = movestogo = movetime = None
    depth = 4  # default depth
    
    i = 1  # skip 'go'
    while i < len(parts):
        if parts[i] == "wtime" and i + 1 < len(parts):
            wtime = int(parts[i + 1])
            i += 2
        elif parts[i] == "btime" and i + 1 < len(parts):
            btime = int(parts[i + 1])
            i += 2
        elif parts[i] == "winc" and i + 1 < len(parts):
            winc = int(parts[i + 1])
            i += 2
        elif parts[i] == "binc" and i + 1 < len(parts):
            binc = int(parts[i + 1])
            i += 2
        elif parts[i] == "movestogo" and i + 1 < len(parts):
            movestogo = int(parts[i + 1])
            i += 2
        elif parts[i] == "movetime" and i + 1 < len(parts):
            movetime = int(parts[i + 1])
            i += 2
        elif parts[i] == "depth" and i + 1 < len(parts):
            depth = int(parts[i + 1])
            i += 2
        elif parts[i] == "infinite":
            depth = 20
            i += 1
        else:
            i += 1
    
    return wtime, btime, winc, binc, movestogo, movetime, depth

def calculate_time_for_move(engine, wtime, btime, winc, binc, movestogo):
    """Calculate how much time to spend on this move."""
    if engine.board.turn == chess.WHITE:
        my_time = wtime
        my_inc = winc
    else:
        my_time = btime
        my_inc = binc
    
    if my_time is None:
        return None
    
    # Convert from milliseconds to seconds
    my_time /= 1000.0
    if my_inc:
        my_inc /= 1000.0
    else:
        my_inc = 0
    
    # Simple time management
    if movestogo:
        # Time per move when moves to go is specified
        time_per_move = (my_time + my_inc * movestogo) / (movestogo + 1)
    else:
        # Assume 40 moves left in game on average
        estimated_moves_left = 40
        time_per_move = (my_time + my_inc * estimated_moves_left) / (estimated_moves_left + 10)
    
    # Don't use more than 1/3 of remaining time
    max_time = my_time / 3.0
    
    # Add increment
    time_per_move += my_inc * 0.8
    
    # Ensure we don't exceed maximum
    return min(time_per_move, max_time, my_time - 0.1)

def uci_loop():
    engine = Engine()
    
    while True:
        try:
            command = input().strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not command:
            continue

        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "uci":
            print("id name Pawnstar Enhanced")
            print("id author Ali Raza Khalid")
            print("option name Hash type spin default 64 min 1 max 1024")
            print("option name Threads type spin default 1 min 1 max 1")
            print("uciok")

        elif cmd == "isready":
            print("readyok")

        elif cmd == "setoption":
            # Handle options like hash size
            if len(parts) >= 5 and parts[1] == "name" and parts[3] == "value":
                option_name = parts[2].lower()
                option_value = parts[4]
                
                if option_name == "hash":
                    try:
                        hash_mb = int(option_value)
                        # Rough conversion: 64MB ≈ 1M entries
                        engine.tt_size = max(1000, hash_mb * 16384)
                        print(f"# Hash size set to {hash_mb}MB ({engine.tt_size} entries)")
                    except ValueError:
                        pass

        elif cmd == "ucinewgame":
            engine = Engine()
            print("# New game started")

        elif cmd == "position":
            if len(parts) < 2:
                continue
                
            try:
                if parts[1] == "startpos":
                    engine.set_fen(chess.STARTING_FEN)
                    move_index = 2
                    
                    if len(parts) > 2 and parts[2] == "moves":
                        move_index = 3
                        
                elif parts[1] == "fen":
                    if len(parts) < 8:
                        continue
                    fen = " ".join(parts[2:8])
                    engine.set_fen(fen)
                    move_index = 8
                    
                    if len(parts) > 8 and parts[8] == "moves":
                        move_index = 9
                else:
                    continue
                
                # Apply moves if present
                if move_index < len(parts):
                    move_list = parts[move_index:]
                    moves = []
                    for move_str in move_list:
                        try:
                            move = chess.Move.from_uci(move_str)
                            if move in engine.board.legal_moves:
                                moves.append(move)
                            else:
                                print(f"# Illegal move: {move_str}")
                                break
                        except ValueError:
                            print(f"# Invalid move format: {move_str}")
                            break
                    
                    engine.set_moves(moves)
                    
            except Exception as e:
                print(f"# Error setting position: {e}")

        elif cmd == "go":
            if engine.board.is_game_over():
                legal_moves = list(engine.board.legal_moves)
                if legal_moves:
                    print(f"bestmove {legal_moves[0].uci()}")
                else:
                    print("bestmove 0000")
                continue
            
            wtime, btime, winc, binc, movestogo, movetime, depth = parse_time_control(parts)
            
            # Determine time to use
            max_time = None
            if movetime:
                max_time = movetime / 1000.0  # Convert to seconds
            elif wtime is not None or btime is not None:
                max_time = calculate_time_for_move(engine, wtime, btime, winc, binc, movestogo)
                if max_time and max_time < 0.01:  # Minimum time
                    max_time = 0.01
            
            try:
                start_time = time.time()
                best_move = engine.search(depth=depth, max_time=max_time)
                elapsed = time.time() - start_time
                
                if best_move:
                    print(f"bestmove {best_move}")
                else:
                    # Fallback to any legal move
                    legal_moves = list(engine.board.legal_moves)
                    if legal_moves:
                        print(f"bestmove {legal_moves[0].uci()}")
                    else:
                        print("bestmove 0000")
                        
                print(f"# Search completed in {elapsed:.2f}s")
                
            except Exception as e:
                print(f"# Search error: {e}")
                legal_moves = list(engine.board.legal_moves)
                if legal_moves:
                    print(f"bestmove {legal_moves[0].uci()}")
                else:
                    print("bestmove 0000")

        elif cmd == "stop":
            # In a real implementation, this would interrupt the search
            print("# Stop command received")

        elif cmd == "quit" or cmd == "exit":
            print("# Goodbye!")
            break

        elif cmd == "d" or cmd == "display":
            engine.print_board()
            print(f"FEN: {engine.get_fen()}")
            print(f"Turn: {'White' if engine.board.turn else 'Black'}")
            print(f"Legal moves: {len(list(engine.board.legal_moves))}")

        elif cmd == "eval":
            score = engine.evaluate()
            print(f"Static evaluation: {score} cp")

        elif cmd == "perft":
            if len(parts) >= 2:
                try:
                    depth = int(parts[1])
                    start_time = time.time()
                    nodes = perft(engine.board, depth)
                    elapsed = time.time() - start_time
                    print(f"Perft {depth}: {nodes} nodes in {elapsed:.3f}s ({int(nodes/max(elapsed, 0.001))} nps)")
                except ValueError:
                    print("# Invalid perft depth")

        elif cmd.startswith("#"):
            # Comment, ignore
            pass

        else:
            print(f"# Unknown command: {command}")

def perft(board, depth):
    """Performance test - count nodes at given depth."""
    if depth == 0:
        return 1
    
    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += perft(board, depth - 1)
        board.pop()
    
    return nodes

if __name__ == "__main__":
    print("# Pawnstar Enhanced Chess Engine")
    print("# Type 'uci' to start UCI mode")
    uci_loop()