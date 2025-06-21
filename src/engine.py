import chess
from enum import Enum
import chess.polyglot
import time

class NodeType(Enum):
    EXACT = 0
    UPPER_BOUND = 1
    LOWER_BOUND = 2

class TTEntry:
    def __init__(self, hash_key, depth, score, node_type, best_move, age=0):
        self.hash_key = hash_key
        self.depth = depth
        self.score = score
        self.node_type = node_type
        self.best_move = best_move
        self.age = age

MATE_SCORE = 1000000
INFINITY = float('inf')

# Improved piece-square tables with better endgame values
pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
    'KE': ( -50, -40, -30, -20, -20, -30, -40, -50,
            -30, -20, -10,   0,   0, -10, -20, -30,
            -30, -10,  20,  30,  30,  20, -10, -30,
            -30, -10,  30,  40,  40,  30, -10, -30,
            -30, -10,  30,  40,  40,  30, -10, -30,
            -30, -10,  20,  30,  30,  20, -10, -30,
            -30, -30,   0,   0,   0,   0, -30, -30,
            -50, -30, -30, -30, -30, -30, -30, -50)
}

piece_to_pst = {
    chess.PAWN: 'P',
    chess.KNIGHT: 'N',
    chess.BISHOP: 'B',
    chess.ROOK: 'R',
    chess.QUEEN: 'Q',
    chess.KING: 'K',
}

# Killer moves for better move ordering
class KillerMoves:
    def __init__(self, max_ply=64):
        self.killers = [[None, None] for _ in range(max_ply)]
    
    def add_killer(self, ply, move):
        if self.killers[ply][0] != move:
            self.killers[ply][1] = self.killers[ply][0]
            self.killers[ply][0] = move
    
    def is_killer(self, ply, move):
        return move in self.killers[ply]

class Engine:
    def __init__(self):
        self.board = chess.Board()
        self.nodes_searched = 0
        self.transposition_table = {}
        self.tt_size = 1000000  # Increased TT size
        self.tt_hits = 0
        self.age = 0
        self.killer_moves = KillerMoves()
        self.history_table = {}  # History heuristic
        self.max_time = None
        self.start_time = None

    def set_fen(self, fen: str):
        self.board = chess.Board(fen)

    def set_moves(self, moves):
        for move in moves:
            self.board.push(move)

    def get_fen(self):
        return self.board.fen()
    
    def print_board(self):
        print(self.board)
    
    def piece_value(self, piece):
        """Assign values to pieces."""
        values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        return values[piece.piece_type] if piece.color == chess.WHITE else -values[piece.piece_type]
    
    def is_endgame(self):
        """Improved endgame detection."""
        white_queens = len(self.board.pieces(chess.QUEEN, chess.WHITE))
        black_queens = len(self.board.pieces(chess.QUEEN, chess.BLACK))
        
        white_material = (
            len(self.board.pieces(chess.ROOK, chess.WHITE)) * 500 +
            len(self.board.pieces(chess.KNIGHT, chess.WHITE)) * 320 +
            len(self.board.pieces(chess.BISHOP, chess.WHITE)) * 330
        )
        black_material = (
            len(self.board.pieces(chess.ROOK, chess.BLACK)) * 500 +
            len(self.board.pieces(chess.KNIGHT, chess.BLACK)) * 320 +
            len(self.board.pieces(chess.BISHOP, chess.BLACK)) * 330
        )
        
        # Endgame if no queens and low material, or total material < 2500
        return ((white_queens + black_queens == 0 and white_material + black_material < 1300) or
                white_material + black_material < 2500)

    def get_piece_position_score(self, piece, square, is_endgame):
        """Get positional score for piece."""
        piece_symbol = piece_to_pst[piece.piece_type]
        
        if is_endgame and piece.piece_type == chess.KING:
            piece_symbol += "E"

        if piece.color == chess.WHITE:
            return pst[piece_symbol][63-square]
        else:
            return -pst[piece_symbol][square]

    def get_rank_file(self, square):
        """Convert square to rank/file."""
        return square // 8, square % 8

    def king_endgame_score(self, white_king_sq, black_king_sq, our_color):
        """Improved king endgame evaluation."""
        def distance_to_edge(sq):
            rank, file = self.get_rank_file(sq)
            return min(rank, 7 - rank) + min(file, 7 - file)

        def manhattan_distance(sq1, sq2):
            r1, f1 = self.get_rank_file(sq1)
            r2, f2 = self.get_rank_file(sq2)
            return abs(r1 - r2) + abs(f1 - f2)

        if our_color == chess.WHITE:
            friendly_king = white_king_sq
            enemy_king = black_king_sq
        else:
            friendly_king = black_king_sq
            enemy_king = white_king_sq

        dist_between_kings = manhattan_distance(friendly_king, enemy_king)
        proximity_bonus = (14 - dist_between_kings) * 15
        edge_bonus = (7 - distance_to_edge(enemy_king)) * 10

        return proximity_bonus + edge_bonus

    def evaluate_mobility(self):
        """Evaluate piece mobility."""
        current_turn = self.board.turn
        
        # Count legal moves for current side
        mobility = len(list(self.board.legal_moves))
        
        # Switch turn and count opponent moves
        self.board.turn = not self.board.turn
        opponent_mobility = len(list(self.board.legal_moves))
        self.board.turn = current_turn
        
        mobility_diff = mobility - opponent_mobility
        return mobility_diff * 2 if current_turn == chess.WHITE else -mobility_diff * 2

    def evaluate_pawn_structure(self):
        """Basic pawn structure evaluation."""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            multiplier = 1 if color == chess.WHITE else -1
            pawns = self.board.pieces(chess.PAWN, color)
            
            # Doubled pawns penalty
            file_counts = [0] * 8
            for pawn in pawns:
                file_counts[pawn % 8] += 1
            
            for count in file_counts:
                if count > 1:
                    score += multiplier * -20 * (count - 1)
            
            # Isolated pawns penalty
            for pawn in pawns:
                file = pawn % 8
                has_neighbor = False
                
                for neighbor_file in [file - 1, file + 1]:
                    if 0 <= neighbor_file <= 7:
                        neighbor_pawns = [p for p in pawns if p % 8 == neighbor_file]
                        if neighbor_pawns:
                            has_neighbor = True
                            break
                
                if not has_neighbor:
                    score += multiplier * -15
        
        return score

    def evaluate(self, ply=0):
        """Enhanced evaluation function."""
        # Terminal positions
        if self.board.is_checkmate():
            return (-MATE_SCORE + ply) if self.board.turn == chess.WHITE else (MATE_SCORE - ply)
        
        if self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_fifty_moves():
            return 0

        score = 0
        white_king_sq = black_king_sq = None
        
        white_pieces = self.board.occupied_co[chess.WHITE]
        black_pieces = self.board.occupied_co[chess.BLACK]
        is_endgame = self.is_endgame()
        
        # Material and positional evaluation
        for square in chess.scan_forward(white_pieces):
            piece = self.board.piece_at(square)
            score += self.piece_value(piece)
            score += self.get_piece_position_score(piece, square, is_endgame)
            if piece.piece_type == chess.KING:
                white_king_sq = square
        
        for square in chess.scan_forward(black_pieces):
            piece = self.board.piece_at(square)
            score += self.piece_value(piece)
            score += self.get_piece_position_score(piece, square, is_endgame)
            if piece.piece_type == chess.KING:
                black_king_sq = square
        
        # Endgame king activity
        if is_endgame and white_king_sq is not None and black_king_sq is not None:
            endgame_score = (self.king_endgame_score(white_king_sq, black_king_sq, chess.WHITE) - 
                           self.king_endgame_score(white_king_sq, black_king_sq, chess.BLACK))
            score += endgame_score
        
        # Additional evaluation factors
        if not is_endgame:
            score += self.evaluate_mobility()
            score += self.evaluate_pawn_structure()
        
        return score

    def score_move(self, move, ply, tt_move=None):
        """Enhanced move scoring."""
        if tt_move and move.uci() == tt_move:
            return 100000
        
        score = 0
        
        # Captures with MVV-LVA
        if self.board.is_capture(move):
            victim = self.board.piece_at(move.to_square)
            attacker = self.board.piece_at(move.from_square)
            if victim and attacker:
                score = 50000 + abs(self.piece_value(victim)) * 10 - abs(self.piece_value(attacker))
        
        # Promotions
        elif move.promotion:
            score = 40000 + self.piece_value(chess.Piece(move.promotion, self.board.turn))
        
        # Checks
        elif self.board.gives_check(move):
            score = 30000
        
        # Killer moves
        elif self.killer_moves.is_killer(ply, move.uci()):
            score = 20000
        
        # History heuristic
        else:
            move_key = (move.from_square, move.to_square)
            score = self.history_table.get(move_key, 0)
        
        return score

    def ordered_moves(self, ply):
        """Get ordered moves."""
        pos_hash = chess.polyglot.zobrist_hash(self.board)
        tt_entry = self.transposition_table.get(pos_hash)
        tt_move = tt_entry.best_move if tt_entry else None
        
        moves = list(self.board.legal_moves)
        scored_moves = [(move, self.score_move(move, ply, tt_move)) for move in moves]
        scored_moves.sort(key=lambda x: -x[1])
        
        return [move for move, _ in scored_moves]

    def store_position(self, hash_key, depth, score, node_type, best_move):
        """Store position with age-based replacement."""
        if len(self.transposition_table) >= self.tt_size:
            # Remove oldest entries
            keys_to_remove = []
            for key, entry in self.transposition_table.items():
                if entry.age < self.age - 2:  # Remove entries older than 2 searches
                    keys_to_remove.append(key)
                    if len(keys_to_remove) >= self.tt_size // 10:  # Remove 10% at a time
                        break
            
            for key in keys_to_remove:
                self.transposition_table.pop(key)
        
        self.transposition_table[hash_key] = TTEntry(
            hash_key, depth, score, node_type, best_move, self.age
        )

    def quiescence(self, alpha, beta, ply=0):
        """Quiescence search to avoid horizon effect."""
        self.nodes_searched += 1
        
        # Check time limit
        if self.max_time and time.time() - self.start_time > self.max_time:
            return 0
        
        stand_pat = self.evaluate(ply)
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        
        # Only search captures and checks
        for move in self.board.legal_moves:
            if not (self.board.is_capture(move) or self.board.gives_check(move)):
                continue
            
            self.board.push(move)
            score = -self.quiescence(-beta, -alpha, ply + 1)
            self.board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        return alpha

    def minmax(self, depth, alpha, beta, maximizing_player, ply=0):
        """Enhanced minimax with various improvements."""
        self.nodes_searched += 1
        
        # Check time limit
        if self.max_time and time.time() - self.start_time > self.max_time:
            return self.evaluate(ply), []
        
        alpha_orig = alpha
        
        if depth == 0:
            return self.quiescence(alpha, beta, ply), []
        
        if self.board.is_game_over():
            return self.evaluate(ply), []
        
        # Transposition table lookup
        pos_hash = chess.polyglot.zobrist_hash(self.board)
        tt_entry = self.transposition_table.get(pos_hash)
        
        if tt_entry and tt_entry.depth >= depth:
            self.tt_hits += 1
            if tt_entry.node_type == NodeType.EXACT:
                return tt_entry.score, [tt_entry.best_move] if tt_entry.best_move else []
            elif tt_entry.node_type == NodeType.LOWER_BOUND:
                alpha = max(alpha, tt_entry.score)
            elif tt_entry.node_type == NodeType.UPPER_BOUND:
                beta = min(beta, tt_entry.score)
            
            if alpha >= beta:
                return tt_entry.score, [tt_entry.best_move] if tt_entry.best_move else []
        
        # Null move pruning (in non-endgame positions)
        if (depth >= 3 and not self.is_endgame() and 
            not self.board.is_check() and maximizing_player):
            
            self.board.push(chess.Move.null())
            null_score, _ = self.minmax(depth - 3, -beta, -beta + 1, False, ply + 1)
            self.board.pop()
            
            if -null_score >= beta:
                return beta, []
        
        moves = self.ordered_moves(ply)
        if not moves:
            return self.evaluate(ply), []
        
        best_move = None
        best_line = []
        
        if maximizing_player:
            max_eval = -INFINITY
            
            for i, move in enumerate(moves):
                self.board.push(move)
                
                # Late move reduction
                reduction = 0
                if (i >= 4 and depth >= 3 and 
                    not self.board.is_capture(move) and 
                    not self.board.gives_check(move)):
                    reduction = 1
                
                eval_score, line = self.minmax(depth - 1 - reduction, alpha, beta, False, ply + 1)
                
                # Re-search if LMR failed
                if reduction > 0 and eval_score > alpha:
                    eval_score, line = self.minmax(depth - 1, alpha, beta, False, ply + 1)
                
                self.board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_line = [move.uci()] + line
                    best_move = move.uci()
                
                alpha = max(max_eval, alpha)
                
                if beta <= alpha:
                    # Update killer moves and history
                    if not self.board.is_capture(move):
                        self.killer_moves.add_killer(ply, move.uci())
                        move_key = (move.from_square, move.to_square)
                        self.history_table[move_key] = self.history_table.get(move_key, 0) + depth * depth
                    break
            
            # Store in transposition table
            node_type = NodeType.EXACT
            if max_eval <= alpha_orig:
                node_type = NodeType.UPPER_BOUND
            elif max_eval >= beta:
                node_type = NodeType.LOWER_BOUND
            
            self.store_position(pos_hash, depth, max_eval, node_type, best_move)
            return max_eval, best_line
            
        else:
            min_eval = INFINITY
            
            for i, move in enumerate(moves):
                self.board.push(move)
                
                # Late move reduction
                reduction = 0
                if (i >= 4 and depth >= 3 and 
                    not self.board.is_capture(move) and 
                    not self.board.gives_check(move)):
                    reduction = 1
                
                eval_score, line = self.minmax(depth - 1 - reduction, alpha, beta, True, ply + 1)
                
                # Re-search if LMR failed
                if reduction > 0 and eval_score < beta:
                    eval_score, line = self.minmax(depth - 1, alpha, beta, True, ply + 1)
                
                self.board.pop()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_line = [move.uci()] + line
                    best_move = move.uci()
                
                beta = min(min_eval, beta)
                
                if alpha >= beta:
                    # Update killer moves and history
                    if not self.board.is_capture(move):
                        self.killer_moves.add_killer(ply, move.uci())
                        move_key = (move.from_square, move.to_square)
                        self.history_table[move_key] = self.history_table.get(move_key, 0) + depth * depth
                    break
            
            # Store in transposition table
            node_type = NodeType.EXACT
            if min_eval <= alpha_orig:
                node_type = NodeType.UPPER_BOUND
            elif min_eval >= beta:
                node_type = NodeType.LOWER_BOUND
            
            self.store_position(pos_hash, depth, min_eval, node_type, best_move)
            return min_eval, best_line

    def iterative_deepening(self, max_depth, max_time=None):
        """Iterative deepening with time management."""
        self.max_time = max_time
        self.start_time = time.time()
        self.age += 1
        
        best_move = None
        
        for depth in range(1, max_depth + 1):
            if self.max_time and time.time() - self.start_time > self.max_time * 0.8:
                break
                
            self.nodes_searched = 0
            self.tt_hits = 0
            
            move = self.search_depth(depth)
            if move:
                best_move = move
            
            elapsed = time.time() - self.start_time
            if self.max_time and elapsed > self.max_time * 0.5:
                break
        
        return best_move

    def search_depth(self, depth):
        """Search at specific depth."""
        best_score = -INFINITY if self.board.turn == chess.WHITE else INFINITY
        best_move = None
        best_line = []
        
        moves = self.ordered_moves(0)
        if not moves:
            return None
        
        for move in moves:
            self.board.push(move)
            score, line = self.minmax(depth - 1, -INFINITY, INFINITY, self.board.turn, ply=1)
            self.board.pop()
            
            if self.board.turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
                    best_line = [move.uci()] + line
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                    best_line = [move.uci()] + line
        
        # Output UCI info
        elapsed = time.time() - self.start_time if self.start_time else 0
        nps = int(self.nodes_searched / max(elapsed, 0.001))
        
        if abs(best_score) > 900000:
            mate_in = (MATE_SCORE - abs(best_score)) // 2
            mate_in = -mate_in if best_score < 0 else mate_in
            print(f"info depth {depth} score mate {mate_in} nodes {self.nodes_searched} nps {nps} time {int(elapsed * 1000)} pv {' '.join(best_line)}")
        else:
            print(f"info depth {depth} score cp {best_score} nodes {self.nodes_searched} nps {nps} time {int(elapsed * 1000)} hashfull {len(self.transposition_table)} pv {' '.join(best_line)}")
        
        return best_move.uci() if best_move else None

    def search(self, depth=6, max_time=None):
        """Main search function."""
        return self.iterative_deepening(depth, max_time)