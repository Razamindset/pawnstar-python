import chess
from enum import Enum

import chess.polyglot

class NodeType(Enum):
    EXACT = 0
    UPPER_BOUND = 1
    LOWER_BOUND = 2

class TTEntry():
    def __init__(self, hash_key, depth, score, node_type, best_move):
        self.hash_key = hash_key    # Position hash
        self.depth = depth          # Search depth
        self.score = score          # Position score
        self.node_type = node_type  # Type of node (EXACT, UPPERBOUND, LOWERBOUND)
        self.best_move = best_move  # Best move found

MATE_SCORE = 1000000


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
}

# Create mirrored tables for black pieces
pst_mirrored = {}
for piece, table in pst.items():
    pst_mirrored[piece.lower()] = tuple(reversed(table))

# print(pst_mirrored['p'][45])
# print( pst['P'][45])

piece_to_pst = {
    chess.PAWN: 'P',
    chess.KNIGHT: 'N',
    chess.BISHOP: 'B',
    chess.ROOK: 'R',
    chess.QUEEN: 'Q',
    chess.KING: 'K',
}


class Engine():
    def __init__(self):
        self.board = chess.Board()
        self.nodes_searched = 0
        self.transposition_table = {}
        self.tt_size = 1000000
        self.tt_hits = 0
        pass

    def set_fen(self, fen:str):
        self.board = chess.Board(fen)

    def set_moves(self, moves):
        for move in moves:
            self.board.push(move)

    def get_fen(self):
        return self.board.fen()
    
    def print_board(self):
        print(self.board)
    
    def piece_value(self, piece):
        """Assign simple values to pieces."""
        values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0  # King has infinite value, but not useful in evaluation
        }
        return values[piece.piece_type] if piece.color == chess.WHITE else -values[piece.piece_type]
    
    def get_piece_position_score(self, piece, square):
        """Get the position score for a piece on a square."""
        piece_type = piece_to_pst[piece.piece_type]
        
        # Adjust square index for black pieces (mirror vertically)
        if piece.color == chess.WHITE:
            return pst_mirrored[piece_type.lower()][square]
        else:
            # For black pieces, we need to flip the square vertically
            return -pst[piece_type][square]

    def evaluate(self, ply=0):
        """Evaluate based on material difference, mate, and draw"""
        # Check for checkmate
        if self.board.is_checkmate():
            if self.board.turn == chess.WHITE:
                return -MATE_SCORE + ply # Black wins and played the last move
            else:
                return MATE_SCORE - ply  # White wins and played the last move
            
        
        # Check for stalemate
        if self.board.is_stalemate():
            return 0 

        # Check for draw by insufficient material or fifty-move rule
        if self.board.is_insufficient_material() or self.board.is_fifty_moves():
            return 0  
    
        score = 0
        
        # Evaluate material and piece positioning
        for square, piece in self.board.piece_map().items():
            # Add material value
            score += self.piece_value(piece)
            
            # Add position value
            pos_score = self.get_piece_position_score(piece, square)

            score += pos_score
        
        return score
    
    def score_move(self, move):
        """Scores for moves, The higher the better"""
        # Captues, promotion and then checks ->
        if self.board.is_capture(move):
            captured = self.board.piece_at(move.to_square)
            attacker = self.board.piece_at(move.from_square)
            if captured and attacker:
                # _ is numerical literal has no affect on value
                return 10_000 + self.piece_value(captured) - self.piece_value(attacker)
        
        if move.promotion:
            return 9_000 + (self.piece_value(chess.Piece(move.promotion, self.board.turn)))

        if self.board.gives_check(move):
            return 5_000
        
        
        # Else it is quiet move 
        return 0

    
    def ordered_moves(self):
        """Orders Moves based on priority"""
        moves = list(self.board.legal_moves)
        moves.sort(key=lambda move: -self.score_move(move))
        return moves

    def store_position(self, hash_key, depth, score, node_type, best_move):
        """Store position in transposition table with simple size management"""
        if len(self.transposition_table) >= self.tt_size:
            # Simple replacement: remove a random entry
            self.transposition_table.pop(next(iter(self.transposition_table)))
            
        self.transposition_table[hash_key] = TTEntry(
            hash_key, depth, score, node_type, best_move
        )

    def minmax(self, depth, alpha, beta, maximizing_player, ply=0):
        self.nodes_searched += 1

        alpha_orig = alpha
        
        if(depth == 0 or self.board.is_game_over()):
            return  self.evaluate(ply), []

        # Search the table 
        pos_hash = chess.polyglot.zobrist_hash(self.board)

        tt_entry = self.transposition_table.get(pos_hash)

        # I have partially understood the upperbound and lower bound logic
        if tt_entry and tt_entry.depth >= depth:
            self.tt_hits +=1
            if tt_entry.node_type == NodeType.EXACT:
                return tt_entry.score, [tt_entry.best_move]
            elif tt_entry.node_type == NodeType.LOWER_BOUND:
                alpha = max(alpha, tt_entry.score)
            elif tt_entry.node_type == NodeType.UPPER_BOUND:
                beta = min(beta, tt_entry.score)
            
            if alpha >= beta:
                return tt_entry.score, [tt_entry.best_move]



        # Alpha is the best eval so far for the maximizing player
        if maximizing_player:
            max_eval = -float('inf')
            best_line = []
            best_move = None

            for move in self.ordered_moves():
                self.board.push(move)
                eval, line = self.minmax(depth -1, alpha, beta, False, ply + 1)
                self.board.pop()

                # From the best eval so far and the curr eval we will choose the best one
                if eval > max_eval:
                    max_eval = eval
                    best_line = [move.uci()] + line
                    best_move = move

                alpha = max(max_eval, alpha) # This will update the best eval so far if possible

                #* If at any point the score returned by the minimizing player is greater than or equal to beta, the maximizing player knows that the minimizing player will avoid this branch, so it can stop exploring further down this path (beta cutoff). Means there is a better move somewhere else for black
                if beta <= alpha:
                    break
            # Determine node type and store in transposition table
            node_type = NodeType.EXACT
            if max_eval <= alpha_orig:
                node_type = NodeType.UPPER_BOUND
            elif max_eval >= beta:
                node_type = NodeType.LOWER_BOUND

            # Store position in transposition table
            self.store_position(pos_hash, depth, max_eval, node_type, best_move)

            return max_eval, best_line
            
        # Opposite to that of maximizing player
        else:
            min_eval = float('inf')
            best_line = []
            best_move = None
            for move in self.ordered_moves():
                self.board.push(move)
                eval, line = self.minmax(depth -1, alpha, beta, True, ply + 1)
                self.board.pop()

                # Update the beta and eval values
                if eval < min_eval:
                    min_eval = eval
                    best_line = [move.uci()] + line
                    best_move = move

                beta = min(min_eval, beta) # Minimizing player wants the lowest value

                # If at any point the score returned by the maximizing player is less than or equal to alpha, the minimizing player knows that the maximizing player will avoid this branch, so it can stop exploring further down this path (alpha cutoff).
                if alpha >= beta:
                    break
            
            # Determine node type and store in transposition table
            node_type = NodeType.EXACT
            if min_eval <= alpha_orig:
                node_type = NodeType.UPPER_BOUND
            elif min_eval >= beta:
                node_type = NodeType.LOWER_BOUND

            # Store position in transposition table
            self.store_position(pos_hash, depth, min_eval, node_type, best_move)
            
            return min_eval, best_line



    def search(self, depth):
        """Do simple search for now"""

        # If white to move best score is the worst that can happen so any other move will be evaluated greater than that
        best_score = -float('inf') if self.board.turn == chess.WHITE else float('inf')
        best_move = None
        best_line = []

        # Loop and evaluate each move
        for move in self.ordered_moves():
            self.board.push(move)
            score, line = self.minmax(depth -1, -float('inf'), float('inf'), self.board.turn, ply=1)
            self.board.pop()

            if(self.board.turn == chess.WHITE):
                if score > best_score:
                    best_score = score
                    best_move = move
                    best_line = [move.uci()] + line
            
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                    best_line = [move.uci()] + line


        # print("Nodes Searched: ", self.nodes_searched)

        if abs(best_score) > 900_000:
            mate_in = (MATE_SCORE - best_score) // 2 if self.board.turn == chess.WHITE else (MATE_SCORE + best_score) // 2
            print(f"info depth {depth} score mate {mate_in} pv {' '.join(best_line)}")
        else:
            print(f"info depth {depth} score cp {best_score} pv {' '.join(best_line)}")

        print("tt hits: ", self.tt_hits)

        return best_move.uci() if best_move else None