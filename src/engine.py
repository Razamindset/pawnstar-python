import chess

MATE_SCORE = 1000000

class Engine():
    def __init__(self):
        self.board = chess.Board()
        self.nodes_searched = 0
        pass

    def set_fen(self, fen:str):
        self.board = chess.Board(fen)

    def set_moves(self, moves):
        for move in moves:
            self.board.push(move)

    def get_fen(self):
        return self.board.fen()
    
    def generate_moves(self):
        return self.board.legal_moves
    
    def undo_move(self):
        """Undo the last move."""
        if self.board.move_stack:
            self.board.pop()

    def is_game_over(self):
        """Check if the game is over."""
        return self.board.is_game_over()
    
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
        for piece in self.board.piece_map().values():
            score += self.piece_value(piece)
        
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

    
    def minmax(self, depth, alpha, beta, maximizing_player, ply=0):
        self.nodes_searched += 1
        if(depth == 0 or self.board.is_game_over()):
            return  self.evaluate(ply)

        # Alpha is the best eval so far for the maximizing player
        if maximizing_player:
            max_eval = -float('inf')
            for move in self.ordered_moves():
                self.board.push(move)
                eval = self.minmax(depth -1, alpha, beta, False, ply + 1)
                self.board.pop()

                # From the best eval so far and the curr eval we will choose the best one
                max_eval = max(max_eval, eval)
                alpha = max(max_eval, alpha) # This will update the best eval so far if possible

                # If beta goes negative it means the other player is winning so we will not make such move and prune the branch
                if beta <= alpha:
                    break

            return max_eval
            
        # Opposite to that of maximizing player
        else:
            min_eval = float('inf')
            for move in self.ordered_moves():
                self.board.push(move)
                eval = self.minmax(depth -1, alpha, beta, True, ply + 1)
                self.board.pop()

                # Update the beta and eval values
                min_eval = min(eval, min_eval)
                beta = min(min_eval, beta) # Minimizing player wants the lowest value

                # If value gets in favor of maximizing player prune the branch
                if alpha >= beta:
                    break

            return min_eval



    def search(self, depth):
        """Do simple search for now"""

        # If white to move best score is the worst that can happen so any other move will be evaluated greator than that
        best_score = -float('inf') if self.board.turn == chess.WHITE else float('inf')
        best_move = None

        # Loop and evaluate each move
        for move in self.ordered_moves():
            self.board.push(move)
            score = self.minmax(depth -1, -float('inf'), float('inf'), self.board.turn, ply=1)
            self.board.pop()

            if(self.board.turn == chess.WHITE):
                if score > best_score:
                    best_score = score
                    best_move = move
            
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        print("Nodes Searched: ", self.nodes_searched)

        mate_In = MATE_SCORE + best_score if self.board.turn == chess.BLACK else MATE_SCORE - best_score 
        if(mate_In <=  2*depth):
            print("Mate IN: ", mate_In)

        print("Best Move: ", best_move)
        print("Centipawns: ", best_score)


        return best_move.uci() if best_move else None