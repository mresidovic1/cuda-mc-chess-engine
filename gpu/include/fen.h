// fen.h - FEN (Forsyth-Edwards Notation) parsing and generation
// Allows setting up arbitrary chess positions for testing

#ifndef FEN_H
#define FEN_H

#include "chess_types.cuh"
#include <string>
#include <cstring>
#include <sstream>
#include <cctype>

// ============================================================================
// FEN Parser Results
// ============================================================================

enum class FENError {
    OK = 0,
    EMPTY_STRING,
    INVALID_PIECE_PLACEMENT,
    INVALID_RANK_COUNT,
    INVALID_FILE_COUNT,
    INVALID_SIDE_TO_MOVE,
    INVALID_CASTLING_RIGHTS,
    INVALID_EN_PASSANT,
    INVALID_HALFMOVE_CLOCK,
    INVALID_FULLMOVE_NUMBER,
    MISSING_KINGS,
    MULTIPLE_KINGS
};

inline const char* FENErrorToString(FENError err) {
    switch (err) {
        case FENError::OK: return "OK";
        case FENError::EMPTY_STRING: return "Empty FEN string";
        case FENError::INVALID_PIECE_PLACEMENT: return "Invalid piece placement";
        case FENError::INVALID_RANK_COUNT: return "Invalid rank count (expected 8)";
        case FENError::INVALID_FILE_COUNT: return "Invalid file count in rank";
        case FENError::INVALID_SIDE_TO_MOVE: return "Invalid side to move (expected 'w' or 'b')";
        case FENError::INVALID_CASTLING_RIGHTS: return "Invalid castling rights";
        case FENError::INVALID_EN_PASSANT: return "Invalid en passant square";
        case FENError::INVALID_HALFMOVE_CLOCK: return "Invalid halfmove clock";
        case FENError::INVALID_FULLMOVE_NUMBER: return "Invalid fullmove number";
        case FENError::MISSING_KINGS: return "Missing king(s)";
        case FENError::MULTIPLE_KINGS: return "Multiple kings for same color";
        default: return "Unknown error";
    }
}

// ============================================================================
// Square name utilities
// ============================================================================

inline int squareFromName(const char* name) {
    if (!name || !name[0] || !name[1]) return -1;
    int file = name[0] - 'a';
    int rank = name[1] - '1';
    if (file < 0 || file > 7 || rank < 0 || rank > 7) return -1;
    return rank * 8 + file;
}

inline std::string squareToName(int sq) {
    if (sq < 0 || sq > 63) return "-";
    char name[3];
    name[0] = 'a' + (sq % 8);
    name[1] = '1' + (sq / 8);
    name[2] = '\0';
    return std::string(name);
}

// ============================================================================
// FEN Parser
// ============================================================================

class FENParser {
public:
    // Standard starting position FEN
    static const char* STARTPOS;

    // Parse FEN string into BoardState
    // Returns FENError::OK on success, error code on failure
    static FENError parse(const std::string& fen, BoardState& board) {
        if (fen.empty()) return FENError::EMPTY_STRING;

        // Clear the board
        memset(&board, 0, sizeof(BoardState));
        board.ep_square = -1;

        std::istringstream ss(fen);
        std::string token;

        // 1. Piece placement
        if (!(ss >> token)) return FENError::INVALID_PIECE_PLACEMENT;
        FENError err = parsePiecePlacement(token, board);
        if (err != FENError::OK) return err;

        // Validate kings
        err = validateKings(board);
        if (err != FENError::OK) return err;

        // 2. Side to move
        if (!(ss >> token)) return FENError::INVALID_SIDE_TO_MOVE;
        if (token == "w") {
            board.side_to_move = WHITE;
        } else if (token == "b") {
            board.side_to_move = BLACK;
        } else {
            return FENError::INVALID_SIDE_TO_MOVE;
        }

        // 3. Castling rights
        if (!(ss >> token)) return FENError::INVALID_CASTLING_RIGHTS;
        err = parseCastlingRights(token, board);
        if (err != FENError::OK) return err;

        // 4. En passant target square
        if (!(ss >> token)) return FENError::INVALID_EN_PASSANT;
        err = parseEnPassant(token, board);
        if (err != FENError::OK) return err;

        // 5. Halfmove clock (optional)
        if (ss >> token) {
            try {
                board.halfmove = std::stoi(token);
            } catch (...) {
                return FENError::INVALID_HALFMOVE_CLOCK;
            }
        }

        // 6. Fullmove number (optional, we don't store it but validate)
        if (ss >> token) {
            try {
                std::stoi(token);  // Just validate it's a number
            } catch (...) {
                return FENError::INVALID_FULLMOVE_NUMBER;
            }
        }

        board.result = RESULT_ONGOING;
        return FENError::OK;
    }

    // Convert BoardState to FEN string
    static std::string toFEN(const BoardState& board) {
        std::ostringstream fen;

        // 1. Piece placement
        for (int rank = 7; rank >= 0; rank--) {
            int emptyCount = 0;
            for (int file = 0; file < 8; file++) {
                int sq = rank * 8 + file;
                char piece = getPieceChar(board, sq);

                if (piece == '.') {
                    emptyCount++;
                } else {
                    if (emptyCount > 0) {
                        fen << emptyCount;
                        emptyCount = 0;
                    }
                    fen << piece;
                }
            }
            if (emptyCount > 0) {
                fen << emptyCount;
            }
            if (rank > 0) fen << '/';
        }

        // 2. Side to move
        fen << ' ' << (board.side_to_move == WHITE ? 'w' : 'b');

        // 3. Castling rights
        fen << ' ';
        std::string castling;
        if (board.castling & CASTLE_WK) castling += 'K';
        if (board.castling & CASTLE_WQ) castling += 'Q';
        if (board.castling & CASTLE_BK) castling += 'k';
        if (board.castling & CASTLE_BQ) castling += 'q';
        fen << (castling.empty() ? "-" : castling);

        // 4. En passant
        fen << ' ';
        if (board.ep_square >= 0) {
            fen << squareToName(board.ep_square);
        } else {
            fen << '-';
        }

        // 5. Halfmove clock
        fen << ' ' << (int)board.halfmove;

        // 6. Fullmove number (we use 1 as default)
        fen << " 1";

        return fen.str();
    }

    // Validate that a FEN string is parseable and represents a legal position
    static bool validate(const std::string& fen, std::string& errorMsg) {
        BoardState board;
        FENError err = parse(fen, board);
        if (err != FENError::OK) {
            errorMsg = FENErrorToString(err);
            return false;
        }
        errorMsg = "OK";
        return true;
    }

private:
    static FENError parsePiecePlacement(const std::string& placement, BoardState& board) {
        int rank = 7;
        int file = 0;

        for (char c : placement) {
            if (c == '/') {
                if (file != 8) return FENError::INVALID_FILE_COUNT;
                rank--;
                file = 0;
                if (rank < 0) return FENError::INVALID_RANK_COUNT;
            } else if (std::isdigit(c)) {
                int skip = c - '0';
                file += skip;
                if (file > 8) return FENError::INVALID_FILE_COUNT;
            } else {
                if (file >= 8) return FENError::INVALID_FILE_COUNT;
                int sq = rank * 8 + file;
                Bitboard bb = C64(1) << sq;

                switch (c) {
                    case 'P': board.pieces[WHITE][PAWN]   |= bb; break;
                    case 'N': board.pieces[WHITE][KNIGHT] |= bb; break;
                    case 'B': board.pieces[WHITE][BISHOP] |= bb; break;
                    case 'R': board.pieces[WHITE][ROOK]   |= bb; break;
                    case 'Q': board.pieces[WHITE][QUEEN]  |= bb; break;
                    case 'K': board.pieces[WHITE][KING]   |= bb; break;
                    case 'p': board.pieces[BLACK][PAWN]   |= bb; break;
                    case 'n': board.pieces[BLACK][KNIGHT] |= bb; break;
                    case 'b': board.pieces[BLACK][BISHOP] |= bb; break;
                    case 'r': board.pieces[BLACK][ROOK]   |= bb; break;
                    case 'q': board.pieces[BLACK][QUEEN]  |= bb; break;
                    case 'k': board.pieces[BLACK][KING]   |= bb; break;
                    default: return FENError::INVALID_PIECE_PLACEMENT;
                }
                file++;
            }
        }

        if (rank != 0 || file != 8) return FENError::INVALID_RANK_COUNT;
        return FENError::OK;
    }

    static FENError parseCastlingRights(const std::string& castling, BoardState& board) {
        board.castling = 0;
        if (castling == "-") return FENError::OK;

        for (char c : castling) {
            switch (c) {
                case 'K': board.castling |= CASTLE_WK; break;
                case 'Q': board.castling |= CASTLE_WQ; break;
                case 'k': board.castling |= CASTLE_BK; break;
                case 'q': board.castling |= CASTLE_BQ; break;
                default: return FENError::INVALID_CASTLING_RIGHTS;
            }
        }
        return FENError::OK;
    }

    static FENError parseEnPassant(const std::string& ep, BoardState& board) {
        if (ep == "-") {
            board.ep_square = -1;
            return FENError::OK;
        }

        if (ep.length() != 2) return FENError::INVALID_EN_PASSANT;

        int sq = squareFromName(ep.c_str());
        if (sq < 0) return FENError::INVALID_EN_PASSANT;

        // EP square should be on rank 3 or 6
        int rank = sq / 8;
        if (rank != 2 && rank != 5) return FENError::INVALID_EN_PASSANT;

        board.ep_square = sq;
        return FENError::OK;
    }

    static FENError validateKings(const BoardState& board) {
        int whiteKings = popcount(board.pieces[WHITE][KING]);
        int blackKings = popcount(board.pieces[BLACK][KING]);

        if (whiteKings == 0 || blackKings == 0) return FENError::MISSING_KINGS;
        if (whiteKings > 1 || blackKings > 1) return FENError::MULTIPLE_KINGS;

        return FENError::OK;
    }

    static char getPieceChar(const BoardState& board, int sq) {
        Bitboard bb = C64(1) << sq;

        if (board.pieces[WHITE][PAWN]   & bb) return 'P';
        if (board.pieces[WHITE][KNIGHT] & bb) return 'N';
        if (board.pieces[WHITE][BISHOP] & bb) return 'B';
        if (board.pieces[WHITE][ROOK]   & bb) return 'R';
        if (board.pieces[WHITE][QUEEN]  & bb) return 'Q';
        if (board.pieces[WHITE][KING]   & bb) return 'K';
        if (board.pieces[BLACK][PAWN]   & bb) return 'p';
        if (board.pieces[BLACK][KNIGHT] & bb) return 'n';
        if (board.pieces[BLACK][BISHOP] & bb) return 'b';
        if (board.pieces[BLACK][ROOK]   & bb) return 'r';
        if (board.pieces[BLACK][QUEEN]  & bb) return 'q';
        if (board.pieces[BLACK][KING]   & bb) return 'k';

        return '.';
    }
};

// Standard starting position
const char* FENParser::STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// ============================================================================
// Convenience functions
// ============================================================================

inline FENError ParseFEN(const std::string& fen, BoardState& board) {
    return FENParser::parse(fen, board);
}

inline std::string BoardToFEN(const BoardState& board) {
    return FENParser::toFEN(board);
}

inline bool ValidateFEN(const std::string& fen) {
    std::string errorMsg;
    return FENParser::validate(fen, errorMsg);
}

#endif // FEN_H
