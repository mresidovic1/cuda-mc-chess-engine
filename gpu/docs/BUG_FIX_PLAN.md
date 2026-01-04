# Perft Bug Fix Plan

## Problem Summary
Perft test at depth 4 produces 197,469 nodes instead of expected 197,281 (188 extra nodes).
Depths 0-3 pass perfectly, indicating the bug manifests in positions at depth 2-3.

## Perft Divide Analysis (Depth 4)

### Moves with CORRECT counts:
- b2b3 (9345), f2f3 (8457), g2g3 (9345) ✓
- b2b4 (9332), d2d4 (12435), f2f4 (8929), g2g4 (9328) ✓
- All knight moves (Na3, Nc3, Nf3, Nh3) ✓

### Moves with INCORRECT counts:

| Move  | Actual | Expected | Diff   | Pattern |
|-------|--------|----------|--------|---------|
| a2a3  | 8457   | 9893     | -1436  | Edge A  |
| h2h3  | 8457   | 9893     | -1436  | Edge H  |
| a2a4  | 9329   | 9467     | -138   | Edge A  |
| h2h4  | 9329   | 9467     | -138   | Edge H  |
| c2c3  | 9302   | 9272     | +30    | Center  |
| c2c4  | 9774   | 9744     | +30    | Center  |
| d2d3  | 11959  | 8073     | +3886  | Center  |
| e2e3  | 13198  | 9726     | +3472  | Center  |
| e2e4  | 13224  | 13134    | +90    | Center  |

## Key Observations

1. **Perfect Symmetry**: a-file and h-file show identical errors
   - a2a3 = h2h3 = 8457 (both wrong by same amount)
   - a2a4 = h2h4 = 9329 (both wrong by same amount)

2. **Center Files Have Extra Moves**: d & e files generate too many nodes
   - d2d3: +3886 extra!
   - e2e3: +3472 extra!

3. **Edge Files Missing Moves**: a & h files generate too few nodes
   - Suggests moves are being excluded incorrectly

4. **b, f, g files are PERFECT**: No bugs in these positions

## Hypothesis: Multiple Bugs

### Bug 1: Edge File Wrapping (A/H files)
**Symptoms**: a-file and h-file moves produce too few nodes

**Possible Causes**:
- Bitboard shift operations wrapping around edges incorrectly
- Pawn capture generation on a-file or h-file after position flip
- Attack detection on edge squares failing

**Code to Check**:
- `gpu_chess_bitops.cuh`: east(), west(), north_east(), north_west(), etc.
- `gpu_chess_movegen.cuh`: pawn capture generation
- `gpu_chess_position.cuh`: flip_bitboard() function

### Bug 2: Center Pawn Move Generation (D/E files)
**Symptoms**: d/e pawn moves produce WAY too many nodes (+3000-4000)

**Possible Causes**:
- Duplicate move generation for center pawns
- En passant bug creating illegal moves
- Double pawn push bug after flipping
- Pin detection failing for center files

**Code to Check**:
- `gpu_chess_movegen.cuh`: en passant generation (lines 359-395)
- `gpu_chess_movegen.cuh`: pawn double push logic
- `gpu_chess_position.cuh`: ep_square handling in apply_move() and flip_position()

### Bug 3: Minor Issues (C file)
**Symptoms**: c-file moves have +30 extra (small bug)

**Possible Causes**:
- Related to Bug 1 or Bug 2 but less severe
- Could be en passant related

## Investigation Priority

### Phase 1: Fix Edge File Bug (HIGH PRIORITY)
1. Check if flip_bitboard() correctly maps edge squares
2. Verify pawn_attacks() doesn't wrap incorrectly for a/h files
3. Test attack detection for a1/h1 and a8/h8 squares
4. Add unit test for position after a2a3 and h2h3

### Phase 2: Fix Center File Bug (CRITICAL - largest impact)
1. Review en passant logic thoroughly
2. Check if ep_square is flipped correctly (line 205-207 in gpu_chess_position.cuh)
3. Verify ep_square is set/cleared properly in apply_move()
4. Test position after d2d4 and e2e4 (which enable en passant)
5. Check if unpinned pawns can incorrectly capture en passant

### Phase 3: Verify Fix
1. Run perft divide again
2. All moves should match expected counts
3. Run full perft up to depth 5

## Specific Code Sections to Audit

### 1. En Passant Square Flipping
```cpp
// gpu_chess_position.cuh:205-207
if (pos->ep_square >= 0) {
    pos->ep_square = 63 - pos->ep_square;
}
```
**Question**: Is this correct? If ep_square is 16 (a3 in 0-63 notation), after flip it becomes 47 (h6). Is this right?

### 2. En Passant Capture Generation
```cpp
// gpu_chess_movegen.cuh:359-395
if (pos->ep_square >= 0) {
    Bitboard ep_bb = 1ULL << pos->ep_square;
    Bitboard ep_capturers = pawn_attacks(ep_bb, them) & pawns;
    // ... simulation-based validation ...
}
```
**Question**: Does this handle edge cases correctly? Are there positions where invalid ep captures are generated?

### 3. En Passant Square Setting
```cpp
// gpu_chess_position.cuh:135-137
if (flags == DOUBLE_PUSH) {
    pos->ep_square = us == WHITE ? from + 8 : from - 8;
}
```
**Question**: Is this the correct square? Should it be the square the pawn passed over, or something else?

### 4. Pawn Attacks on Edges
```cpp
// gpu_chess_bitops.cuh:80-86
__device__ __forceinline__ Bitboard pawn_attacks(Bitboard pawns, Color c) {
    if (c == WHITE) {
        return north_east(pawns) | north_west(pawns);
    } else {
        return south_east(pawns) | south_west(pawns);
    }
}
```
**Question**: Are north_east/north_west handling edge files correctly with the FILE_A/FILE_H masks?

## Testing Strategy

### Test 1: Single Move Position Analysis
Create positions after a2a3 and verify move generation manually:
- Count legal moves from resulting position
- Compare to expected

### Test 2: En Passant Specific Tests
Create position with en passant available and verify:
- Correct ep_square value after double push
- Correct ep_square after flip
- Correct moves generated with ep capture available
- No illegal ep captures generated

### Test 3: Edge File Specific Tests
Test positions with pieces on a-file and h-file:
- Verify attack generation doesn't wrap
- Verify moves generated correctly
- Verify flipping preserves edge file positions correctly

## Expected Outcome
After fixing these bugs, perft should pass at all depths:
- Depth 4: 197,281 nodes (currently 197,469)
- Depth 5: 4,865,609 nodes
- All perft divide counts should match reference values exactly
