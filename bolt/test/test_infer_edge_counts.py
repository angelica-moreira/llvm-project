#!/usr/bin/env python3
"""
Validate the InferEdgeCounts propagation against Wu-Larus Section 4.

Reproduces the algorithm on hand-crafted CFGs with known expected results.
Uses the same logic as InferEdgeCounts.cpp: reverse postorder traversal,
cyclic probability scaling for loop headers, proportional edge distribution.
"""

def propagate(blocks, edges, samples, entry):
    """
    blocks: list of block names
    edges: dict mapping block -> list of successor blocks
    samples: dict mapping block -> sample count
    entry: name of entry block
    
    Returns: dict mapping (src, dst) -> edge count
    """
    # Build predecessor map.
    preds = {b: [] for b in blocks}
    for src, succs in edges.items():
        for dst in succs:
            preds[dst].append(src)

    # Detect back edges: (src, dst) where dst dominates src.
    # Simple heuristic: dst appears before src in block order (preorder).
    block_order = {b: i for i, b in enumerate(blocks)}
    back_edges = set()
    for src, succs in edges.items():
        for dst in succs:
            if block_order[dst] <= block_order[src] and dst != src:
                # Check if dst is a loop header (has a pred that comes after it).
                back_edges.add((src, dst))

    # Identify loop headers.
    loop_headers = {dst for (_, dst) in back_edges}

    # Block frequencies (will be updated during propagation).
    freq = dict(samples)

    # Edge counts.
    edge_count = {}

    # Process in block order (approximates reverse postorder for reducible CFGs).
    visited = set()

    for bb in blocks:
        if bb != entry:
            # Sum incoming non-back-edge frequencies.
            in_sum = 0
            for pred in preds[bb]:
                if pred in visited and (pred, bb) not in back_edges:
                    in_sum += edge_count.get((pred, bb), 0)

            # Loop header scaling.
            if bb in loop_headers and samples[bb] > 0:
                non_back_samples = sum(
                    samples[p] for p in preds[bb]
                    if (p, bb) not in back_edges
                )
                cp = 0.0
                if non_back_samples < samples[bb]:
                    cp = 1.0 - non_back_samples / samples[bb]
                cp = min(cp, 0.99)
                if cp > 0.0 and in_sum > 0:
                    in_sum = int(in_sum / (1.0 - cp))

            if in_sum > 0:
                freq[bb] = in_sum

        visited.add(bb)

        count = freq[bb]
        succs = edges.get(bb, [])
        if not succs or count == 0:
            continue

        if len(succs) == 1:
            edge_count[(bb, succs[0])] = count
            continue

        total_succ = sum(samples.get(s, 0) for s in succs)
        if total_succ == 0:
            each = count // len(succs)
            for s in succs:
                edge_count[(bb, s)] = each
        else:
            remaining = count
            for s in succs:
                ec_val = int(count * samples.get(s, 0) / total_succ)
                if ec_val == 0 and samples.get(s, 0) > 0:
                    ec_val = 1
                if ec_val > remaining:
                    ec_val = remaining
                edge_count[(bb, s)] = ec_val
                remaining -= ec_val

    return edge_count, freq


def test_linear():
    """A -> B -> C, all sampled equally."""
    blocks = ["A", "B", "C"]
    edges = {"A": ["B"], "B": ["C"]}
    samples = {"A": 100, "B": 100, "C": 100}
    ec, freq = propagate(blocks, edges, samples, "A")

    assert ec[("A", "B")] == 100, f"A->B: {ec[('A','B')]}"
    assert ec[("B", "C")] == 100, f"B->C: {ec[('B','C')]}"
    print("PASS: linear")


def test_diamond():
    """
    A -> B, A -> C, B -> D, C -> D.
    A=100, B=70, C=30, D=100.
    """
    blocks = ["A", "B", "C", "D"]
    edges = {"A": ["B", "C"], "B": ["D"], "C": ["D"]}
    samples = {"A": 100, "B": 70, "C": 30, "D": 100}
    ec, freq = propagate(blocks, edges, samples, "A")

    assert ec[("A", "B")] == 70, f"A->B: {ec[('A','B')]}"
    assert ec[("A", "C")] == 30, f"A->C: {ec[('A','C')]}"
    # B gets 70 from A, passes it to D.
    assert ec[("B", "D")] == 70, f"B->D: {ec[('B','D')]}"
    # C gets 30 from A, passes it to D.
    assert ec[("C", "D")] == 30, f"C->D: {ec[('C','D')]}"

    # Flow conservation: D incoming = 70 + 30 = 100 = D.count.
    assert freq["D"] == 100, f"D freq: {freq['D']}"
    print("PASS: diamond")


def test_simple_loop():
    """
    Wu-Larus style loop:
      Entry(A) -> Header(H) -> Body(B) -> H (back edge)
                  H -> Exit(E)
    
    Samples: A=10, H=100, B=90, E=10.
    The loop iterates ~10 times (100/10 entry).
    CyclicProb = 1 - 10/100 = 0.9.
    
    Expected:
      A->H = 10  (entry)
      H freq = 10 / (1-0.9) = 100
      H->B = 100 * 90/(90+10) = 90
      H->E = 100 * 10/(90+10) = 10
      B->H = 90  (back edge, from B's count)
    """
    blocks = ["A", "H", "B", "E"]
    edges = {"A": ["H"], "H": ["B", "E"], "B": ["H"]}
    samples = {"A": 10, "H": 100, "B": 90, "E": 10}
    ec, freq = propagate(blocks, edges, samples, "A")

    assert ec[("A", "H")] == 10, f"A->H: {ec[('A','H')]}"
    assert freq["H"] == 100, f"H freq: {freq['H']}"
    assert ec[("H", "B")] == 90, f"H->B: {ec[('H','B')]}"
    assert ec[("H", "E")] == 10, f"H->E: {ec[('H','E')]}"
    assert ec[("B", "H")] == 90, f"B->H: {ec[('B','H')]}"

    # Flow conservation at H: in = A->H + B->H = 10 + 90 = 100 = H.count.
    in_h = ec[("A", "H")] + ec[("B", "H")]
    assert in_h == 100, f"H incoming: {in_h}"
    # Flow conservation at H: out = H->B + H->E = 90 + 10 = 100.
    out_h = ec[("H", "B")] + ec[("H", "E")]
    assert out_h == 100, f"H outgoing: {out_h}"
    print("PASS: simple_loop")


def test_nested_loop():
    """
    A -> H1 -> H2 -> B -> H2 (inner back)
                H2 -> C -> H1 (outer back)
                H1 -> E
    
    Samples: A=1, H1=10, H2=100, B=90, C=9, E=1.
    Inner loop: ~10 iterations (100/10 entries from H1).
    Outer loop: ~10 iterations (10/1 entries from A).
    """
    blocks = ["A", "H1", "H2", "B", "C", "E"]
    edges = {
        "A": ["H1"],
        "H1": ["H2", "E"],
        "H2": ["B", "C"],
        "B": ["H2"],  # inner back edge
        "C": ["H1"],  # outer back edge
    }
    samples = {"A": 1, "H1": 10, "H2": 100, "B": 90, "C": 9, "E": 1}
    ec, freq = propagate(blocks, edges, samples, "A")

    # H1: entry from A=1, cyclicProb = 1 - 1/10 = 0.9, freq = 1/(1-0.9) = 10.
    assert freq["H1"] == 10, f"H1 freq: {freq['H1']}"

    # H1->H2 and H1->E distributed by samples: H2=100, E=1.
    # H1->H2 = 10 * 100/101 ~ 9, H1->E = 10 * 1/101 ~ 0.
    # (integer truncation)
    # H1: freq should be ~10 (entry=1, 10 iterations).
    assert freq["H1"] == 10, f"H1 freq: {freq['H1']}"

    # H1->E must be > 0 (exit path must have flow).
    assert ec.get(("H1", "E"), 0) >= 1, f"H1->E: {ec.get(('H1','E'), 0)}"

    # H1->H2 should be most of H1's count.
    assert ec.get(("H1", "H2"), 0) >= 8, f"H1->H2: {ec.get(('H1','H2'), 0)}"

    # Flow conservation at H1: out = H1->H2 + H1->E should equal H1.freq.
    out_h1 = ec.get(("H1", "H2"), 0) + ec.get(("H1", "E"), 0)
    assert abs(out_h1 - freq["H1"]) <= 1, f"H1 out: {out_h1} vs freq: {freq['H1']}"

    # H2 should be scaled by inner loop.
    assert freq["H2"] >= 80, f"H2 freq: {freq['H2']}"

    print(f"  H1->H2: {ec.get(('H1','H2'), 0)}, H1->E: {ec.get(('H1','E'), 0)}")
    print(f"  H2 freq: {freq['H2']}")
    print(f"  H2->B: {ec.get(('H2','B'), 0)}, H2->C: {ec.get(('H2','C'), 0)}")
    print("PASS: nested_loop")


def test_flow_conservation():
    """Verify flow conservation on a larger graph."""
    blocks = ["E", "A", "B", "C", "D", "X"]
    edges = {
        "E": ["A"],
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["D"],
        "D": ["X"],
    }
    samples = {"E": 50, "A": 50, "B": 35, "C": 15, "D": 50, "X": 50}
    ec, freq = propagate(blocks, edges, samples, "E")

    for bb in blocks:
        if bb == "E":
            continue
        in_sum = sum(ec.get((p, bb), 0) for p in
                     [b for b, ss in edges.items() if bb in ss])
        out_sum = sum(ec.get((bb, s), 0) for s in edges.get(bb, []))
        if edges.get(bb):
            assert abs(in_sum - out_sum) <= max(in_sum, out_sum) * 0.1 + 1, \
                f"Flow violation at {bb}: in={in_sum} out={out_sum}"
    print("PASS: flow_conservation")


if __name__ == "__main__":
    test_linear()
    test_diamond()
    test_simple_loop()
    test_nested_loop()
    test_flow_conservation()
    print("\nAll tests passed.")
