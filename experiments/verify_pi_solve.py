import sys, random, time
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from seven_hearts import deal_random_hands, initial_state_for_hands, FullInformationState, solve_full_information
from pi_solve import Solver, owner_from_hands
mismatch = 0; total = 0
for cps in (3, 4, 5, 6):
    t = time.perf_counter(); n = 300 if cps <= 5 else 40
    for seed in range(n):
        hands = deal_random_hands(random.Random(1000 * cps + seed), cards_per_suit=cps)
        state = initial_state_for_hands(hands)
        ref = solve_full_information(FullInformationState.from_hands(hands, state))
        ref_w = max(range(4), key=lambda p: ref.value[p])
        assert sorted(ref.value) == [0.0, 0.0, 0.0, 1.0], ref.value
        w = Solver(owner_from_hands(hands)).solve_root()
        total += 1
        if w != ref_w:
            mismatch += 1
            print("MISMATCH", cps, seed, w, ref_w)
    print(f"cards_per_suit={cps}: {n} deals checked in {time.perf_counter()-t:.1f}s", flush=True)
print("total", total, "mismatches", mismatch)
