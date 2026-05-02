import test_seven_hearts


TESTS = [
    test_seven_hearts.test_closed_suit_only_allows_seven,
    test_seven_hearts.test_open_suit_allows_adjacent_cards,
    test_seven_hearts.test_player_legal_moves_are_intersection_of_hand_and_table_legal_cards,
    test_seven_hearts.test_pass_removes_then_legal_cards_from_that_opponent_possibilities,
    test_seven_hearts.test_recommend_move_returns_none_when_forced_to_pass,
    test_seven_hearts.test_recommend_move_works_without_hand_counts,
    test_seven_hearts.test_known_player_count_is_decremented_but_unknown_counts_remain_unknown,
]


def main() -> None:
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(TESTS)} tests passed")


if __name__ == "__main__":
    main()
