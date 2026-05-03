import test_seven_hearts


TESTS = [
    test_seven_hearts.test_closed_suit_only_allows_seven,
    test_seven_hearts.test_after_opening_move_closed_suits_allow_their_sevens,
    test_seven_hearts.test_open_suit_allows_adjacent_cards,
    test_seven_hearts.test_player_legal_moves_are_intersection_of_hand_and_table_legal_cards,
    test_seven_hearts.test_pass_removes_then_legal_cards_from_that_opponent_possibilities,
    test_seven_hearts.test_opponent_play_removes_card_from_other_opponents_possibilities,
    test_seven_hearts.test_multi_pass_removes_cards_legal_at_each_pass_moment,
    test_seven_hearts.test_recommend_move_returns_none_when_forced_to_pass,
    test_seven_hearts.test_recommend_move_works_without_hand_counts,
    test_seven_hearts.test_known_player_count_is_decremented_but_unknown_counts_remain_unknown,
    test_seven_hearts.test_validate_turn_rejects_passing_with_legal_moves,
    test_seven_hearts.test_validate_turn_rejects_non_opening_card_on_empty_table,
    test_seven_hearts.test_endgame_urgency_has_gradient_for_known_counts,
    test_seven_hearts.test_endgame_urgency_is_not_an_active_score_component,
    test_seven_hearts.test_earlier_side_card_releases_more_future_chain_than_later_side_card,
    test_seven_hearts.test_opening_seven_is_better_when_it_controls_both_immediate_gates,
    test_seven_hearts.test_opening_seven_gets_some_credit_for_distant_tail_card,
    test_seven_hearts.test_tail_bonus_uses_tail_distance_not_average_side_distance,
    test_seven_hearts.test_future_chain_impact_handles_ace_and_king_edges,
    test_seven_hearts.test_time_to_playable_counts_required_chain_steps,
    test_seven_hearts.test_holder_probabilities_sum_to_one_for_possible_unseen_cards,
    test_seven_hearts.test_opponent_model_reports_impossible_known_counts,
    test_seven_hearts.test_score_components_are_immutable,
]


def main() -> None:
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(TESTS)} tests passed")


if __name__ == "__main__":
    main()
