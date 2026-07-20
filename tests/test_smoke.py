from nowak_coordination.smoke import run_smoke


def test_cpu_smoke_is_deterministic_and_covers_all_arms():
    first = run_smoke(num_episodes=25, seed=123)
    second = run_smoke(num_episodes=25, seed=123)
    assert first == second
    assert set(first["mean_reward_by_model"]) == {"A", "B", "C", "D", "E"}
    # Group episodes contribute one pairwise outcome per peer and round.
    assert sum(first["outcome_counts"].values()) > 25 * 6
