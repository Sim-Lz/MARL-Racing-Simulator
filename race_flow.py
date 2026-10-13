from config import POST_RACE_DELAY_MS


def should_exit_early(state, key, escape_key):
    return state == "POST_RACE" and key == escape_key


def should_auto_start_next_race(state, elapsed_ms, auto_advance=True):
    return auto_advance and state == "POST_RACE" and elapsed_ms >= POST_RACE_DELAY_MS
