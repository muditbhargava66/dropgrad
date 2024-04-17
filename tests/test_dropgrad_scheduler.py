import torch
from dropgrad import LinearDropRateScheduler, CosineAnnealingDropRateScheduler, StepDropRateScheduler

def test_linear_drop_rate_scheduler():
    initial_drop_rate = 0.5
    final_drop_rate = 0.1
    num_steps = 100
    scheduler = LinearDropRateScheduler(initial_drop_rate, final_drop_rate, num_steps)

    assert scheduler.get_drop_rate() == initial_drop_rate

    for step in range(num_steps):
        drop_rate = scheduler.get_drop_rate()
        assert initial_drop_rate >= drop_rate >= final_drop_rate
        scheduler.step()

    assert scheduler.get_drop_rate() == final_drop_rate

def test_cosine_annealing_drop_rate_scheduler():
    initial_drop_rate = 0.5
    final_drop_rate = 0.1
    num_steps = 100
    scheduler = CosineAnnealingDropRateScheduler(initial_drop_rate, final_drop_rate, num_steps)

    assert scheduler.get_drop_rate() == initial_drop_rate

    for step in range(num_steps):
        drop_rate = scheduler.get_drop_rate()
        assert final_drop_rate <= drop_rate <= initial_drop_rate
        scheduler.step()

    assert scheduler.get_drop_rate() == final_drop_rate

def test_step_drop_rate_scheduler():
    initial_drop_rate = 0.5
    drop_rate_schedule = {
        50: 0.3,
        80: 0.1
    }
    scheduler = StepDropRateScheduler(initial_drop_rate, drop_rate_schedule)

    assert scheduler.get_drop_rate() == initial_drop_rate

    for step in range(100):
        drop_rate = scheduler.get_drop_rate()
        if step < 50:
            assert drop_rate == initial_drop_rate
        elif step < 80:
            assert drop_rate == 0.3
        else:
            assert drop_rate == 0.1
        scheduler.step()