from scripts.run_dense_ratio_search import select_recommendation


def row(name,seconds,error,complete=True,f1=0):
    return dict(name=name,seconds=seconds,teacher_train_nmse=error,
        speedup_vs_vggt=100/seconds,quality=dict(complete=complete,values=dict(f1=f1)))


def test_speed_guard_is_not_silently_relaxed_for_better_teacher_error():
    rows=[row('manual',60,1),row('slow',70,.1),row('fast',55,.2)]
    assert select_recommendation(rows)=='fast'
    assert select_recommendation(rows[:2]) is None


def test_ground_truth_quality_never_selects_oracle_ratio():
    rows=[row('manual',60,1),row('teacher_best',55,.1,f1=.1),row('gt_best',50,.2,f1=.99)]
    assert select_recommendation(rows)=='teacher_best'
    assert select_recommendation([rows[0],row('incomplete',50,.01,complete=False)]) is None
