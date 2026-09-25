from dataclasses import replace
import math

import model as m


def close(x, y, tol=5e-7):
    return math.isclose(float(x), float(y), rel_tol=0.0, abs_tol=tol)


def first_time(series, threshold):
    idx = series.index[series >= threshold]
    return int(idx[0]) if len(idx) else None


def main():
    p = m.ModelParams()
    base = m.run_model(p, depth_mode="endogenous")
    h = base.history
    checks = m.validate_against_theory(base)

    assert close(checks["A_star_theory"], 6.25)
    assert checks["first_floor_period"] == 66
    assert checks["final_adoption_matches_A_star"]
    assert checks["number_of_final_adopters"] == 85
    assert close(checks["long_run_VS"], 0.7958757905405266)
    assert close(checks["long_run_CS"], 0.425)
    assert close(checks["maximum_final_nonadopter_size"], 6.232221, tol=1e-5)
    assert close(checks["minimum_final_adopter_size"], 6.293612, tol=1e-5)

    fixed = m.run_model(p, sizes=base.sizes, depth_mode="fixed")
    hf = fixed.history

    floor = p.kappa * p.S_USD
    on_floor = h.index[(h["S_dir"] - floor).abs() < 1e-12]
    off_floor = hf.index[(hf["S_dir"] - floor).abs() < 1e-12]
    assert int(on_floor[0]) == 66
    assert int(off_floor[0]) == 73
    assert first_time(h["VS"], 0.50) == 64
    assert first_time(hf["VS"], 0.50) == 66
    assert first_time(h["CS"], 0.25) == 64
    assert first_time(hf["CS"], 0.25) == 68

    for seed_depth, first_adoption, vs50, cs25 in [
        (10.0, 72, 73, 73),
        (50.0, 63, 64, 64),
        (200.0, 61, 62, 63),
    ]:
        pp = replace(p, V_seed=seed_depth)
        r = m.run_model(pp)
        hh = r.history
        idx = hh.index[hh["CS"] > 0]
        assert int(idx[0]) == first_adoption
        assert first_time(hh["VS"], 0.50) == vs50
        assert first_time(hh["CS"], 0.25) == cs25
        assert close(hh.tail(pp.long_run_window)["VS"].mean(), 0.7958757905405266)
        assert close(hh.tail(pp.long_run_window)["CS"].mean(), 0.425)

    pareto_expected = {
        1.5: (0.901806, 0.545),
        2.0: (0.795876, 0.425),
        3.0: (0.573911, 0.250),
    }
    for a, (vs_exp, cs_exp) in pareto_expected.items():
        pp = replace(p, pareto_a=a)
        r = m.run_model(pp)
        hh = r.history.tail(pp.long_run_window)
        assert close(hh["VS"].mean(), vs_exp, tol=1e-6)
        assert close(hh["CS"].mean(), cs_exp, tol=1e-6)

    print("All replication checks passed.")


if __name__ == "__main__":
    main()
