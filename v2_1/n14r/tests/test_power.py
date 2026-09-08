import json
from pathlib import Path

import pytest

from v2_1.n14r.power import planning_record, two_sided_one_sample_t_power


def test_frozen_power_choice_and_static_record_agree() -> None:
    record = planning_record()
    frozen_path = Path(__file__).parents[1] / "power_analysis.json"
    frozen_text = frozen_path.read_text(encoding="utf-8")
    frozen = json.loads(frozen_text)
    assert record == frozen
    assert json.dumps(record, indent=1, sort_keys=True) + "\n" == frozen_text
    assert record["pilot"]["confirmatory_inclusion"] is False
    assert record["design"]["first_n_reaching_target"] == 24
    assert record["design"]["primary_planned_units_before_retries"] == 1920
    assert record["design"]["maximum_unique_planned_units_before_retries"] == 2240
    assert record["design"]["theoretical_maximum_training_attempts"] == 4480
    assert record["design"]["projected_primary_power"] == pytest.approx(
        frozen["design"]["projected_primary_power"], abs=1e-12
    )
    assert frozen["resource_limit_is_scientific_stopping_rule"] is False


def test_n23_is_below_target_and_n24_reaches_it() -> None:
    assert two_sided_one_sample_t_power(23, 1.5, 2.5, 0.05) < 0.80
    assert two_sided_one_sample_t_power(24, 1.5, 2.5, 0.05) >= 0.80
