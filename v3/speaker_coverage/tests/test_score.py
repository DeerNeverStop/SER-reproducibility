import numpy as np
import pytest
from v3.speaker_coverage.score import uar, paired_estimate


def test_pooled_recall_is_not_average_small_rotation_recalls():
    a_correct=np.array([1,1,1,1,1,1]); a_support=np.array([1,1,1,1,1,1])
    b_correct=np.zeros(6); b_support=np.full(6,3)
    assert uar(a_correct+b_correct,a_support+b_support)==25.0
    assert (uar(a_correct,a_support)+uar(b_correct,b_support))/2==50.0


def test_missing_class_is_not_silently_dropped():
    with pytest.raises(ValueError): uar(np.zeros(6),np.array([1,1,1,1,1,0]))


def test_paired_interval_uses_people_after_repeat_average():
    first=np.array([[12.,30.,50.],[10.,32.,48.]])
    second=np.array([[10.,28.,48.],[8.,30.,46.]])
    idx=np.array([[0,0,1],[2,1,2],[0,1,2]])
    result=paired_estimate(first,second,idx)
    assert result['difference_pp']==2.0
    assert result['ci95_low_pp']==result['ci95_high_pp']==2.0
    assert result['repeat_differences_pp']==[2.0,2.0]


def test_unpaired_repeat_arrays_rejected():
    with pytest.raises(ValueError): paired_estimate(np.ones((2,3)),np.ones((3,3)),np.ones((10,3),dtype=int))
