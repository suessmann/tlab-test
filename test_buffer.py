import pytest
import torch

from buffer import ReplayBuffer, TensorDeque


@pytest.fixture
def empty_deque():
    return TensorDeque(10, 1, torch.float32, "cpu")


@pytest.fixture
def full_deque():
    deque = TensorDeque(10, 1, torch.float32, "cpu")
    deque.add_data(torch.ones(10, 1))

    return deque


@pytest.fixture
def one_to_full_deque():
    deque = TensorDeque(10, 1, torch.float32, "cpu")
    deque.add_data(torch.ones(9, 1))

    return deque


@pytest.fixture
def custom_deque():
    deque = TensorDeque(5, 1, torch.float32, "cpu")
    deque.add_data(torch.tensor([1.0, 2.0, 3.0]).reshape(3, 1))

    return deque


def test_empty_deque(empty_deque):
    assert empty_deque.size() == 0
    assert empty_deque._pointer == 0


def test_full_deque(full_deque):
    assert full_deque.size() == 10
    assert full_deque._pointer == 0
    assert full_deque._is_full


def test_one_to_full_deque(one_to_full_deque):
    to_add = torch.ones(2, 1)
    assert one_to_full_deque.size() == 9
    assert one_to_full_deque._pointer == 9
    assert not one_to_full_deque._is_full

    one_to_full_deque.add_data(to_add)

    assert one_to_full_deque.size() == 10
    assert one_to_full_deque._pointer == 1
    assert one_to_full_deque._is_full


def test_custom_deque(custom_deque):
    assert custom_deque.size() == 3
    assert custom_deque._pointer == 3
    assert not custom_deque._is_full

    to_add = torch.tensor([4.0, 5.0, 6.0]).reshape(3, 1)
    custom_deque.add_data(to_add)

    assert custom_deque.size() == 5
    assert custom_deque._pointer == 1
    assert custom_deque._is_full
    assert all(
        custom_deque._data == torch.tensor([6.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
    )
