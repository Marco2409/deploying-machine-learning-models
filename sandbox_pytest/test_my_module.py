from my_module import square

import pytest

@pytest.fixture
def input_value():
	return 4

def test_square_gives_correct_value(input_value):
	# When
	subject = square(input_value)

	# Then
	assert subject == 16