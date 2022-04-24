from classification_model.config.core import config
from classification_model.processing.features import ExtractLetterTransformer


def test_extract_letter_transformer(sample_input_data):
    # Given
    transformer = ExtractLetterTransformer(
        variables=config.model_config.variables_for_custom_processing
    )

    assert sample_input_data["cabin"].iat[84] == "E10"

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject["cabin"].iat[84] == "E"
