def generate_options_string(question: str, options: list[str]) -> str:
    """Generate a string with a question and numbered options.

    Args:
        question: Prompt text shown above options.
        options: Available option labels in display order.
    """
    assert question, "Question cannot be empty"
    assert options, "Options cannot be empty"
    for option in options:
        assert option, "Option cannot be empty"

    options_str = "\n".join(
        f"[{i+1}] {option.capitalize()}" for i, option in enumerate(options))

    return question + "\n" + options_str
