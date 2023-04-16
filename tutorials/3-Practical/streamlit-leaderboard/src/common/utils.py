

def remove_illegal_filename_characters(input_string: str) -> str:
    return "".join(x if (x.isalnum() or x in "._- ") else '_' for x in input_string).strip()


def is_legal_filename(filename: str) -> bool:
    return remove_illegal_filename_characters(filename) == filename