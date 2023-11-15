def format_possibility(pos):
    return '(' + ' '.join(p.nameWithOctave.ljust(3) for p in reversed(pos)) + ')'
