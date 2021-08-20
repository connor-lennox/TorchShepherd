def progress_string(done, total, bar_length=16, include_count=True):
    num_filled = int(done / total * bar_length)
    num_unfilled = bar_length - num_filled
    bar = "[" + "=" * num_filled
    if num_unfilled != 0:
        bar += '>'
    bar += " " * (num_unfilled-1) + "]"
    if include_count:
        bar += f" [{done}/{total}]"
    return bar