from tinylm.context import Context, current_context, get_context


def test_context():
    ctx = Context()
    with current_context(ctx):
        assert get_context() is ctx
