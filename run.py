from zenml import step, pipeline


@step
def basic_step() -> str:
    """A simple step that returns a greeting message."""
    return "Hello World!"


@pipeline
def basic_pipeline() -> str:
    """A simple pipeline with just one step."""
    greeting = basic_step()
    return greeting


if __name__ == "__main__":
    basic_pipeline()