import logging


def configure_logging() -> None:
    """Configure concise process-wide logging for local and container runs."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
