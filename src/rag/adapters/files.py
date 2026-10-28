"""Facts about files on disk that the document loaders share."""

import datetime as dt
from pathlib import Path

from rag.domain.models import DateSource


def file_created(path: Path) -> tuple[dt.date, DateSource] | tuple[None, None]:
    """The date the file was created on this disk, where the operating system records one.

    macOS and Windows keep a creation (birth) time; most Linux file systems do not expose
    one, and then there is no date. The modification time is never used instead: it changes
    with every edit, so it would only be a guess at when the document was created.
    """
    birth = getattr(path.stat(), "st_birthtime", None)
    if not isinstance(birth, float | int):
        return None, None
    local = dt.datetime.fromtimestamp(birth, tz=dt.UTC).astimezone()
    return local.date(), DateSource.FILE_SYSTEM
