from pathlib import Path

from delfos.scip.reader import ScipIndex

idx = ScipIndex(Path("index.scip"))
print(idx.files[:5])

