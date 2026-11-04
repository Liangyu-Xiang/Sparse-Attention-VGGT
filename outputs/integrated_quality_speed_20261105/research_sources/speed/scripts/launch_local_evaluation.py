"""Select the repository's benchmarks namespace over the unrelated pip package."""
from pathlib import Path
import runpy,sys,types
script=Path(sys.argv[1]).resolve();root=Path.cwd()
module=types.ModuleType('benchmarks');module.__path__=[str(root/'evaluation/benchmarks')]
sys.modules['benchmarks']=module
sys.argv=sys.argv[1:]
runpy.run_path(str(script),run_name='__main__')
