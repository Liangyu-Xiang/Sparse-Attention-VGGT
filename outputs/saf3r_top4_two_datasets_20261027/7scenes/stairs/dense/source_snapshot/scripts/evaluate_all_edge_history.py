"""Use unchanged final attention math with disclosed environment compatibility."""
import runpy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.unified_diagnostic_runtime import install
install(diagnostic_reporting=False)
if __name__=='__main__':
    runpy.run_path(str(ROOT/'scripts/evaluate_reuse_exploration.py'),run_name='__main__')
