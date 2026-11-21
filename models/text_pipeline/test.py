import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.pipeline import parser,run_test
if __name__=='__main__':
    args=parser('text','test').parse_args(); run_test('text',args)
