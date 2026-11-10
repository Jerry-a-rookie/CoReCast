                      
                                                                
from pathlib import Path
import runpy

if __name__ == '__main__':
    print('RUNNER_DISPATCH gpu_cached', flush=True)
    runpy.run_path(str(Path(__file__).resolve().with_name('run_component_raft_gpu.py')), run_name='__main__')
