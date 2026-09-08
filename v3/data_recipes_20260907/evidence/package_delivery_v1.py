"""Package already verified metadata; no training or scientific score access."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

if os.name == 'nt':
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

root = Path('D:/SER-final-program-20260907/data_recipes')
repo = Path('C:/Users/jock8/Documents/ChatGPT/论文/SER-speaker-execution-20260906')
source = root / 'build_v1'
delivery = root / 'delivery_v1'
archive = root / 'study2_recording_recipes_v1.zip'
accepted = root / 'delivery_accept_v1'
example = root / 'delivery_example_v1'
assert all(not p.exists() for p in (delivery, archive, accepted, example))
assert hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest() == '1115dcfc0ce5512a9968a6f7a951fa34ef0a484e0a310b481d0f1da03533c48d'
delivery.mkdir()
shutil.copytree(source, delivery/'data')
(delivery/'tools').mkdir()
shutil.copyfile(repo/'v3/data_recipes_20260907/recipe_bundle.py', delivery/'tools/recipe_bundle.py')
(delivery/'README.md').write_text('''# Study II 原录音配置 / Original recording configurations

本包保存已经执行的1,440个数据/模型配置。无需原仓库路径即可校验并导出单个配置。仅需Python标准库，不用GPU。

在解压后的本目录执行：

```shell
python tools/recipe_bundle.py verify --bundle data
python tools/recipe_bundle.py export --bundle data --unit-id e890d24810c9f91a11f1fcf8b78d5904d497b78ed0bda3b75b4642b335fe5fc5 --out example
```

示例只是原计划首项，不是推荐配置。全部原单位在data/units.json；720个fit、90个val、90个test清单按原顺序去重引用。导出fit.csv/val.csv/test.csv携带原录音相对路径、SHA、人物、文本、情绪、强度和来源性别元数据；unit.json保留原配置及种子。val是早停验证，test是最终测试。每次选择不存在或为空的输出目录。

本包不含音频、缓存、权重或预测数组。verify核对当前包和原计划身份，不重新训练，也不复算科学结果；真实音频尚需使用者逐文件哈希核对。历史验收记录及其原电脑路径按原字节保存，不代表在当前机器重新执行了该验收。

人数12/48和每人文本覆盖同时变化；288/576是拟合录音条数预算。prompt_seen/new指文本，最终测试人物均未参与拟合。不同划分不得合并后仍声称测试未见人。这些配置未证明通用最优或SD/SI差距消除。

This metadata-only package losslessly reconstructs all 1,440 executed Study II units from 720 fit, 90 validation, and 90 test lists. The example is the first original unit, not an outcome-selected recommendation. Python's standard library is sufficient. No audio, caches, weights, or prediction arrays are distributed. Package validation is not retraining, audio verification, or scientific result recomputation. Original historical receipts are preserved without rebinding their paths. Speaker count and per-speaker prompt coverage vary jointly; no optimal dataset or elimination of SD/SI gaps is established.

The delivery manifest records this packaging operation after the original study's results were known. It is not a new preregistration. The original plan lock is preserved inside data/originals. The outer ZIP SHA must be shared separately when transferring the package.
''', encoding='utf-8')

def file_identity(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

manifest = {'schema':'ser-study2-recipe-delivery-1', 'metadata_only':True,
            'inner_manifest_sha256':file_identity(delivery/'data/manifest.json')['sha256'],
            'files':{p.relative_to(delivery).as_posix():file_identity(p)
                     for p in sorted(delivery.rglob('*')) if p.is_file()}}
(delivery/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
    for path in sorted(delivery.rglob('*')):
        if path.is_file():
            output.write(path, path.relative_to(delivery).as_posix())
with zipfile.ZipFile(archive) as incoming:
    names = incoming.namelist()
    assert len(names) == len(set(names)) and set(names) == set(manifest['files']) | {'DELIVERY_MANIFEST.json'}
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts and ':' not in n and '\\' not in n for n in names)
    incoming.extractall(accepted)
for name, expected in manifest['files'].items():
    assert file_identity(accepted/name) == expected
assert (accepted/'DELIVERY_MANIFEST.json').read_bytes() == (delivery/'DELIVERY_MANIFEST.json').read_bytes()
base = [sys.executable, '-X', 'utf8', '-B', str(accepted/'tools/recipe_bundle.py')]
common = dict(check=True, capture_output=True, encoding='utf-8', creationflags=0x4000 if os.name=='nt' else 0)
verified = json.loads(subprocess.run(base+['verify','--bundle',str(accepted/'data')], **common).stdout)
exported = json.loads(subprocess.run(base+['export','--bundle',str(accepted/'data'),'--unit-id','e890d24810c9f91a11f1fcf8b78d5904d497b78ed0bda3b75b4642b335fe5fc5','--out',str(example)], **common).stdout)
report = {'pass':True,'archive':str(archive),**file_identity(archive),'delivery_files':len(names),
          'tool_sha256':file_identity(accepted/'tools/recipe_bundle.py')['sha256'],
          'fresh_process_relocated_verify':verified,'fresh_process_export':exported}
(root/'delivery_acceptance_v1.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('fresh_process_relocated_verify','fresh_process_export')}))
