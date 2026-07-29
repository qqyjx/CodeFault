import re

with open('experiments/run_all_experiments.py', 'r') as f:
    content = f.read()

# 添加自定义JSON encoder在import之后
import_section = '''import warnings
warnings.filterwarnings('ignore')'''

new_import = '''import warnings
warnings.filterwarnings('ignore')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return super().default(obj)'''

content = content.replace(import_section, new_import)

# 替换所有json.dump调用
content = re.sub(r'json\.dump\(([^,]+), f, indent=2\)', r'json.dump(\1, f, indent=2, cls=NumpyEncoder)', content)
content = re.sub(r'json\.dump\(([^,]+), f, indent=2, ensure_ascii=False\)', r'json.dump(\1, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)', content)

with open('experiments/run_all_experiments.py', 'w') as f:
    f.write(content)

print("已修复所有json.dump调用")
