import yaml

path = r"c:\Users\Зефирка\StartUp\configs\sandbox.yaml"
with open(path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

print(yaml.safe_dump(data, allow_unicode=True))
