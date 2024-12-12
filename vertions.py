import subprocess

def get_installed_version(package_name):
    try:
        result = subprocess.run(['pip', 'show', package_name], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if line.startswith('Version:'):
                return line.split()[1]
    except Exception as e:
        return str(e)

with open('requirements.txt', 'r') as file:
    packages = file.readlines()

for package in packages:
    package_name = package.split('==')[0].strip()
    version = get_installed_version(package_name)
    print(f'{package_name}: {version}')