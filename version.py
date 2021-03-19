if __name__ == '__main__':
    version_namespace = {}
    with open('src/qtt/version.py', 'r') as f:
        exec(f.read(), version_namespace)
    print(version_namespace['__version__'])
