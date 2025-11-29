
import sys
import os
import importlib

# Create a dummy package structure
os.makedirs("dummy_pkg", exist_ok=True)
with open("dummy_pkg/__init__.py", "w") as f:
    f.write("from .module import MyClass\n")

with open("dummy_pkg/module.py", "w") as f:
    f.write("class MyClass:\n    def method1(self):\n        pass\n")

# 1. Import package and class
import dummy_pkg
from dummy_pkg import MyClass
obj1 = MyClass()
print(f"Obj1 has method2? {hasattr(obj1, 'method2')}")

# 2. Modify module to add method2
with open("dummy_pkg/module.py", "w") as f:
    f.write("class MyClass:\n    def method1(self):\n        pass\n    def method2(self):\n        pass\n")

# 3. Reload module only
import dummy_pkg.module
importlib.reload(dummy_pkg.module)

# 4. Import from package again (simulating notebook behavior)
from dummy_pkg import MyClass as MyClass2
obj2 = MyClass2()
print(f"Obj2 (from pkg) has method2? {hasattr(obj2, 'method2')}")

# 5. Import from module directly
from dummy_pkg.module import MyClass as MyClass3
obj3 = MyClass3()
print(f"Obj3 (from module) has method2? {hasattr(obj3, 'method2')}")

# 6. Reload package
importlib.reload(dummy_pkg)
from dummy_pkg import MyClass as MyClass4
obj4 = MyClass4()
print(f"Obj4 (after pkg reload) has method2? {hasattr(obj4, 'method2')}")

# Cleanup
import shutil
shutil.rmtree("dummy_pkg")
