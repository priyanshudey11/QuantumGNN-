"""
Quick check to see if the model fix is actually loaded
"""

import sys
import importlib

# Force reload the module
if 'ligand_pocket_qgnn.model' in sys.modules:
    print("⚠️  Model module already loaded - reloading...")
    importlib.reload(sys.modules['ligand_pocket_qgnn.model'])
else:
    print("✓ Model module not yet loaded")

# Try to import
try:
    from ligand_pocket_qgnn.model import QuantumInteractionLayer
    print("✓ Import successful")

    # Check the source code to see if ExecutionConfig is still there
    import inspect
    source = inspect.getsource(QuantumInteractionLayer.__init__)

    if 'ExecutionConfig' in source:
        print("\n❌ BUG STILL PRESENT!")
        print("The model.py file still has ExecutionConfig")
        print("You need to RESTART THE JUPYTER KERNEL!")
        print("\nIn Jupyter: Kernel → Restart Kernel")
    else:
        print("\n✓ BUG IS FIXED!")
        print("The model.py file has been updated correctly")

    # Show the relevant line
    print("\n" + "="*70)
    print("CURRENT CODE:")
    print("="*70)
    for i, line in enumerate(source.split('\n')[15:35], 15):
        print(f"{i:3d}: {line}")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("Cannot import model - check if ligand_pocket_qgnn package exists")
