#!/usr/bin/env python3
"""
?????????
"""

import sys
from pathlib import Path

# ?????????
sys.path.insert(0, str(Path(__file__).parent))

try:
    from load_material_data import MaterialDataLoader
    import pandas as pd
    print("? ????????\n")
except ImportError as e:
    print(f"? ????: {e}")
    print("?????????Python?: pandas, numpy, matplotlib, scipy")
    sys.exit(1)

def test_data_loading():
    """????????"""
    print("=" * 60)
    print("???????")
    print("=" * 60)
    
    loader = MaterialDataLoader()
    
    tests_passed = 0
    tests_total = 0
    
    # ??1: ??????
    tests_total += 1
    try:
        green_props = loader.load_green_state_properties()
        assert len(green_props) > 0
        print(f"? ??1: ?????? - {len(green_props)} ???")
        tests_passed += 1
    except Exception as e:
        print(f"? ??1??: {e}")
    
    # ??2: ??????
    tests_total += 1
    try:
        dilatometry = loader.load_dilatometry_data("8YSZ")
        assert len(dilatometry) > 0
        print(f"? ??2: ?????? (8YSZ) - {len(dilatometry)} ???")
        tests_passed += 1
    except Exception as e:
        print(f"? ??2??: {e}")
    
    # ??3: CTE??
    tests_total += 1
    try:
        cte = loader.load_CTE_data("8YSZ")
        assert len(cte) > 0
        print(f"? ??3: CTE?? (8YSZ) - {len(cte)} ???")
        tests_passed += 1
    except Exception as e:
        print(f"? ??3??: {e}")
    
    # ??4: ????
    tests_total += 1
    try:
        creep_params = loader.load_norton_parameters()
        assert len(creep_params) > 0
        print(f"? ??4: Norton???? - {len(creep_params)} ???")
        tests_passed += 1
    except Exception as e:
        print(f"? ??4??: {e}")
    
    # ??5: ???????
    tests_total += 1
    try:
        strain_rate = loader.get_creep_strain_rate("8YSZ", "Green", 1200, 10)
        assert strain_rate > 0
        print(f"? ??5: ??????? - {strain_rate:.2e} s??")
        tests_passed += 1
    except Exception as e:
        print(f"? ??5??: {e}")
    
    # ??6: ????
    tests_total += 1
    try:
        youngs = loader.load_youngs_modulus("8YSZ")
        assert len(youngs) > 0
        print(f"? ??6: ?????? (8YSZ) - {len(youngs)} ???")
        tests_passed += 1
    except Exception as e:
        print(f"? ??6??: {e}")
    
    # ??7: ???
    tests_total += 1
    try:
        poissons = loader.load_poissons_ratio("8YSZ")
        assert len(poissons) > 0
        print(f"? ??7: ????? (8YSZ) - {len(poissons)} ???")
        tests_passed += 1
    except Exception as e:
        print(f"? ??7??: {e}")
    
    print("\n" + "=" * 60)
    print(f"????: {tests_passed}/{tests_total} ??")
    print("=" * 60)
    
    if tests_passed == tests_total:
        print("\n? ?????????????????")
        return True
    else:
        print(f"\n? {tests_total - tests_passed} ?????????????")
        return False

def show_data_summary():
    """??????"""
    print("\n" + "=" * 60)
    print("?????")
    print("=" * 60)
    
    loader = MaterialDataLoader()
    
    # ???????
    print("\n??????:")
    
    categories = {
        "?????": [
            ("??????", lambda: loader.load_green_state_properties()),
            ("????-8YSZ", lambda: loader.load_dilatometry_data("8YSZ")),
            ("????-NiYSZ", lambda: loader.load_dilatometry_data("NiYSZ")),
            ("????", lambda: loader.load_dilatometry_data("bilayer_anode_electrolyte")),
            ("CTE??", lambda: loader.load_CTE_data()),
        ],
        "????": [
            ("????-8YSZ", lambda: loader.load_creep_test_data("8YSZ")),
            ("????-NiYSZ", lambda: loader.load_creep_test_data("NiYSZ")),
            ("Norton??", lambda: loader.load_norton_parameters()),
        ],
        "????": [
            ("????", lambda: loader.load_youngs_modulus()),
            ("???", lambda: loader.load_poissons_ratio()),
        ]
    }
    
    for category, tests in categories.items():
        print(f"\n{category}:")
        for name, load_func in tests:
            try:
                df = load_func()
                print(f"  - {name}: {len(df)} ???")
            except Exception as e:
                print(f"  - {name}: ?? - {e}")

if __name__ == "__main__":
    success = test_data_loading()
    show_data_summary()
    
    if success:
        print("\n????????????????FEM???")
        sys.exit(0)
    else:
        print("\n????????????????")
        sys.exit(1)
