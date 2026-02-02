#!/usr/bin/env python3
"""
Test script to verify the 3D panel label fix works correctly
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass


@dataclass
class TestConfig:
    """Minimal configuration for testing"""
    TITLE_SIZE: int = 12
    COLORS: dict = None
    
    def __post_init__(self):
        if self.COLORS is None:
            self.COLORS = {'text_dark': '#1A1A1A'}


def add_panel_label_old(ax, label, cfg, x=-0.08, y=1.06):
    """OLD VERSION - This will fail for 3D axes"""
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=cfg.TITLE_SIZE + 2, fontweight='bold',
            color=cfg.COLORS['text_dark'],
            verticalalignment='top', horizontalalignment='left')


def add_panel_label_fixed(ax, label, cfg, x=-0.08, y=1.06):
    """FIXED VERSION - Works for both 2D and 3D axes"""
    if hasattr(ax, 'text2D'):
        # For 3D axes, use text2D which works with 2D coordinates
        ax.text2D(x, y, f'({label})', 
                  transform=ax.transAxes,
                  fontsize=cfg.TITLE_SIZE + 2, 
                  fontweight='bold',
                  color=cfg.COLORS['text_dark'],
                  verticalalignment='top', 
                  horizontalalignment='left')
    else:
        # For 2D axes, use regular text
        ax.text(x, y, f'({label})', 
                transform=ax.transAxes,
                fontsize=cfg.TITLE_SIZE + 2, 
                fontweight='bold',
                color=cfg.COLORS['text_dark'],
                verticalalignment='top', 
                horizontalalignment='left')


def test_2d_axis():
    """Test with 2D axis"""
    print("Testing 2D axis...")
    cfg = TestConfig()
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot something simple
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x))
    ax.set_title("2D Plot")
    
    # Try the fixed version
    try:
        add_panel_label_fixed(ax, 'a', cfg)
        print("✅ 2D axis: PASSED")
        return True
    except Exception as e:
        print(f"❌ 2D axis: FAILED - {e}")
        return False
    finally:
        plt.close(fig)


def test_3d_axis_old():
    """Test 3D axis with OLD (broken) function"""
    print("\nTesting 3D axis with OLD function...")
    cfg = TestConfig()
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot something simple
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(0, 1, 100)
    x = np.sin(theta)
    y = np.cos(theta)
    ax.plot(x, y, z)
    ax.set_title("3D Plot")
    
    # Try the old version (should fail)
    try:
        add_panel_label_old(ax, 'b', cfg)
        print("❌ 3D axis OLD: Unexpected success (should have failed)")
        plt.close(fig)
        return False
    except TypeError as e:
        print(f"✅ 3D axis OLD: FAILED as expected - {e}")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"❓ 3D axis OLD: Unexpected error - {e}")
        plt.close(fig)
        return False


def test_3d_axis_fixed():
    """Test 3D axis with FIXED function"""
    print("\nTesting 3D axis with FIXED function...")
    cfg = TestConfig()
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot something simple
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(0, 1, 100)
    x = np.sin(theta)
    y = np.cos(theta)
    ax.plot(x, y, z)
    ax.set_title("3D Plot")
    
    # Try the fixed version
    try:
        add_panel_label_fixed(ax, 'c', cfg, x=-0.05, y=1.02)
        print("✅ 3D axis FIXED: PASSED")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"❌ 3D axis FIXED: FAILED - {e}")
        plt.close(fig)
        return False


def test_mixed_layout():
    """Test with mixed 2D and 3D axes in same figure"""
    print("\nTesting mixed 2D and 3D axes...")
    cfg = TestConfig()
    fig = plt.figure(figsize=(12, 4))
    
    # Create 2D subplot
    ax1 = fig.add_subplot(131)
    x = np.linspace(0, 10, 100)
    ax1.plot(x, np.sin(x))
    ax1.set_title("2D Plot 1")
    
    # Create 3D subplot
    ax2 = fig.add_subplot(132, projection='3d')
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(0, 1, 100)
    ax2.plot(np.sin(theta), np.cos(theta), z)
    ax2.set_title("3D Plot")
    
    # Create another 2D subplot
    ax3 = fig.add_subplot(133)
    ax3.plot(x, np.cos(x))
    ax3.set_title("2D Plot 2")
    
    # Try adding labels to all
    try:
        add_panel_label_fixed(ax1, 'a', cfg)
        add_panel_label_fixed(ax2, 'b', cfg, x=-0.05, y=1.02)
        add_panel_label_fixed(ax3, 'c', cfg)
        print("✅ Mixed layout: PASSED")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"❌ Mixed layout: FAILED - {e}")
        plt.close(fig)
        return False


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("3D PANEL LABEL FIX - TEST SUITE")
    print("="*60)
    
    results = []
    
    results.append(("2D Axis", test_2d_axis()))
    results.append(("3D Axis (Old)", test_3d_axis_old()))
    results.append(("3D Axis (Fixed)", test_3d_axis_fixed()))
    results.append(("Mixed Layout", test_mixed_layout()))
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:20s} {status}")
    
    all_passed = all(result for _, result in results)
    
    print("="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The fix is working correctly.")
    else:
        print("⚠️  SOME TESTS FAILED! Please review the errors above.")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
