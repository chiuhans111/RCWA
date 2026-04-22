import os
import sys
import inspect
import pkgutil
import re
from dataclasses import dataclass
from typing import List, Dict

# Ensure we can import rcwa
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rcwa

@dataclass
class MemberAudit:
    name: str
    is_covered: bool

@dataclass
class ModuleAudit:
    name: str
    members: List[MemberAudit]
    
    @property
    def coverage(self) -> float:
        if not self.members: return 100.0
        covered = sum(1 for m in self.members if m.is_covered)
        return (covered / len(self.members)) * 100

def get_public_members(module) -> List[str]:
    """Returns a list of public functions and classes in a module."""
    return [
        name for name, obj in inspect.getmembers(module)
        if not name.startswith('_') 
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and inspect.getmodule(obj) == module
    ]

def scan_tests() -> str:
    """Reads all test files into a single searchable string."""
    content = []
    test_dir = os.path.dirname(__file__)
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.endswith(".py") and f != os.path.basename(__file__):
                with open(os.path.join(root, f), 'r', encoding='utf-8') as fh:
                    content.append(fh.read())
    return "\n".join(content)

def audit_package() -> List[ModuleAudit]:
    test_corpus = scan_tests()
    package_path = os.path.dirname(rcwa.__file__)
    results = []

    for _, mod_name, _ in pkgutil.walk_packages([package_path], rcwa.__name__ + '.'):
        try:
            module = __import__(mod_name, fromlist=[''])
            members = get_public_members(module)
            if not members: continue
            
            audit_results = [
                MemberAudit(name, bool(re.search(r'\b' + name + r'\b', test_corpus)))
                for name in sorted(members)
            ]
            results.append(ModuleAudit(mod_name, audit_results))
        except ImportError:
            continue
    return results

def print_report(audits: List[ModuleAudit]):
    print(f"\n{'='*60}")
    print(f"{'RCWA API COVERAGE REPORT':^60}")
    print(f"{'='*60}\n")

    for mod in sorted(audits, key=lambda x: x.name):
        covered = [m.name for m in mod.members if m.is_covered]
        missing = [m.name for m in mod.members if not m.is_covered]
        
        print(f"MODULE: {mod.name}")
        print(f"  Status: {mod.coverage:5.1f}% | {len(covered)}/{len(mod.members)} members")
        
        if missing:
            print(f"  Missing: {', '.join(missing)}")
        else:
            print(f"  [Perfect Coverage]")
        print("-" * 60)

    # Summary Table
    total_m = sum(len(m.members) for m in audits)
    total_c = sum(sum(1 for memb in m.members if memb.is_covered) for m in audits)
    overall_pct = (total_c / total_m * 100) if total_m > 0 else 0

    print(f"\nSUMMARY")
    print(f"  Total API Members: {total_m}")
    print(f"  Total Referenced:  {total_c}")
    print(f"  Overall Coverage:  {overall_pct:.1f}%")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    audits = audit_package()
    print_report(audits)
