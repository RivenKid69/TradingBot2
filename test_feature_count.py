#!/usr/bin/env python3
"""Test feature count with different max_num_tokens"""

from feature_config import make_layout

# Test with max_num_tokens=1
layout1 = make_layout({'max_num_tokens': 1})
total1 = sum(b['size'] for b in layout1)
print('With max_num_tokens=1:')
print(f'  Total features: {total1}')
for b in layout1:
    print(f'  {b["name"]}: {b["size"]}')
print()

# Test with max_num_tokens=16 (default)
layout16 = make_layout({'max_num_tokens': 16})
total16 = sum(b['size'] for b in layout16)
print('With max_num_tokens=16 (default):')
print(f'  Total features: {total16}')
for b in layout16:
    print(f'  {b["name"]}: {b["size"]}')
print()

# Test with max_num_tokens=15
layout15 = make_layout({'max_num_tokens': 15})
total15 = sum(b['size'] for b in layout15)
print('With max_num_tokens=15:')
print(f'  Total features: {total15}')
print()

print('Observation space sizes:')
print(f'  With max_tokens=1:  {total1} + 4 = {total1 + 4}')
print(f'  With max_tokens=15: {total15} + 4 = {total15 + 4}')
print(f'  With max_tokens=16: {total16} + 4 = {total16 + 4}')
print()

print('obs_builder fills (42 + max_num_tokens):')
print(f'  With max_tokens=1:  42 + 1 = 43')
print(f'  With max_tokens=15: 42 + 15 = 57')
print(f'  With max_tokens=16: 42 + 16 = 58')
