# Module 1: Python for Data Science

---

## 1. CONCEPT

This module covers Python as a **data science tool**, not general software engineering Python. The distinction matters: DS Python leans heavily on data structures for holding tabular/nested data, vectorized numeric operations (via NumPy), and rapid iterative exploration (via Jupyter) — rather than object-oriented design, packaging, or web frameworks.

Core building blocks you need fluent, no-lookup command of:

- **Data structures:** lists (ordered, mutable), dictionaries (key-value, the backbone of JSON-like data), tuples (immutable, used for fixed-size records), sets (uniqueness/membership checks)
- **Control flow:** `for`/`while` loops, `if`/`elif`/`else`, list comprehensions (Python's idiomatic way to build lists — used constantly in DS code)
- **Functions:** `def`, default arguments, `*args`/`**kwargs`, lambda functions (small anonymous functions, common inside `.apply()` and `.sort()`)
- **File I/O:** reading/writing CSV, JSON, and Excel — the actual entry point of almost every DS project
- **NumPy:** arrays as the foundation pandas is built on; vectorized operations (applying an operation to an entire array at once, avoiding slow Python loops); broadcasting (how NumPy handles operations between arrays of different shapes)

---

## 2. WHY IT MATTERS

Every pandas DataFrame is built on NumPy arrays underneath, and pandas' error messages, performance quirks, and API design only make sense once you understand what's happening at the NumPy level. Skipping straight to pandas without this foundation means you'll hit walls you can't debug — a `SettingWithCopyWarning`, a mysteriously slow `.apply()`, a broadcasting shape mismatch — and have no mental model for why they're happening.

Real-world data also rarely arrives as a single clean CSV. It arrives as a folder of inconsistent files, nested JSON from an API, or an Excel workbook with multiple sheets and merged header cells. Pure Python file-handling skills are what let you get that data into a workable state before pandas ever enters the picture.

---

## 3. SYNTAX REFERENCE

```python
# --- Data structures ---
prices = [19.99, 24.50, 9.99]                     # list
customer = {"id": 1042, "name": "A. Rivera", "tier": "gold"}  # dict
coordinates = (43.65, -79.38)                      # tuple (immutable)
unique_categories = {"electronics", "home", "toys"}  # set

# --- Control flow & comprehensions ---
discounted = [p * 0.9 for p in prices]             # list comprehension
high_value = [p for p in prices if p > 15]         # with filter

# --- Functions ---
def apply_discount(price, pct=0.1):                # default argument
    return price * (1 - pct)

get_price = lambda p: p * 1.1                      # lambda

# --- File I/O ---
import csv
import json

with open("sales.csv") as f:
    reader = csv.DictReader(f)
    rows = [row for row in reader]

with open("config.json") as f:
    data = json.load(f)

# --- NumPy ---
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
arr * 2                          # vectorized: [2, 4, 6, 8, 10] — no loop needed
arr.mean(), arr.std()            # built-in aggregate methods

matrix = np.array([[1, 2], [3, 4]])
matrix + np.array([10, 20])      # broadcasting: adds [10, 20] to every row
```

---

## 4. WORKED EXAMPLE

**Scenario:** You receive a folder of 5 daily sales export files from different regional store systems. They're inconsistent: different column orders, one file uses `"revenue"` while others use `"total_sales"`, and one file is missing a header row entirely. Before pandas is even introduced, standardize them using pure Python.

```python
import csv
import glob

# Mapping of known inconsistent column names to a standard schema
COLUMN_MAP = {
    "revenue": "total_sales",
    "total_sales": "total_sales",
    "store": "store_id",
    "store_id": "store_id",
    "date": "sale_date",
}

STANDARD_COLUMNS = ["sale_date", "store_id", "total_sales"]

def standardize_row(row, header):
    """Map an inconsistent row to the standard schema."""
    mapped = {}
    for original_col, value in zip(header, row):
        standard_col = COLUMN_MAP.get(original_col.strip().lower())
        if standard_col:
            mapped[standard_col] = value
    # Fill any missing standard columns with None so every row has the same shape
    return {col: mapped.get(col) for col in STANDARD_COLUMNS}

all_clean_rows = []

for filepath in glob.glob("raw_sales/*.csv"):
    with open(filepath) as f:
        reader = csv.reader(f)
        first_row = next(reader)

        # Detect whether the first row is a real header or actual data
        looks_like_header = any(
            cell.strip().lower() in COLUMN_MAP for cell in first_row
        )

        if looks_like_header:
            header = first_row
        else:
            # No header present — assume a known fallback order and
            # treat the first row as data, not a header
            header = ["date", "store", "revenue"]
            all_clean_rows.append(standardize_row(first_row, header))

        for row in reader:
            all_clean_rows.append(standardize_row(row, header))

print(f"Standardized {len(all_clean_rows)} rows from "
      f"{len(glob.glob('raw_sales/*.csv'))} files")
```

This is deliberately verbose — real data ingestion often is. The goal is recognizing the *pattern* (inconsistent schemas → mapping dictionary → standardized output), which you'll reuse constantly, including later when writing production ingestion pipelines.

---

## 5. COMMON MISTAKES

- **Using a Python `for` loop instead of a NumPy vectorized operation** for numeric work on arrays. `[x * 2 for x in my_list]` works but is far slower than `np.array(my_list) * 2` at any real scale — and this habit carries over painfully once you reach pandas.
- **Mutating a list while iterating over it.** Modifying a list's contents inside a `for` loop that's iterating over that same list produces silent, confusing bugs. Iterate over a copy (`for x in my_list[:]`) or build a new list instead.
- **Confusing `==` and `is`.** `==` checks value equality; `is` checks object identity. Using `is` to compare values (e.g., `if x is 5`) works by accident for small integers due to Python internals, then mysteriously breaks for larger numbers or other types.
- **Not closing files / not using `with`.** Always use `with open(...) as f:` — it auto-closes the file even if an error occurs mid-read. Manually calling `.close()` is easy to forget, especially inside a function with an early return.
- **Assuming CSV columns are always in the same order across files.** As shown in the worked example — always read the header and map by name, never by fixed position, unless you've explicitly verified every file matches.

---

## 6. PRACTICE EXERCISES

**Exercise 1 — Comprehensions & Filtering**
Given a list of order dictionaries, use a list comprehension to extract the `order_id` of every order where `status == "completed"` and `amount > 50`.

```python
orders = [
    {"order_id": 101, "status": "completed", "amount": 75.00},
    {"order_id": 102, "status": "cancelled", "amount": 30.00},
    {"order_id": 103, "status": "completed", "amount": 42.00},
    {"order_id": 104, "status": "completed", "amount": 120.00},
]
```

<details>
<summary>Hint</summary>
A list comprehension with an `if` clause: `[expr for item in iterable if condition]`. You can chain multiple conditions with `and`.
</details>

<details>
<summary>Solution</summary>

```python
result = [
    o["order_id"] for o in orders
    if o["status"] == "completed" and o["amount"] > 50
]
# result: [101, 104]
```
</details>

---

**Exercise 2 — NumPy Vectorization**
You have an array of daily temperatures in Celsius. Convert them to Fahrenheit using a vectorized NumPy operation (no loop), then find how many days exceeded 90°F.

```python
celsius = np.array([22, 35, 28, 40, 15, 33, 38])
```

<details>
<summary>Hint</summary>
Fahrenheit = Celsius × 9/5 + 32. To count elements meeting a condition, a boolean array (`arr > threshold`) sums to the count of `True` values, since `True` behaves as 1.
</details>

<details>
<summary>Solution</summary>

```python
fahrenheit = celsius * 9/5 + 32
days_over_90 = (fahrenheit > 90).sum()
# fahrenheit: [71.6, 95., 82.4, 104., 59., 91.4, 100.4]
# days_over_90: 4
```
</details>

---

**Exercise 3 — File Standardization (Applied)**
Extend the worked example's `standardize_row` function to also handle a file where the date column is named `"transaction_date"` instead of `"date"`. Then write a second function that flags and prints any row where `total_sales` couldn't be parsed as a float (i.e., is missing or malformed) — don't silently drop it, log it.

<details>
<summary>Hint</summary>
Add `"transaction_date": "sale_date"` to `COLUMN_MAP`. For the validation function, wrap the `float()` conversion in a `try`/`except ValueError` block and print the offending row when it fails, rather than letting the program crash or silently skipping it.
</details>

<details>
<summary>Sample Solution</summary>

```python
COLUMN_MAP["transaction_date"] = "sale_date"

def validate_sales_amount(rows):
    valid_rows = []
    for row in rows:
        try:
            row["total_sales"] = float(row["total_sales"])
            valid_rows.append(row)
        except (TypeError, ValueError):
            print(f"Flagged unparseable row: {row}")
    return valid_rows
```

The key habit here: never silently drop bad data. Log it so someone (including future you) can audit what was excluded and why.
</details>
