# Module 2: R for Statistical Computing

---

## 1. CONCEPT

R was built by statisticians, for statistics — and it still shows in ways that make certain DS tasks faster and more natural in R than in Python. This module builds parallel fluency, not competing fluency: the goal is to recognize the same operations you just learned in Module 1 and do them in R's idiom.

Core concepts:

- **Vectors** — R's fundamental data structure; almost everything in R is a vector operation under the hood, similar to how NumPy arrays underpin Python's numeric work
- **Data frames** — R's native tabular structure, the direct analog of a pandas DataFrame (R actually had this concept before pandas existed)
- **Factors** — R's representation of categorical variables, with an explicit set of levels — this has no exact Python equivalent and is a common source of confusion for Python-first learners
- **Base R vs. tidyverse** — Base R (built-in functions) is more verbose but has zero dependencies; the **tidyverse** (a collection of packages including `dplyr`, `tidyr`, `ggplot2`) provides a more consistent, readable grammar for data manipulation and is what you'll use for almost all practical work in this course
- **RStudio / Quarto** — RStudio is R's primary IDE; Quarto (successor to R Markdown) lets you combine code, output, and narrative text into a single reproducible document — critical for the "technical notebook + plain-English summary" habit from Module 0

---

## 2. WHY IT MATTERS

Many DS teams — especially in biostatistics, academic research, finance, and government — still run primarily on R, and job postings frequently list "Python and/or R" or specifically require R. Beyond employability, R's statistical modeling functions (`lm()`, `glm()`, hypothesis tests) return richer, more statistically-oriented output by default than their Python equivalents, which makes R genuinely faster for pure statistical exploration — one reason it hasn't been fully displaced by Python despite Python's dominance in ML/production.

Building this fluency now, in parallel with Python, is far easier than trying to bolt on R later after your mental models are Python-only.

---

## 3. SYNTAX REFERENCE

```r
# --- Vectors ---
prices <- c(19.99, 24.50, 9.99)          # <- is the traditional assignment operator
prices * 0.9                              # vectorized, like NumPy — no loop needed

# --- Data frames ---
orders <- data.frame(
  order_id = c(101, 102, 103),
  amount   = c(75.00, 30.00, 120.00),
  status   = c("completed", "cancelled", "completed")
)

# --- Factors (categorical variables) ---
tier <- factor(c("gold", "silver", "gold", "bronze"),
                levels = c("bronze", "silver", "gold"))  # explicit ordering

# --- dplyr (tidyverse) ---
library(dplyr)

orders %>%
  filter(status == "completed") %>%
  mutate(amount_with_tax = amount * 1.08) %>%
  arrange(desc(amount))

# The %>% (pipe) operator chains operations left to right —
# read it as "then do." R 4.1+ also supports a native pipe: |>

# --- Reading data ---
df <- read.csv("sales.csv")
library(readr)
df <- read_csv("sales.csv")   # tidyverse version: faster, more consistent types
```

---

## 4. WORKED EXAMPLE

**Scenario:** Reproduce Module 1's Python cleaning task in R — standardizing inconsistent sales files — using `dplyr`, to build a direct side-by-side reference.

```r
library(dplyr)
library(readr)
library(purrr)   # for iterating over multiple files
library(stringr)

# Standard column name mapping (same logic as the Python version)
column_map <- c(
  "revenue"      = "total_sales",
  "total_sales"  = "total_sales",
  "store"        = "store_id",
  "store_id"     = "store_id",
  "date"         = "sale_date",
  "transaction_date" = "sale_date"
)

standardize_file <- function(filepath) {
  df <- read_csv(filepath, show_col_types = FALSE)

  # Rename columns using the mapping, keeping only recognized columns
  names(df) <- str_to_lower(names(df))
  recognized_cols <- names(df)[names(df) %in% names(column_map)]

  df <- df %>%
    select(all_of(recognized_cols)) %>%
    rename_with(~ column_map[.x], all_of(recognized_cols))

  df
}

# Apply to every file in the folder and combine into one data frame
all_files <- list.files("raw_sales", pattern = "*.csv", full.names = TRUE)
clean_sales <- map_dfr(all_files, standardize_file)

glimpse(clean_sales)
```

Notice the structural parallel to the Python version: a mapping dictionary/named vector, a per-file standardization function, then a combine step (`map_dfr` here is R's equivalent of looping and concatenating). Once you see this pattern once, you'll recognize it in both languages permanently.

---

## 5. COMMON MISTAKES

- **Forgetting factors have a fixed level set.** Assigning a new category to a factor column that isn't in its existing `levels` silently produces `NA` instead of an error in older R versions — always check `levels()` before assuming a value made it through unchanged.
- **Confusing `=` and `<-`.** Both can assign in some contexts, but `<-` is the R convention and behaves more predictably, especially inside function calls (`=` inside a function call is argument-passing, not assignment) — use `<-` for assignment, always.
- **1-indexing surprises.** R vectors and data frames are 1-indexed (`df[1, ]` is the first row), unlike Python's 0-indexing. This trips up almost every Python-first learner at least once — internalize it early rather than debugging it under pressure later.
- **Piping into a function that doesn't expect the data frame as its first argument.** The `%>%` pipe passes the left side as the *first* argument of the right side by default. Some base R functions don't take a data frame first, which silently breaks the pipe chain — check function signatures when a pipe produces unexpected results.
- **Mixing base R subsetting (`df[df$status == "completed", ]`) and dplyr verbs inconsistently within the same script.** Both work, but pick one style per project for readability — mixing them is a common habit among people learning R via scattered Stack Overflow answers rather than a structured path.

---

## 6. PRACTICE EXERCISES

**Exercise 1 — dplyr Fluency**
Given the `orders` data frame from the syntax reference section, write a single `dplyr` pipe that: filters to orders with `amount > 40`, creates a new column `amount_category` that's `"high"` if amount > 100 and `"standard"` otherwise, and sorts by amount descending.

<details>
<summary>Hint</summary>
Use `filter()`, then `mutate()` with `ifelse(condition, value_if_true, value_if_false)`, then `arrange(desc(...))`. Chain them with `%>%`.
</details>

<details>
<summary>Solution</summary>

```r
orders %>%
  filter(amount > 40) %>%
  mutate(amount_category = ifelse(amount > 100, "high", "standard")) %>%
  arrange(desc(amount))
```
</details>

---

**Exercise 2 — Factors**
Create a factor for customer tier with levels ordered `"bronze" < "silver" < "gold" < "platinum"`. Then write code that would correctly identify which customers are tier `"gold"` or above, using the factor's inherent ordering (not string comparison).

<details>
<summary>Hint</summary>
Use `factor(x, levels = c(...), ordered = TRUE)` to create an ordered factor. Ordered factors support `>=` comparisons directly against their levels.
</details>

<details>
<summary>Solution</summary>

```r
tier <- factor(
  c("bronze", "gold", "platinum", "silver", "gold"),
  levels = c("bronze", "silver", "gold", "platinum"),
  ordered = TRUE
)

high_tier <- tier >= "gold"
# high_tier: FALSE  TRUE  TRUE FALSE  TRUE
```
</details>

---

**Exercise 3 — Python/R Parity (Reference Sheet Building)**
Take three operations you did in Module 1's Python exercises (list comprehension filtering, NumPy vectorized math, file standardization) and write the R equivalent for each, side by side. Save this as a personal "Python ↔ R" reference document — you'll keep extending it through the rest of the course, and it becomes a genuinely useful portfolio artifact showing dual-language fluency.

<details>
<summary>Hint</summary>
For the list comprehension filter: use `dplyr::filter()` or base R subsetting. For NumPy vectorized math: R vectors are vectorized by default, no library needed. For file standardization: you already built this in the worked example above — reuse it directly.
</details>
