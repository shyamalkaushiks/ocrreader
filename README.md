# Image tables → Excel (OpenAI Vision)

Reads images from a folder, extracts the main table with **OpenAI Vision**, groups by **column count**, and writes Excel output.

### How it works (per image — not “whole folder in one call”)

1. Images are taken in **sorted filename order** (predictable, e.g. `WA0000`, `WA0001`, ...).
2. **Each image** is sent to the API **once** → model returns `columnCount` + `rows`.
3. That block is **appended** to the group for that count only (e.g. 4 → 4-col group, 6 → 6-col group). Other groups are unchanged.
4. Order **inside** a group is the order those images appeared in the folder list (so image 5 might be 6 cols and image 6 might be 4 cols — each goes to its own bucket).

So: **decision per image → append to the matching column-count output** (separate files or separate sheets; see flags).

## Rules

- Images are processed in **sorted filename order**.
- Within the same column count: **first image** writes all rows; **later images** skip their **first row** (duplicate header). You can still fix headers manually in Excel.
- After each image block (except the last in that file), one **blank row** is inserted.
- Images that fail API/errors or have **no table** are **skipped** and logged to **stderr** with the filename.
- By default, after each successful image append, the program **saves an Excel snapshot immediately** (safer for rate-limit/interruption cases).

## Setup

### Option A — `.env` file (recommended)

1. Copy the example file:

```bash
cp .env.example .env
```

2. Edit `.env` and set `OPENAI_API_KEY=sk-...` (and optionally `OPENAI_MODEL`).

3. Run from the project folder so `godotenv` finds `.env`:

```bash
cd /Users/you/excelproject
go run .
```

`godotenv` **does not override** variables already set in your shell.

### Option B — shell only

```bash
export OPENAI_API_KEY="sk-..."
# optional; default is gpt-4o
export OPENAI_MODEL="gpt-4o"
```

## Run

From this directory:

```bash
go run . -dir "WhatsApp all images" -out output
```

### Quick test (4-column only, few images)

Still calls the API once per image in the batch — use **`-limit`** to keep the test small and fast:

```bash
go run . -dir "WhatsApp all images" -out output -only-cols 4 -limit 5
```

- **`-only-cols 4`** — Excel mein sirf **4 columns** wali extractions rakhta hai; 8 (ya aur) wali skip + log.
- **`-limit 5`** — sorted order me **sirf pehli 5 images** process (bill / time kam).

Flags:

- `-dir` — image folder (default: `WhatsApp all images`)
- `-out` — output folder for workbooks (default: `output`)
- `-limit` — max images to process, `0` = all (default `0`)
- `-offset` — skip first N sorted images, then process आगे वाले (default `0`)
- `-only-cols` — comma-separated counts, e.g. `4` or `4,8` (empty = keep every column count)
- `-single-xlsx` — one file `output/tables_by_column_count.xlsx` with tabs `4_cols`, `8_cols`, …
- `-v` — print each image decision: `filename -> N columns -> append to N_col group`
- `-flush-each-image` — save after each successful image append (default `true`)

## Output

**Default** (separate files):

- `output/tables_4_cols.xlsx`
- `output/tables_8_cols.xlsx`

**With `-single-xlsx`:**

- `output/tables_by_column_count.xlsx` — one workbook, one sheet per column count

## Cost note

Every image triggers one vision API call. Large folders can use significant tokens; test on a small subset first.
